import os
import glob
import copy
import torch
import numpy as np
from torch import nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from zoopt import Dimension, Objective, Parameter, Opt, Solution # 引入Solution
from BackdoorAttack import prepare_datasets, CONFIG # 假设CONFIG在BackdoorAttack.py中
import re
import time

class Modifier:
    def __init__(self, model, target_list, device):
        self.model = model
        self.target_list = target_list
        self.device = device
        self.in_features_map = {}
        self.totaldim = 0
        for i, (layer_type, lidx, neuron, _) in enumerate(self.target_list):
            key = (layer_type, lidx, neuron)
            if layer_type == 'classifier':
                self.in_features_map[key] = model.classifier[lidx].in_features
                self.totaldim += model.classifier[lidx].in_features
            elif layer_type == 'features':
                # 假设每个卷积核参数元组只修改一个权重
                self.in_features_map[key] = 1
                self.totaldim += 1

    def apply(self, model, values):
        ptr = 0
        for i, (layer_type, lidx, neuron, _) in enumerate(self.target_list):
            key = (layer_type, lidx, neuron)
            if layer_type == 'classifier':
                dim = self.in_features_map[key]
                delta = torch.tensor(values[ptr:ptr+dim], device=self.device, dtype=model.classifier[lidx].weight.dtype)
                with torch.no_grad():
                    model.classifier[lidx].weight[neuron,:] += delta
                ptr += dim
            elif layer_type == 'features':
                out_ch, in_ch, h, w = neuron
                dim = 1
                delta = torch.tensor(values[ptr:ptr+dim], device=self.device, dtype=model.features[lidx].weight.dtype)
                with torch.no_grad():
                    model.features[lidx].weight[out_ch,in_ch,h,w] += delta[0]
                ptr += dim
        return model

    @property
    def total_dims(self):
        return self.totaldim


class Evaluator:
    def __init__(self, base_model, modifier, clean_loader, poison_loader, device):
        self.base_model = base_model
        self.modifier = modifier
        self.clean_loader = clean_loader
        self.poison_loader = poison_loader
        self.device = device
        self.acc_clean_raw = 0 # 将在外部计算并设置
        self.acc_poison_raw = 0 # 将在外部计算并设置
        self.current_stage = 1 # 默认为阶段一

    def acc(self, model, loader):
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                pred = model(x).argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        return correct / total * 100

    def set_stage(self, stage):
        self.current_stage = stage
        print(f"[INFO] Evaluator set to Stage {self.current_stage}")

    def score_stage1(self, acc_clean, acc_poison):
        # 阶段一：主要目标是降低ASR (acc_poison)，对CA下降的容忍度较高
        # objective_to_maximize = -0.05 * max(0, self.acc_clean_raw - acc_clean)**2 + 2.0 * (self.acc_poison_raw - acc_poison)
        # 或者更激进地只关注ASR下降：
        objective_to_maximize = 2.0 * (self.acc_poison_raw - acc_poison) - 0.1 * max(0, self.acc_clean_raw - acc_clean) # 轻微惩罚CA下降
        print(f"Stage 1 Metrics: CA_raw={self.acc_clean_raw:.2f}, CA={acc_clean:.2f}, ASR_raw={self.acc_poison_raw:.2f}, ASR={acc_poison:.2f}")
        print(f"Stage 1 Score Components: CA_penalty_term={-0.1 * max(0, self.acc_clean_raw - acc_clean):.2f}, ASR_benefit_term={2.0 * (self.acc_poison_raw - acc_poison):.2f}")
        return -objective_to_maximize # RACOS最小化

    def score_stage2(self, acc_clean, acc_poison, asr_target_from_stage1):
        # 阶段二：主要目标是恢复CA，同时保持ASR不大幅反弹 (或维持在stage1的目标附近)
        # ASR不应高于 stage1 优化后的asr_target_from_stage1 太多 (例如，允许少量反弹)
        asr_rebound_penalty_factor = 2.0 # 对ASR反弹的惩罚系数
        ca_recovery_benefit_factor = 1.0 # 对CA恢复的奖励系数

        # 计算ASR反弹的惩罚
        asr_rebound = max(0, acc_poison - asr_target_from_stage1) # ASR比目标值高了多少
        asr_penalty = -asr_rebound_penalty_factor * asr_rebound**2 # 平方惩罚ASR反弹

        # 计算CA恢复的奖励 (或者与原始CA的差距的惩罚减小)
        # 我们希望最大化 acc_clean，或者最小化 acc_clean_raw - acc_clean
        # 这里我们最大化 (acc_clean - acc_clean_at_start_of_stage2)
        # 或者，更简单地，直接用一个强调CA的目标，同时惩罚ASR
        # objective_to_maximize = 1.5 * acc_clean - 2.0 * max(0, acc_poison - (asr_target_from_stage1 + 5)) # 允许ASR比stage1结果高5%
        objective_to_maximize = ca_recovery_benefit_factor * (acc_clean - self.acc_clean_at_start_of_stage2) + asr_penalty
        # 或者，一个更像原始的，但权重调整过的
        # objective_to_maximize = -1.0 * max(0, self.acc_clean_raw - acc_clean)**2 + 0.5 * (self.acc_poison_raw - acc_poison) \
        #                         - 2.0 * max(0, acc_poison - (asr_target_from_stage1 + 2)) # 惩罚ASR高于目标值+2%

        print(f"Stage 2 Metrics: CA_raw={self.acc_clean_raw:.2f}, CA_start_S2={self.acc_clean_at_start_of_stage2:.2f}, CA={acc_clean:.2f}, ASR_raw={self.acc_poison_raw:.2f}, ASR_target_S1={asr_target_from_stage1:.2f}, ASR={acc_poison:.2f}")
        print(f"Stage 2 Score Components: CA_benefit_term={(ca_recovery_benefit_factor * (acc_clean - self.acc_clean_at_start_of_stage2)):.2f}, ASR_penalty_term={asr_penalty:.2f}")
        return -objective_to_maximize

    def evaluate(self, solution_array, asr_target_from_stage1=None): # solution应为numpy array
        model_copy = copy.deepcopy(self.base_model)
        start_time = time.time()
        modified_model = self.modifier.apply(model_copy, solution_array)
        acc_clean = self.acc(modified_model, self.clean_loader)
        acc_poison = self.acc(modified_model, self.poison_loader) # 假设acc_poison是ASR

        if self.current_stage == 1:
            score = self.score_stage1(acc_clean, acc_poison)
        elif self.current_stage == 2:
            if asr_target_from_stage1 is None:
                raise ValueError("asr_target_from_stage1 must be provided for Stage 2 evaluation")
            score = self.score_stage2(acc_clean, acc_poison, asr_target_from_stage1)
        else:
            raise ValueError(f"Unknown stage: {self.current_stage}")

        end_time = time.time()
        print(f"Stage {self.current_stage} Eval Time: {end_time - start_time:.2f}s, Score: {score:.4f}, CA: {acc_clean:.2f}%, ASR: {acc_poison:.2f}%")
        return score

    def evaluateRes(self, solution_array):
        model_copy = copy.deepcopy(self.base_model)
        modified_model = self.modifier.apply(model_copy, solution_array)
        acc_clean = self.acc(modified_model, self.clean_loader)
        acc_poison = self.acc(modified_model, self.poison_loader)
        return acc_clean, acc_poison

def select_important_parameters(targets, top_k=10):
    sorted_targets = sorted(targets, key=lambda x: x[3], reverse=True)
    return sorted_targets[:top_k]

def repairbackdoor(arg):
    print("[INFO] Starting backdoor repair using Staged RACOS...")
    start_total_time = time.time()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # ... (CONFIG设置 和 数据加载部分与你原代码类似，确保poison_loader的准确率代表ASR) ...
    CONFIG['dataset_name'] = arg.set
    CONFIG['bd_type'] = arg.bdtype
    CONFIG['architecture'] = arg.arch

    # === Load clean test data ===
    _, test_dataset, _, img_size = prepare_datasets(arg.set)
    CONFIG['img_size'] = img_size
    batch_size_use=32
    num_workers=2
    delta_bound_stage1 = arg.delta_bound_s1 # 从arg获取阶段1的delta_bound
    delta_bound_stage2 = arg.delta_bound_s2 # 从arg获取阶段2的delta_bound


    clean_loader = DataLoader(test_dataset, batch_size=batch_size_use, shuffle=False,num_workers=num_workers)

    # === Determine transform for poisoned data to match clean data ===
    # ... (与你原代码相同)
    transform_list = []
    if arg.set in ['MNIST', 'CIFAR10']:
        transform_list.append(transforms.ToTensor())
    elif arg.set == 'IMAGENET10':
        transform_list.extend([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
    elif arg.set == 'GTRSB':
        transform_list.extend([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])
    else:
        raise ValueError(f"Unsupported dataset: {arg.set}")

    # Apply normalization
    if arg.set == 'MNIST':
        mean, std = (0.1307,), (0.3081,)
    elif arg.set == 'CIFAR10':
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    elif arg.set == 'IMAGENET10':
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    elif arg.set == 'GTRSB':
        mean, std = [0.3403, 0.3121, 0.3214], [0.2724, 0.2608, 0.2669]

    transform_list.append(transforms.Normalize(mean, std))

    poison_dir = f"data/poisoned_{arg.set}_{arg.arch}_{arg.bdtype}"
    poison_dataset = datasets.ImageFolder(poison_dir, transform=transforms.Compose(transform_list))
    poison_loader = DataLoader(poison_dataset, batch_size=batch_size_use, shuffle=False,num_workers=num_workers)


    # === Load model ===
    # ... (与你原代码相同) ...
    print("[INFO] Loading backdoored model...")
    model_path = f"transpace/{arg.set}_{arg.arch}_{arg.bdtype}_bd.pth"
    if arg.arch == 'stdvgg16_class10':
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1 if arg.pretrained else None)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, 10)
    elif arg.arch == 'resnet18_class10':
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if arg.pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, 10)
    elif arg.arch == 'resnet34_class10':
        model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1 if arg.pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, 10)
    else:
        raise NotImplementedError(f"Unsupported architecture: {arg.arch}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    # === Load fault localization results ===
    # ... (与你原代码相同，注意targets的选择可能也需要分阶段或统一处理) ...
    print("[INFO] Parsing fault localization files...")
    fl_files = glob.glob(os.path.join(arg.fldir, "*.txt"))
    targets_all = [] # Store all parsed targets first
    for f in fl_files:
        with open(f, 'r') as fin:
            lines = fin.readlines()
            layer_type_file, layer_idx_file = None, None
            try:
                layer_line = lines[2].strip()
                match_layer = re.search(r'.*\(model\.(.*?)\sindex\)\s*:\s*(\d+)', layer_line)
                if match_layer:
                    layer_type_file = match_layer.group(1)
                    layer_idx_file = int(match_layer.group(2))
                else:
                    print(f"[WARN] Unexpected format in third line of {f}")
                    continue
            except Exception as e:
                print(f"[WARN] Failed to parse layer info from {f}: {e}")
                continue

            for line_num, line_content in enumerate(lines):
                if line_num <= 2: continue # Skip header lines for neuron/edge parsing

                match_conv = re.match(r'.*\((\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\).*', line_content)
                match_fc = re.match(r'.*\(\s*(\d+)\s*\).*', line_content)

                if match_conv:
                    neuron = (int(match_conv.group(1)), int(match_conv.group(2)), int(match_conv.group(3)), int(match_conv.group(4)))
                    importance = float(line_content.strip().split()[-1])
                    targets_all.append((layer_type_file, layer_idx_file, neuron, importance))
                elif match_fc:
                    neuron = int(match_fc.group(1))
                    importance = float(line_content.strip().split()[-1])
                    targets_all.append((layer_type_file, layer_idx_file, neuron, importance))

    # 选择用于修复的参数 (这里简化为选择全部，你可以按需调整top_k)
    targets = select_important_parameters(targets_all, top_k=arg.top_k_FL if hasattr(arg, 'top_k_FL') else len(targets_all))
    print(f"[INFO] Total {len(targets)} neurons or edges selected for repair")
    if not targets:
        print("[ERROR] No targets selected for repair. Exiting.")
        return

    modifier = Modifier(model, targets, device)
    print(f"[DEBUG] Total repair dimension: {modifier.totaldim}")
    if modifier.totaldim == 0:
        print("[ERROR] No repairable parameters found (totaldim is 0). Exiting.")
        return

    evaluator = Evaluator(model, modifier, clean_loader, poison_loader, device)
    # 获取原始模型的准确率
    evaluator.acc_clean_raw, evaluator.acc_poison_raw = evaluator.evaluateRes(np.zeros(modifier.totaldim))
    print(f"[INFO] Original Model: CA_raw={evaluator.acc_clean_raw:.2f}%, ASR_raw={evaluator.acc_poison_raw:.2f}%")

    # --- Stage 1: Focus on ASR reduction ---
    print("\n[INFO] --- Starting Stage 1 Optimization (Focus on ASR reduction) ---")
    evaluator.set_stage(1)
    dim_stage1 = Dimension(modifier.totaldim, [[-delta_bound_stage1, delta_bound_stage1]] * modifier.totaldim, [True] * modifier.totaldim)
    # 传递evaluator.evaluate，但需要用lambda包装solution对象
    obj_stage1 = Objective(lambda sol: evaluator.evaluate(sol.get_x()), dim_stage1)
    
    # 初始样本可以更丰富些
    init_samples_s1 = [np.zeros(modifier.totaldim)]
    if modifier.totaldim > 0:
        for _ in range(min(4, modifier.totaldim)): # 添加几个随机扰动的小样本
             init_samples_s1.append(np.random.uniform(-delta_bound_stage1/3, delta_bound_stage1/3, modifier.totaldim))

    param_stage1 = Parameter(budget=arg.budget_s1, init_samples=init_samples_s1) # arg.budget_s1 为阶段1预算
    opt_stage1 = Opt()
    best_sol_stage1_obj = opt_stage1.min(obj_stage1, param_stage1)
    best_params_stage1 = best_sol_stage1_obj.get_x()

    # 评估阶段1结果
    ca_stage1, asr_stage1 = evaluator.evaluateRes(best_params_stage1)
    print(f"[INFO] Stage 1 Best Params: {list(np.round(best_params_stage1, 4))}")
    print(f"[INFO] Stage 1 Result: CA={ca_stage1:.2f}%, ASR={asr_stage1:.2f}%")

    # --- Stage 2: Focus on CA recovery, maintain ASR ---
    print("\n[INFO] --- Starting Stage 2 Optimization (Focus on CA recovery) ---")
    evaluator.set_stage(2)
    evaluator.acc_clean_at_start_of_stage2 = ca_stage1 # 记录阶段2开始时的CA，用于评估改善
    
    # Stage 2 的 delta_bound 应该作用于 Stage 1 结果的基础上进行微调
    # 所以 Stage 2 优化的 delta 是相对于 Stage 1 修改后的模型的 delta'
    # Dimension 仍然是 delta' 的范围
    dim_stage2 = Dimension(modifier.totaldim, [[-delta_bound_stage2, delta_bound_stage2]] * modifier.totaldim, [True] * modifier.totaldim)
    
    # 目标函数需要知道阶段1的目标ASR
    # 注意：这里的sol.get_x()是第二阶段的delta_prime
    # 评估时，需要将第一阶段的best_params_stage1 和 第二阶段的delta_prime 叠加起来
    obj_stage2 = Objective(
        lambda sol_s2: evaluator.evaluate(
            best_params_stage1 + sol_s2.get_x(), # 关键：叠加两阶段的修改
            asr_target_from_stage1=asr_stage1
        ),
        dim_stage2
    )
    
    # Stage 2 的初始样本应该是全零（代表在Stage 1结果上不进一步修改）
    # 也可以加上一些基于全零的小扰动
    init_samples_s2 = [np.zeros(modifier.totaldim)]
    if modifier.totaldim > 0:
         for _ in range(min(4, modifier.totaldim)):
             init_samples_s2.append(np.random.uniform(-delta_bound_stage2/3, delta_bound_stage2/3, modifier.totaldim))


    param_stage2 = Parameter(budget=arg.budget_s2, init_samples=init_samples_s2) # arg.budget_s2 为阶段2预算
    opt_stage2 = Opt()
    best_sol_stage2_obj = opt_stage2.min(obj_stage2, param_stage2)
    best_delta_prime_stage2 = best_sol_stage2_obj.get_x()

    # 最终的参数修改是两阶段的总和
    final_best_params = best_params_stage1 + best_delta_prime_stage2

    # 应用最终修复并评估
    final_model = modifier.apply(copy.deepcopy(model), final_best_params)
    save_path = f'data/defended_model_staged_{arg.set}_{arg.arch}_{arg.bdtype}.pth'
    torch.save(final_model.state_dict(), save_path)

    ca_final, asr_final = evaluator.evaluateRes(final_best_params)
    print(f"\n[INFO] Final Best Params (S1+S2): {list(np.round(final_best_params,4))}")
    print(f"[INFO] Original Model: CA_raw={evaluator.acc_clean_raw:.2f}%, ASR_raw={evaluator.acc_poison_raw:.2f}%")
    print(f"[INFO] After Stage 1 : CA={ca_stage1:.2f}%, ASR={asr_stage1:.2f}%")
    print(f"[INFO] After Stage 2 (Final): CA={ca_final:.2f}%, ASR={asr_final:.2f}%")
    print(f"[✔] Staged repair completed. Model saved to {save_path}")
    end_total_time = time.time()
    print(f"[INFO] Total execution time: {end_total_time - start_total_time:.2f}s")

# 假设的arg对象，你需要根据你的实际命令行参数来配置
class Args:
    pass

if __name__ == '__main__':
    args = Args()
    args.set = 'CIFAR10' # 'MNIST', 'IMAGENET10', 'GTRSB'
    args.arch = 'resnet18_class10' # 'stdvgg16_class10', 'resnet34_class10'
    args.bdtype = 'BadNets' # 举例
    args.fldir = 'path/to/your/fault/localization/results' # **需要替换为你的FL结果路径**
    args.pretrained = False # 是否加载ImageNet预训练权重（如果模型原始训练时用了）
    args.top_k_FL = 140 # 选择多少个FL定位的参数进行修复，可以设小一点开始实验
    
    # 分阶段优化的新参数
    args.budget_s1 = 50 # 阶段一的预算
    args.budget_s2 = 50 # 阶段二的预算
    args.delta_bound_s1 = 3.0 # 阶段一的参数修改范围
    args.delta_bound_s2 = 1.0 # 阶段二的参数修改范围（通常更小，进行微调）

    if not os.path.exists(args.fldir):
        print(f"[ERROR] Fault localization directory not found: {args.fldir}")
        print("Please create dummy FL files or provide a valid path to run this example.")
        # 创建一些dummy FL文件以便测试
        os.makedirs(args.fldir, exist_ok=True)
        dummy_fl_content_fc = """Layer Name: model.classifier
Layer Index: 6
Layer Info (model.classifier index) : 6
(0)       algumas métricas importantes     0.9
(1)      outras métricas     0.8
"""
        dummy_fl_content_conv = """Layer Name: model.features
Layer Index: 0
Layer Info (model.features index) : 0
(0,0,0,0)      conv weight 1     0.7
(1,0,1,1)      conv weight 2     0.6
"""
        with open(os.path.join(args.fldir, "dummy_fc_fl.txt"), "w") as f:
            f.write(dummy_fl_content_fc)
        with open(os.path.join(args.fldir, "dummy_conv_fl.txt"), "w") as f:
            f.write(dummy_fl_content_conv)
        print(f"Created dummy FL files in {args.fldir} for testing.")


    # 运行修复
    repairbackdoor(args)