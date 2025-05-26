import os
import glob
import copy
import torch
import numpy as np
from torch import nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from zoopt import Dimension, Objective, Parameter, Opt
from BackdoorAttack import prepare_datasets, CONFIG
import re
import time

class Modifier:
    def __init__(self, model, target_list, device):
        self.model = model
        self.target_list = target_list  # list of (layer_type, layer_idx, neuron, importance)
        self.device = device
        self.in_features_map = {}
        self.totaldim=0
        for i,(layer_type, lidx, neuron, _) in enumerate(self.target_list):
            key = (layer_type, lidx, neuron)
            if layer_type == 'classifier':
                self.in_features_map[key] = model.classifier[lidx].in_features
                self.totaldim+=model.classifier[lidx].in_features
            elif layer_type == 'features':
                self.in_features_map[key] = 1
                self.totaldim += 1
                #这里需要专门的理解 是整个代码的关键
                #搞清楚这里的channels是in还是out
                #搞清楚定位文件的一行究竟定位了几个参数
                #self.in_features_map[key]=model.features[lidx].out_channels



    def apply(self, model, values):
        #apply中的维度数增加方式要与上面的维度数产生方式对应 这里需要专门的理解 是整个代码的关键
        ptr = 0

        for i,(layer_type, lidx, neuron, _) in enumerate(self.target_list):

            key = (layer_type, lidx, neuron)
            if layer_type == 'classifier':
                dim = self.in_features_map[key]
                delta = torch.tensor(values[ptr:ptr+dim], device=self.device)
                with torch.no_grad():
                    model.classifier[lidx].weight[neuron,:] += delta
                ptr += dim
            elif layer_type == 'features':

                out_ch,in_ch, h, w = neuron
                #dim=self.in_features_map[key]
                dim=1
                delta = torch.tensor(values[ptr:ptr+dim], device=self.device)
                temp=model.features[lidx].weight
                temp1=model.features[lidx]
                with torch.no_grad():
                    model.features[lidx].weight[out_ch,in_ch,h,w] += delta[0]
                ptr += dim

        return model

    @property
    def total_dims(self):
        count=0
        for key in self.in_features_map:
            count+=1
        #为什么会遗漏???
        #为什么count!=140???
        return sum(self.in_features_map[key] for key in self.in_features_map)


class Evaluator:
    def __init__(self, base_model, modifier, clean_loader, poison_loader, device):
        self.base_model = base_model
        self.modifier = modifier
        self.clean_loader = clean_loader
        self.poison_loader = poison_loader
        self.device = device
        self.acc_clean_raw=100
        self.acc_poison_raw=100
        self.stepcount=0
        self.steppoint=0
        self.asr_target_from_stage1=0
        self.acc_clean_at_start_of_stage2=0
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

    def scoreCompute1(self,acc_clean,acc_poison):
        return -(-2*(self.acc_clean_raw-acc_clean)+(self.acc_poison_raw-acc_poison))+100

    def scoreCompute2(self, acc_clean, acc_poison):


        return -(-0.05*max(0,self.acc_clean_raw-acc_clean)**2+2*(self.acc_poison_raw-acc_poison))

    def scoreCompute3(self, acc_clean, acc_poison):
        self.stepcount+=1
        if self.stepcount<=10:
            objective_to_maximize = 2.0 * (self.acc_poison_raw - acc_poison) - 0.1 * max(0,self.acc_clean_raw - acc_clean)  # 轻微惩罚CA下降
            self.asr_target_from_stage1 = acc_poison
            self.acc_clean_at_start_of_stage2 = acc_clean
            self.steppoint=-objective_to_maximize
            return -objective_to_maximize
        elif self.stepcount>10:
            asr_rebound_penalty_factor = 2.0  # 对ASR反弹的惩罚系数
            ca_recovery_benefit_factor = 5.0  # 对CA恢复的奖励系数
            # 计算ASR反弹的惩罚
            asr_rebound = max(0, acc_poison - self.asr_target_from_stage1)  # ASR比目标值高了多少
            asr_penalty = -asr_rebound_penalty_factor * asr_rebound   # 平方惩罚ASR反弹
            # 计算CA恢复的奖励 (或者与原始CA的差距的惩罚减小)
            # 我们希望最大化 acc_clean，或者最小化 acc_clean_raw - acc_clean
            # 这里我们最大化 (acc_clean - acc_clean_at_start_of_stage2)
            # 或者，更简单地，直接用一个强调CA的目标，同时惩罚ASR
            # objective_to_maximize = 1.5 * acc_clean - 2.0 * max(0, acc_poison - (asr_target_from_stage1 + 5)) # 允许ASR比stage1结果高5%
            objective_to_maximize = ca_recovery_benefit_factor * (acc_clean - self.acc_clean_at_start_of_stage2) + asr_penalty
            return -objective_to_maximize+self.steppoint
        else:
            pass


        return 0

    def evaluate(self, solution):
        model_copy = copy.deepcopy(self.base_model)
        start=time.time()
        modified_model = self.modifier.apply(model_copy, solution)
        acc_clean = self.acc(modified_model, self.clean_loader)
        acc_poison = self.acc(modified_model, self.poison_loader)
        score = self.scoreCompute2(acc_clean,acc_poison)
        end=time.time()
        print(f"单次评估用时:{(end-start)} score:{score} acc_clean:{acc_clean} acc_poison:{acc_poison}")
        return score
    def evaluateRes(self, solution):
        model_copy = copy.deepcopy(self.base_model)
        modified_model = self.modifier.apply(model_copy, solution)
        acc_clean = self.acc(modified_model, self.clean_loader)
        acc_poison = self.acc(modified_model, self.poison_loader)
        score = -(1.0 * acc_clean - 1.0 * acc_poison)
        return acc_clean,acc_poison

def select_important_parameters(targets, top_k=10):
    # 根据importance进行排序，选出最重要的前top_k个参数
    sorted_targets = sorted(targets, key=lambda x: x[3], reverse=True)  # x[3] 是importance
    selected_targets = sorted_targets[:top_k]  # 选择前top_k个重要的参数
    return selected_targets


def generate_initial_samples(dim, num_samples=10, distribution='uniform', low=-1, high=1):
    """
    生成初始样本，返回符合给定维度和分布的样本。

    Parameters:
        dim (int): 每个样本的维度。
        num_samples (int): 初始样本数量，默认生成10个样本。
        distribution (str): 生成样本的分布类型 ('uniform' 或 'normal')。默认为 'uniform'。
        low (float): 均匀分布的下界，默认为 -1。
        high (float): 均匀分布的上界，默认为 1。

    Returns:
        np.ndarray: 生成的初始样本集。
    """
    if distribution == 'uniform':
        # 在 [low, high] 范围内生成均匀分布的样本
        init_samples = np.random.uniform(low=low, high=high, size=(num_samples, dim))
    elif distribution == 'normal':
        # 使用正态分布生成样本，均值为0，标准差为1
        init_samples = np.random.normal(loc=0.0, scale=1.0, size=(num_samples, dim))
    else:
        raise ValueError("Unsupported distribution type. Use 'uniform' or 'normal'.")

    return init_samples

def worker_init_fn(worker_id):
    np.random.seed(worker_id)  # 设置每个 worker 使用不同的随机种子

def repairbackdoor(arg):
    print("[INFO] Starting backdoor repair using RACOS...")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    CONFIG['dataset_name'] = arg.set
    CONFIG['bd_type'] = arg.bdtype
    CONFIG['architecture'] = arg.arch

    # === Load clean test data ===
    _, test_dataset, _, img_size = prepare_datasets(arg.set)
    CONFIG['img_size'] = img_size
    batch_size_use=32
    num_workers=0
    delta_bound=2
    clean_loader = DataLoader(test_dataset, batch_size=batch_size_use, shuffle=False,num_workers=num_workers,pin_memory=True,worker_init_fn=worker_init_fn)

    # === Determine transform for poisoned data to match clean data ===
    transform_list = []
    if arg.set in ['MNIST', 'CIFAR10']:
        transform_list.append(transforms.ToTensor())
    elif arg.set == 'IMAGENET10':
        transform_list.extend([
            # transforms.Resize(256),
            # transforms.CenterCrop(224),
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

    poison_dir = f"data/poisonedval_{arg.set}_{arg.arch}_{arg.bdtype}"
    poison_dataset = datasets.ImageFolder(poison_dir, transform=transforms.Compose(transform_list))
    poison_loader = DataLoader(poison_dataset, batch_size=batch_size_use, shuffle=False,num_workers=num_workers,pin_memory=True,worker_init_fn=worker_init_fn)

    # === Load model ===
    print("[INFO] Loading backdoored model...")
    model_path = f"transpace/{arg.set}_{arg.arch}_{arg.bdtype}_bd.pth"

    if arg.pretrainfile!=None:
        model_path=arg.pretrainfile

    if arg.arch == 'stdvgg16_class10':
        model = models.vgg16(pretrained=True)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, 10)
    elif arg.arch == 'resnet18_class10':
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 10)
    elif arg.arch == 'resnet34_class10':
        model = models.resnet34(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 10)
    else:
        raise NotImplementedError(f"Unsupported architecture: {arg.arch}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    # === Load fault localization results (use third line for layer info) ===
    print("[INFO] Parsing fault localization files...")
    fl_files = glob.glob(os.path.join(arg.fldir, "*.txt"))
    targets = []
    targets_conv=[]
    targets_fc=[]
    for f in fl_files:
        with open(f, 'r') as fin:
            lines = fin.readlines()
            try:
                # 提取第3行中的层名和层索引
                layer_line = lines[2].strip()

                # 使用正则表达式从文本中提取层名（括号内的字符串）和层索引（冒号后的数字）
                match = re.search(r'.*\(model\.(.*?)\sindex\)\s*:\s*(\d+)', layer_line)
                if match:
                    layer_type = match.group(1)  # 提取括号中的层名
                    layer_idx = int(match.group(2))  # 提取层索引
                else:
                    print(f"[WARN] Unexpected format in third line of {f}")
                    continue

            except Exception as e:
                print(f"[WARN] Failed to parse layer info from {f}: {e}")
                continue


            for line in lines:
                # 匹配卷积层：提取 (OutCh, InCh, KPH, KPW)
                # 对于卷积层：提取 (OutCh, InCh, KPH, KPW)
                match_conv = re.match(r'.*\((\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\).*', line)

                # 对于全连接层：提取单一神经元索引
                match_fc = re.match(r'.*\(\s*(\d+)\s*\).*', line)

                if match_conv and match_fc:
                    raise ValueError(
                        f"Line '{line.strip()}' cannot match both convolution and fully connected layer patterns.")

                # 匹配卷积层
                elif match_conv:
                    out_ch = int(match_conv.group(1))
                    in_ch = int(match_conv.group(2))
                    kph = int(match_conv.group(3))
                    kpw = int(match_conv.group(4))
                    neuron = (out_ch, in_ch, kph, kpw)  # 卷积层的神经元索引
                    importance = float(line.strip().split()[-1])
                    targets.append((layer_type, layer_idx, neuron, importance))
                    targets_conv.append((layer_type, layer_idx, neuron, importance))
                # 匹配全连接层
                elif match_fc:
                    neuron = int(match_fc.group(1))  # 获取 FC 神经元索引
                    importance = float(line.strip().split()[-1])  # 提取重要性
                    targets.append((layer_type, layer_idx, neuron, importance))
                    targets_fc.append((layer_type, layer_idx, neuron, importance))

    selected_targets_conv = select_important_parameters(targets_conv, top_k=len(targets_conv))  # 选择重要性最高的10个参数
    selected_targets_fc = select_important_parameters(targets_fc, top_k=len(targets_fc))  # 选择重要性最高的10个参数

    targets=[]
    targets=selected_targets_fc+selected_targets_conv
    print(f"[INFO] Total {len(targets)} neurons or edges to repair")
    for t in targets:
        print(f" - Layer: {t[0]}[{t[1]}], Index: {t[2]}, Importance: {t[3]:.4f}")

    # === Define modifier and evaluator ===
    modifier = Modifier(model, targets, device)

    print("[DEBUG] Repair targets and inferred parameter dimensions:")
    for k, v in modifier.in_features_map.items():
        print(f"  - Target: {k}, Dimensions: {v}")
    print(f"[DEBUG] Total repair dimension: {modifier.totaldim}")

    if modifier.totaldim == 0:
        print("[ERROR] No repairable parameters found. Please check the fault localization results.")
        return

    evaluator = Evaluator(model, modifier, clean_loader, poison_loader, device)
    evaluator.acc_clean_raw, evaluator.acc_poison_raw = evaluator.evaluateRes(np.zeros(modifier.totaldim))
    print(f"acc_clean_raw:{evaluator.acc_clean_raw},acc_poison_raw:{evaluator.acc_poison_raw}")
    dim = Dimension(modifier.totaldim, [[-delta_bound, delta_bound]] * modifier.totaldim, [True] * modifier.totaldim)
    obj = Objective(lambda sol: evaluator.evaluate(sol.get_x()), dim)
    num_samples=10
    init_samples = generate_initial_samples(modifier.totaldim, num_samples, distribution='uniform')
    param = Parameter(budget=1000, init_samples=[np.zeros(modifier.totaldim)])

    opt = Opt()

    print("[INFO] Launching RACOS optimization...")
    best = opt.min(obj, param)

    best_model = modifier.apply(copy.deepcopy(model), best.get_x())
    save_path = f'data/defended_model_{arg.set}_{arg.arch}_{arg.bdtype}.pth'
    torch.save(best_model.state_dict(), save_path)

    print(list(best.get_x()))
    acc_clean_rep, acc_poison_rep=evaluator.evaluateRes(best.get_x())
    print(f"acc_clean_raw:{evaluator.acc_clean_raw} acc_poison_raw:{evaluator.acc_poison_raw}")
    print(f"acc_clean_repaired:{acc_clean_rep} acc_poison_repaired:{acc_poison_rep}")
    print(f"[✔] Repair completed. Model saved to {save_path}")
