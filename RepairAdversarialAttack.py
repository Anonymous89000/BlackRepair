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
from model.inner_vgg import VGG16_dense
from model.inner_vgg import VGG13_dense
import cma

class Modifier:
    def __init__(self, model, target_list, device,arch):
        self.model = model
        self.target_list = target_list  # list of (layer_type, layer_idx, neuron, importance)
        self.device = device
        self.in_features_map = {}
        self.totaldim=0
        self.arch=arch
        if self.arch=='stdvgg16_class10':
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
        elif self.arch=='innervgg13':
            for i,(layer_type, lidx, neuron, _) in enumerate(self.target_list):
                key = (layer_type, lidx, neuron)

                if 'dense' in layer_type:
                    exec_scope = {"model": model}
                    tar = "in_features_tmp"
                    # 将 exec 的结果写入 exec_scope
                    exec(f"{tar} = model.{layer_type}.in_features", globals(), exec_scope)
                    # 从字典中提取值
                    in_features_tmp = exec_scope[tar]
                    self.in_features_map[key] = in_features_tmp
                    self.totaldim+=in_features_tmp
                elif 'features' in layer_type:
                    self.in_features_map[key] = 1
                    self.totaldim += 1
                    #这里需要专门的理解 是整个代码的关键
                    #搞清楚这里的channels是in还是out
                    #搞清楚定位文件的一行究竟定位了几个参数
                    #self.in_features_map[key]=model.features[lidx].out_channels
        elif self.arch == 'resnet18_class43':
            for i,(layer_type, lidx, neuron, _) in enumerate(self.target_list):
                key = (layer_type, lidx, neuron)
                if 'fc' in layer_type:
                    exec_scope = {"model": model}
                    tar = "in_features_tmp"
                    # 将 exec 的结果写入 exec_scope
                    exec(f"{tar} = model.{layer_type}.in_features", globals(), exec_scope)
                    # 从字典中提取值
                    in_features_tmp = exec_scope[tar]
                    self.in_features_map[key] = in_features_tmp
                    self.totaldim+=in_features_tmp
                elif 'conv' in layer_type:
                    self.in_features_map[key] = 1
                    self.totaldim += 1
                    #这里需要专门的理解 是整个代码的关键
                    #搞清楚这里的channels是in还是out
                    #搞清楚定位文件的一行究竟定位了几个参数
                    #self.in_features_map[key]=model.features[lidx].out_channels
            pass



    def apply(self, model, values):
        #apply中的维度数增加方式要与上面的维度数产生方式对应 这里需要专门的理解 是整个代码的关键
        ptr = 0
        arch=self.arch
        if arch=='stdvgg16_class10':
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
        elif arch=='innervgg13':
            for i,(layer_type, lidx, neuron, _) in enumerate(self.target_list):
                key = (layer_type, lidx, neuron)
                if 'dense' in layer_type:
                    dim = self.in_features_map[key]
                    delta = torch.tensor(values[ptr:ptr+dim], device=self.device)
                    exec_scope = {
                        "torch": torch,  # 注入 torch 模块
                        "model": model,  # 注入模型对象
                        "lidx": lidx,  # 分类器层索引
                        "neuron": neuron,  # 目标神经元索引
                        "delta": delta  # 权重增量值
                    }

                    # 2. 构建动态代码字符串

                    code_str =f"with torch.no_grad():\n\tmodel.{layer_type}.weight[neuron, :] += delta"

                    exec(code_str, globals(), exec_scope)

                    # with torch.no_grad():
                    #     model.classifier[lidx].weight[neuron,:] += delta
                    ptr += dim
                elif 'features' in layer_type:
                    out_ch,in_ch, h, w = neuron
                    #dim=self.in_features_map[key]
                    dim=1
                    delta = torch.tensor(values[ptr:ptr+dim], device=self.device)

                    exec_scope = {
                        "torch": torch,  # 注入 torch 模块
                        "model": model,  # 注入模型对象
                        "lidx": lidx,  # 分类器层索引
                        "neuron": neuron,  # 目标神经元索引
                        "delta": delta,  # 权重增量值
                        "out_ch": out_ch,
                        "in_ch": in_ch,
                        "h": h,
                        "w": w
                    }

                    # 2. 构建动态代码字符串
                    code_str = f"with torch.no_grad():\n\tmodel.{layer_type}[lidx].weight[out_ch,in_ch,h,w] += delta[0]"

                    exec(code_str, globals(), exec_scope)
                    # with torch.no_grad():
                    #     model.features[lidx].weight[out_ch,in_ch,h,w] += delta[0]
                    ptr += dim
        elif arch=='resnet18_class43':
            for i,(layer_type, lidx, neuron, _) in enumerate(self.target_list):
                key = (layer_type, lidx, neuron)
                if 'fc' in layer_type:
                    dim = self.in_features_map[key]
                    delta = torch.tensor(values[ptr:ptr+dim], device=self.device)
                    exec_scope = {
                        "torch": torch,  # 注入 torch 模块
                        "model": model,  # 注入模型对象
                        "lidx": lidx,  # 分类器层索引
                        "neuron": neuron,  # 目标神经元索引
                        "delta": delta  # 权重增量值
                    }

                    # 2. 构建动态代码字符串

                    code_str =f"with torch.no_grad():\n\tmodel.{layer_type}.weight[neuron, :] += delta"

                    exec(code_str, globals(), exec_scope)

                    # with torch.no_grad():
                    #     model.classifier[lidx].weight[neuron,:] += delta
                    ptr += dim
                elif 'conv' in layer_type:
                    out_ch,in_ch, h, w = neuron
                    #dim=self.in_features_map[key]
                    dim=1
                    delta = torch.tensor(values[ptr:ptr+dim], device=self.device)

                    exec_scope = {
                        "torch": torch,  # 注入 torch 模块
                        "model": model,  # 注入模型对象
                        "lidx": lidx,  # 分类器层索引
                        "neuron": neuron,  # 目标神经元索引
                        "delta": delta  ,# 权重增量值
                        "out_ch":out_ch,
                        "in_ch": in_ch,
                        "h":h ,
                        "w":w
                    }

                    # 2. 构建动态代码字符串
                    code_str = f"with torch.no_grad():\n\tmodel.{layer_type}.weight[out_ch,in_ch,h,w] += delta[0]"
                    exec(code_str, globals(), exec_scope)
                    # with torch.no_grad():
                    #     model.features[lidx].weight[out_ch,in_ch,h,w] += delta[0]
                    ptr += dim

        return model

    @property
    def total_dims(self):
        #该函数存在一些问题 没有达到预期 仍未查明
        #问题表现为 total_dims!=totaldim
        count=0
        for key in self.in_features_map:
            count+=1
        #为什么会遗漏???
        #为什么count!=140???
        return sum(self.in_features_map[key] for key in self.in_features_map)


# ==============================================================================
# Part 1: ActivationHook 帮助类 (用于捕获中间层激活值)
# ==============================================================================
class ActivationHook:
    """
    一个简单的hook管理器，用于捕获和访问指定层的激活值。
    """

    def __init__(self):
        self.activations = {}
        self.handles = []

    def register_hook(self, model, target_list,arch):
        layers_to_hook = set()
        for layer_type, lidx, _, _ in target_list:
            layers_to_hook.add((layer_type, lidx))

        if arch=='stdvgg16_class10':
            for layer_type, lidx in layers_to_hook:
                if layer_type == 'features':
                    layer = model.features[lidx]
                elif layer_type == 'classifier':
                    layer = model.classifier[lidx]
                else:
                    continue

                key = (layer_type, lidx)
                hook_fn = lambda module, input, output, k=key: self.activations.update({k: output.detach().cpu()})
                handle = layer.register_forward_hook(hook_fn)
                self.handles.append(handle)
        elif arch=='innervgg13':
            for layer_type, lidx in layers_to_hook:
                if 'dense' in layer_type:
                    exec_scope = {
                        "model": model,  # 注入模型对象
                        "lidx": lidx,  # 分类器层索引
                    }
                    # 2. 构建动态代码字符串
                    code_str = f"layer=model.{layer_type}"
                    exec(code_str, globals(), exec_scope)
                    layer = exec_scope["layer"]
                elif 'features' in layer_type:
                    exec_scope = {
                        "model": model,  # 注入模型对象
                        "lidx": lidx,  # 分类器层索引
                    }
                    # 2. 构建动态代码字符串
                    code_str = f"layer=model.{layer_type}[lidx]"
                    exec(code_str, globals(), exec_scope)
                    layer = exec_scope["layer"]
                else:
                    continue

                key = (layer_type, lidx)
                hook_fn = lambda module, input, output, k=key: self.activations.update({k: output.detach().cpu()})
                handle = layer.register_forward_hook(hook_fn)
                self.handles.append(handle)
        elif arch=='resnet18_class43':
            for layer_type, lidx in layers_to_hook:
                if 'conv' in layer_type:
                    exec_scope = {
                        "model": model,  # 注入模型对象
                        "lidx": lidx,  # 分类器层索引
                    }
                    # 2. 构建动态代码字符串
                    code_str = f"layer=model.{layer_type}"
                    exec(code_str, globals(), exec_scope)
                    layer= exec_scope["layer"]
                elif 'fc' in layer_type:
                    #layer=None
                    exec_scope = {
                        "model": model,  # 注入模型对象
                        "lidx": lidx,  # 分类器层索引
                    }
                    # 2. 构建动态代码字符串
                    code_str=f"layer=model.{layer_type}"
                    exec(code_str, globals(), exec_scope)
                    layer= exec_scope["layer"]

                else:
                    continue

                key = (layer_type, lidx)
                hook_fn = lambda module, input, output, k=key: self.activations.update({k: output.detach().cpu()})
                handle = layer.register_forward_hook(hook_fn)
                self.handles.append(handle)


    def get_activation(self, key):
        return self.activations.get(key, None)

    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.activations = {}

# ==============================================================================
# Part 1: 修正后的 HybridEvaluator 类 (使用绝对CA阈值, 打印原始准确率)
# ==============================================================================
class HybridEvaluator:
    def __init__(self, base_model, modifier, clean_loader, poison_loader, device, target_list, arg_for_naming,
                 w_act, w_acc, w_reg,
                 ca_target_abs, asr_target_abs, w_ca, w_asr):  # <-- 修改：接收 ca_target_abs

        self.base_model = base_model
        self.modifier = modifier
        self.clean_loader = clean_loader
        self.poison_loader = poison_loader
        self.device = device
        self.target_list = target_list
        self.args_for_naming = arg_for_naming
        self.w_act, self.w_acc, self.w_reg = w_act, w_acc, w_reg
        self.ca_target_abs, self.asr_target_abs = ca_target_abs, asr_target_abs  # <-- 修改：存储 ca_target_abs
        self.w_ca, self.w_asr = w_ca, w_asr

        self.golden_activations = {}
        self.acc_clean_raw = 100.0
        self.acc_poison_raw = 100.0
        self.stepcount = 0

        self.initial_act_dist = 1.0
        self.initial_acc_pen = 1.0

        self.save_dir = "repair_result"
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.last_saved_step_for_threshold = -1

    # ... acc, precompute_golden_activations 方法保持不变 ...
    def acc(self, model, loader):
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                pred = model(x).argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        return (correct / total) * 100 if total > 0 else 0

    def precompute_golden_activations(self):
        print("[INFO] Pre-computing golden activations from clean data...")
        # ... (内部逻辑不变) ...
        hook_manager = ActivationHook()
        hook_manager.register_hook(self.base_model, self.target_list,self.modifier.arch)
        for layer_type, lidx, _, _ in self.target_list:
            self.golden_activations[(layer_type, lidx)] = []
        self.base_model.eval()
        with torch.no_grad():
            for x, y in self.clean_loader:
                x = x.to(self.device)
                self.base_model(x)
                for layer_type, lidx, _, _ in self.target_list:
                    key = (layer_type, lidx)
                    act = hook_manager.get_activation(key)
                    if act is not None:
                        for sample_act in act:
                            self.golden_activations[key].append(sample_act.clone())
        hook_manager.remove_hooks()
        for key in self.golden_activations:
            self.golden_activations[key] = torch.stack(self.golden_activations[key])
        print("[INFO] Golden activations pre-computation complete.")

    def calculate_accuracy_penalty(self, acc_clean, acc_poison):
        # 【核心修改】直接使用绝对值ca_target_abs，不再乘以原始准确率
        ca_target = self.ca_target_abs
        ca_penalty = max(0, ca_target - acc_clean) ** 2
        asr_penalty = max(0, acc_poison - self.asr_target_abs) ** 2
        return (self.w_ca * ca_penalty) + (self.w_asr * asr_penalty)

    # ... _get_penalties_and_accuracies 方法保持不变 ...
    def _get_penalties_and_accuracies(self, solution_vector):
        model_copy = copy.deepcopy(self.base_model)
        modified_model = self.modifier.apply(model_copy, solution_vector)
        modified_model.eval()
        hook_manager = ActivationHook()
        hook_manager.register_hook(modified_model, self.target_list,self.modifier.arch)
        total_distance = 0.0
        current_sample_idx = 0
        with torch.no_grad():
            for x_poison, _ in self.poison_loader:
                batch_size = x_poison.size(0)
                x_poison = x_poison.to(self.device)
                modified_model(x_poison)
                for layer_type, lidx, _, _ in self.target_list:
                    key = (layer_type, lidx)
                    act_poison = hook_manager.get_activation(key)
                    end_idx = current_sample_idx + batch_size
                    if end_idx > len(self.golden_activations.get(key, [])): continue
                    act_golden_slice = self.golden_activations.get(key)[current_sample_idx:end_idx]
                    if act_poison is not None and act_golden_slice.shape[0] > 0 and act_poison.shape[0] == \
                            act_golden_slice.shape[0]:
                        act_poison, act_golden_slice = act_poison.to(self.device), act_golden_slice.to(self.device)
                        total_distance += torch.nn.functional.mse_loss(act_poison, act_golden_slice,
                                                                       reduction='sum').item()
                current_sample_idx += batch_size
        hook_manager.remove_hooks()
        act_dist = total_distance / len(self.poison_loader.dataset) if self.poison_loader.dataset else 0
        acc_clean = self.acc(modified_model, self.clean_loader)
        acc_poison = self.acc(modified_model, self.poison_loader)
        acc_pen = self.calculate_accuracy_penalty(acc_clean, acc_poison)
        reg_pen = np.linalg.norm(solution_vector) ** 2
        return act_dist, acc_pen, reg_pen, acc_clean, acc_poison, modified_model

    def evaluate(self, solution_vector):
        self.stepcount += 1
        start_time = time.time()
        act_dist, acc_pen, reg_pen, acc_clean, acc_poison, modified_model = self._get_penalties_and_accuracies(
            solution_vector)
        norm_act_dist = act_dist / (self.initial_act_dist + 1e-8)
        norm_acc_pen = acc_pen / (self.initial_acc_pen + 1e-8)
        final_score = (self.w_act * norm_act_dist) + (self.w_acc * norm_acc_pen) + (self.w_reg * reg_pen)
        end_time = time.time()

        # 【核心修改】更新打印信息，加入原始准确率
        print(f"Step:{self.stepcount} EvalTime:{(end_time - start_time):.2f}s Score:{final_score:.4f} "
              f"| NormActD:{norm_act_dist:.2f} NormAccP:{norm_acc_pen:.2f} RegP:{reg_pen:.2f} "
              f"| CA:{acc_clean:.2f}% (Raw:{self.acc_clean_raw:.2f}%) ASR:{acc_poison:.2f}% (Raw:{self.acc_poison_raw:.2f}%)")

        # "达标即存"逻辑现在也使用绝对阈值
        if acc_poison < self.asr_target_abs and acc_clean > self.ca_target_abs:
            if self.stepcount > self.last_saved_step_for_threshold + 10:
                self.save_model(modified_model, acc_clean, acc_poison)
                self.last_saved_step_for_threshold = self.stepcount

        return final_score

    # ... evaluateRes, save_model 方法保持不变 ...
    def evaluateRes(self, solution_vector):
        _, _, _, acc_clean, acc_poison, _ = self._get_penalties_and_accuracies(solution_vector)
        return acc_clean, acc_poison

    def save_model(self, model_to_save, ca, asr):
        filename = f"repaired_{self.args_for_naming.set}_{self.args_for_naming.arch}_{self.args_for_naming.bdtype}" \
                   f"_CA{ca:.2f}_ASR{asr:.2f}_Step{self.stepcount}.pth"
        save_path = os.path.join(self.save_dir, filename)
        try:
            torch.save(model_to_save.state_dict(), save_path)
            print(f"\n[SUCCESS] Model met threshold and saved to: {save_path}\n")
        except Exception as e:
            print(f"\n[ERROR] Failed to save model: {e}\n")

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

    def scoreCompute4(self, acc_clean, acc_poison):


        return -(-0.05*max(0,self.acc_clean_raw-acc_clean)**2-2*(self.acc_poison_raw-acc_poison))

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
def ParsingFLfiles(fldir,arch):
    # === Load fault localization results (use third line for layer info) ===
    print("[INFO] Parsing fault localization files...")

    fl_files = glob.glob(os.path.join(fldir, "*.txt"))
    targets = []
    targets_conv = []
    targets_fc = []
    if arch=='stdvgg16_class10':
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

    elif arch=='innervgg13':
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
                    #match_fc = re.match(r'.*\(\s*(\d+)\s*\).*', line)

                    match_fc = re.match(r'.*?\(\s*(\d+)\s*\).*', line)

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
    elif arch=='resnet18_class43':
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
                    #match_fc = re.match(r'.*\(\s*(\d+)\s*\).*', line)

                    match_fc = re.match(r'.*?\(\s*(\d+)\s*\).*', line)

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

    return targets_conv,targets_fc


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

def repairadversarialattack(arg):
    print("[INFO] Starting backdoor repair using RACOS/CMAES")

    # --- 1. 定义超参数 ---
    # 修复目标：定义“好”模型的标准 (使用绝对值)
    CA_TARGET_ABS = 78.0  # CA的目标是达到78%以上
    ASR_TARGET_ABS = 15.0  # ASR的目标是降低到15%以下

    # 修复方法：定义“如何”进行修复
    WEIGHT_ACCURACY = 20.0
    WEIGHT_ACTIVATION = 1.0
    WEIGHT_REG = 0.001

    # 准确率惩罚内部权重
    WEIGHT_CA_IN_ACC_PEN = 40.0
    WEIGHT_ASR_IN_ACC_PEN = 5.0

    # 修复范围与预算
    DELTA_BOUND = 6.0
    BUDGET = 20000


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
    elif arg.set == 'GTSRB':
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
    elif arg.set == 'GTSRB':
        mean, std = [0.3403, 0.3121, 0.3214], [0.2724, 0.2608, 0.2669]

    transform_list.append(transforms.Normalize(mean, std))


    #poison_dir = f"data/poisonedval_{arg.set}_{arg.arch}_{arg.bdtype}"
    poison_dir=arg.addir
    poison_dataset = datasets.ImageFolder(poison_dir, transform=transforms.Compose(transform_list))
    poison_loader = DataLoader(poison_dataset, batch_size=batch_size_use, shuffle=False,num_workers=num_workers,pin_memory=True,worker_init_fn=worker_init_fn)

    # === Load model ===
    print("[INFO] Loading adattack model...")
    #model_path = f"transpace/{arg.set}_{arg.arch}_{arg.bdtype}_bd.pth"

    if arg.pretrainfile!=None:
        model_path=arg.pretrainfile
    else:
        print("WRONG")
        return

    if arg.arch == 'stdvgg16_class10':
        model = models.vgg16(pretrained=True)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, 10)
    elif arg.arch == 'resnet18_class10':
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 10)
    elif arg.arch == 'resnet34_class10':
        model = models.resnet34(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 10)
    elif arg.arch == "innervgg16":
        # imagenet10
        model = VGG16_dense()
    elif arg.arch == "innervgg13":
        # cifar10
        model = VGG13_dense(vgg_name='VGG13')
    elif arg.arch=="resnet18_class43":
        #gtsrb
        model = models.resnet18(pretrained=True)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.fc = nn.Linear(model.fc.in_features, 43)

    else:
        raise NotImplementedError(f"Unsupported architecture: {arg.arch}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()


    targets_conv,targets_fc=ParsingFLfiles(arg.fldir,arg.arch)


    selected_targets_conv = select_important_parameters(targets_conv, top_k=10)  # 选择重要性最高的10个参数
    selected_targets_fc = select_important_parameters(targets_fc, top_k=5)  # 选择重要性最高的10个参数

    targets=[]
    targets=selected_targets_fc+selected_targets_conv
    print(f"[INFO] Total {len(targets)} neurons or edges to repair")
    for t in targets:
        print(f" - Layer: {t[0]}[{t[1]}], Index: {t[2]}, Importance: {t[3]:.4f}")


    # === Define modifier and evaluator ===
    modifier = Modifier(model, targets, device,arg.arch)

    print("[DEBUG] Repair targets and inferred parameter dimensions:")
    for k, v in modifier.in_features_map.items():
        print(f"  - Target: {k}, Dimensions: {v}")
    print(f"[DEBUG] Total repair dimension: {modifier.totaldim}")

    if modifier.totaldim == 0:
        print("[ERROR] No repairable parameters found. Please check the fault localization results.")
        return


    if arg.method=='RACOS':
        evaluator = Evaluator(model, modifier, clean_loader, poison_loader, device)
        evaluator.acc_clean_raw, evaluator.acc_poison_raw = evaluator.evaluateRes(np.zeros(modifier.totaldim))
        print(f"acc_clean_raw:{evaluator.acc_clean_raw},acc_poison_raw:{evaluator.acc_poison_raw}")
        dim = Dimension(modifier.totaldim, [[-delta_bound, delta_bound]] * modifier.totaldim, [True] * modifier.totaldim)
        obj = Objective(lambda sol: evaluator.evaluate(sol.get_x()), dim)
        num_samples=10
        init_samples = generate_initial_samples(modifier.totaldim, num_samples, distribution='uniform')
        param = Parameter(budget=30, init_samples=[np.zeros(modifier.totaldim)])

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
    elif arg.method=="CMAES":
        # 1. 创建评估器实例，【核心修改】传入新的 ca_target_abs
        evaluator = HybridEvaluator(
            model, modifier, clean_loader, poison_loader, device, targets, arg,
            w_act=WEIGHT_ACTIVATION, w_acc=WEIGHT_ACCURACY, w_reg=WEIGHT_REG,
            ca_target_abs=CA_TARGET_ABS,  # <-- 修改
            asr_target_abs=ASR_TARGET_ABS,
            w_ca=WEIGHT_CA_IN_ACC_PEN, w_asr=WEIGHT_ASR_IN_ACC_PEN
        )

        # 2. 在这个实例上执行所有预计算 (逻辑不变)
        evaluator.precompute_golden_activations()
        evaluator.acc_clean_raw, evaluator.acc_poison_raw = evaluator.evaluateRes(np.zeros(modifier.totaldim))
        initial_act_dist, initial_acc_pen, _, _, _, _ = evaluator._get_penalties_and_accuracies(
            np.zeros(modifier.totaldim))
        evaluator.initial_act_dist = initial_act_dist
        evaluator.initial_acc_pen = initial_acc_pen

        print(
            f"[INFO] Initial penalties for normalization: act_dist={initial_act_dist:.2f}, acc_pen={initial_acc_pen:.2f}")

        # --- 设置并启动 CMA-ES 优化器 (逻辑不变) ---
        initial_solution = np.zeros(modifier.totaldim)
        sigma0 = DELTA_BOUND / 3.0
        options = {'bounds': [-DELTA_BOUND, DELTA_BOUND], 'maxfevals': BUDGET, 'verbose': -9}
        print("[INFO] Launching CMA-ES optimization with normalized objective...")
        best_solution_vector, es = cma.fmin2(evaluator.evaluate, initial_solution, sigma0, options)

        # --- 结束与保存最终最佳模型 (逻辑不变) ---
        print("\n[INFO] Optimization finished. Applying the best found solution...")
        best_model = modifier.apply(copy.deepcopy(model), best_solution_vector)
        final_ca, final_asr = evaluator.evaluateRes(best_solution_vector)
        print(f"Final Repaired Performance -> CleanAcc: {final_ca:.2f}%, PoisonAcc: {final_asr:.2f}%")
        evaluator.save_model(best_model, final_ca, final_asr)
        print(f"[✔] Repair process completed with CMA-ES.")


