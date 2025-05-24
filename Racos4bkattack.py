# RacosDefense.py
import os
import torch
import numpy as np
from triton.language import dtype
from zoopt import Dimension, Objective, Parameter, Opt
from torch.utils.data import DataLoader
from BadNetTest import prepare_datasets, FlexibleCNN, CONFIG
from BadNets import BadNets, CreatePoisonedDataset
from base import accuracy

#第0层次
#[(0, 43), (0, 13),(0,58),(0,33),(0,113)]
all_list=[(0,i) for i in range(128)]
# 防御配置（可调整参数）
DEFENSE_CONFIG = {
    'model_path': 'ckpt_epoch_20.pth',
    #'model_path': 'defended_model3284_dft.pth',
    'target_layers': [(1, 43), (1, 13)],  # (层索引, 神经元索引)
    'param_range': (-0.5, 0.5),
    'budget': 10,
    'lambda_acc': 1.0,  # 基础准确率权重
    'lambda_asr': 2.0  # 攻击成功率权重
}


class ModelModifier:
    """实现根据参数向量修改模型权重的核心逻辑"""

    def __init__(self, model, target_layers,device=torch.device('cuda')):
        self.original_state = model.state_dict()
        self.layer_info = []
        self.device=device

        # 收集目标层的维度信息
        for l_idx, neuron_idx in target_layers:
            layer=getattr(model,f'dense{l_idx}')
            #layer = model.classifier[l_idx]
            if not isinstance(layer, torch.nn.Linear):
                print(layer)
                raise ValueError(f"层 {l_idx} 不是全连接层")

            in_features = layer.weight.shape[1]
            self.layer_info.append({
                'layer_index': l_idx,
                'neuron_index': neuron_idx,
                'in_features': in_features
            })

        # 计算总参数维度

        self.total_dims = sum([info['in_features'] for info in self.layer_info])

    def apply_params(self, model, params):
        """应用新的参数到模型"""
        state_dict = model.state_dict()
        ptr = 0

        for info in self.layer_info:
            l_idx = info['layer_index']
            neuron_idx = info['neuron_index']
            in_features = info['in_features']

            # 截取当前层的参数段
            layer_params = params[ptr:ptr + in_features]
            ptr += in_features

            # 修改权重
            weight_key = f'dense{l_idx}.weight'
            #state_dict[weight_key][neuron_idx] += torch.FloatTensor(layer_params)
            state_dict[weight_key][neuron_idx] += torch.tensor(layer_params,dtype=torch.float32,device=self.device)
        model.load_state_dict(state_dict)
        return model


class DefenseEvaluator:
    """实现目标函数评估逻辑"""

    def __init__(self, model, modifier, clean_loader, poison_loader, device):
        self.base_model = model
        self.modifier = modifier
        self.clean_loader = clean_loader
        self.poison_loader = poison_loader
        self.device = device

    def evaluate_model(self,model, dataloader):
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(self.device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        return (np.array(all_preds) == np.array(all_labels)).mean() * 100

    def evaluate(self, solution):
        # 创建模型副本
        model_copy = FlexibleCNN()
        model_copy.load_state_dict(self.base_model.state_dict())
        model_copy = model_copy.to(self.device)

        # 应用参数修改
        params = np.array(solution)
        modified_model = self.modifier.apply_params(model_copy, params)



        # 评估性能
        clean_acc = self.evaluate_model(modified_model, self.clean_loader)
        poison_acc = self.evaluate_model(modified_model, self.poison_loader)

        # 目标函数：最小化此值 (RACOS默认最小化)
        # 公式：- (λ1*clean_acc - λ2*poison_acc)
        return -(DEFENSE_CONFIG['lambda_acc'] * clean_acc + DEFENSE_CONFIG['lambda_asr'] * poison_acc)


def prepare_evaluation_data(dataset_name, test_dataset):
    """准备毒化测试集（复用现有逻辑）"""
    trigger = torch.ones(1, 28, 28) if dataset_name == 'MNIST' else torch.ones(3, 32, 32)
    PoisonedDataset = CreatePoisonedDataset(
        benign_dataset=test_dataset,
        y_target=CONFIG['target_class'],
        poisoned_rate=1.0,
        pattern=None,
        weight=None,
        poisoned_transform_index=0,
        poisoned_target_transform_index=0
    )
    return PoisonedDataset


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #

    dataset_name = CONFIG['dataset_name']
    train_ds, test_ds, in_channels, img_size = prepare_datasets(dataset_name)

    CONFIG['img_size'] = img_size  # 更新配置
    # 加载被感染的模型
    model = FlexibleCNN().to(device)
    #model.load_state_dict(torch.load(DEFENSE_CONFIG['model_path'], map_location=device))
    model.eval()




    # 初始化参数修改器
    modifier = ModelModifier(model, DEFENSE_CONFIG['target_layers'],device=device)

    # 准备数据集
    poisoned_test = prepare_evaluation_data(dataset_name, test_ds)

    clean_loader = DataLoader(test_ds, batch_size=128, shuffle=False)
    poison_loader = DataLoader(poisoned_test, batch_size=128, shuffle=False)

    # 初始化评估器
    evaluator = DefenseEvaluator(
        model=model,
        modifier=modifier,
        clean_loader=clean_loader,
        poison_loader=poison_loader,
        device=device
    )

    raw_clean_acc = evaluator.evaluate_model(model, clean_loader)
    raw_poison_acc = evaluator.evaluate_model(model, poison_loader)

    print(f"防御前模型性能:")
    print(f"干净准确率: {raw_clean_acc:.2f}%")
    print(f"攻击后准确率: {raw_poison_acc:.2f}%")


    print(modifier.total_dims)
    # RACOS优化配置
    dim = Dimension(modifier.total_dims, [DEFENSE_CONFIG['param_range']] * modifier.total_dims,
                    [True] * modifier.total_dims)
    objective = Objective(lambda sol: evaluator.evaluate(sol.get_x()), dim)

    parameter = Parameter(
        budget=DEFENSE_CONFIG['budget'],
        init_samples=[np.zeros(modifier.total_dims)]  # 从原始参数开始优化
    )

    # 执行优化
    opt = Opt()
    best_solution = opt.min(objective, parameter)

    # 应用最佳参数
    best_model = FlexibleCNN().to(device=device)
    best_model.load_state_dict(model.state_dict())
    modifier.apply_params(best_model, best_solution.get_x())

    # 保存与评估
    torch.save(best_model.state_dict(), 'defended_model.pth')

    best_model = best_model.to(device)  # 确保模型在正确设备上
    final_clean_acc = evaluator.evaluate_model(best_model, clean_loader)
    final_poison_acc = evaluator.evaluate_model(best_model, poison_loader)

    print(f"防御后模型性能:")
    print(f"干净准确率: {final_clean_acc:.2f}%")
    print(f"攻击后准确率: {final_poison_acc:.2f}%")
    print(f"目标函数值: {best_solution.get_value():.2f}")
