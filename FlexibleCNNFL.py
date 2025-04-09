# fault_localization.py
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from collections import defaultdict
import matplotlib.pyplot as plt
import torch.nn.functional as F
# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class Probe(nn.Module):
    def __init__(self, in_ch, layer_num=2, num_class=10):
        super(Probe, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(),
        )
        if layer_num == 2:
            self.convs = nn.Sequential(
                nn.Conv2d(in_ch, in_ch * 2, 3, 2, 1),
                nn.Conv2d(in_ch * 2, in_ch * 2, 3, 2, 1),
                nn.BatchNorm2d(in_ch * 2),
                nn.ReLU(),
                nn.Conv2d(in_ch * 2, in_ch * 4, 3, 1, 1),
                nn.Conv2d(in_ch * 4, in_ch * 4, 3, 1, 1),
                nn.BatchNorm2d(in_ch * 4),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1)
            )
            self.fc = nn.Linear(in_ch * 4, num_class)
        elif layer_num == 1:
            self.convs = nn.Sequential(
                nn.Conv2d(in_ch, in_ch, 3, 1, 1),
                nn.BatchNorm2d(in_ch),
                nn.ReLU(),
                nn.Conv2d(in_ch, in_ch, 3, 1, 1),
                nn.BatchNorm2d(in_ch),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1)
            )
            self.fc = nn.Linear(in_ch, num_class)

    def forward(self, x):
        feat = self.features(x)
        feat = self.convs(feat)
        feat = feat.view(feat.size(0), -1)
        out = self.fc(feat)
        return out
# 1. 带探针的模型定义
class FlexibleCNN(nn.Module):
    def __init__(self, vgg_name='VGG13', num_class=10):
        super(FlexibleCNN, self).__init__()
        self.in_channels = 3
        # self.features = self._make_layers(cfg[vgg_name])
        self.features1 = self._make_layers(cfg[vgg_name][0:3])
        self.features2 = self._make_layers(cfg[vgg_name][3:6])
        self.features3 = self._make_layers(cfg[vgg_name][6:9])
        self.features4 = self._make_layers(cfg[vgg_name][9:12])
        self.features5 = self._make_layers(cfg[vgg_name][12:])
        self.dense1 = nn.Linear(512, 1024)
        self.dense2 = nn.Linear(1024, 1024)
        self.classifier = nn.Linear(1024, num_class)
        self.probe1 = Probe(64, 2, num_class=num_class)
        self.probe2 = Probe(128, 2, num_class=num_class)
        self.probe3 = Probe(256, 1, num_class=num_class)
        self.probe4 = Probe(512, 1, num_class=num_class)
        self.probe5 = nn.Linear(512, num_class)
        self.probe6 = nn.Linear(1024, num_class)
        self.probe7 = nn.Linear(1024, num_class)

        # 使用结构化的梯度存储
        self.grad_info = {
            # name: (is_conv_layer, tensor_shape)
            'features1': (True, None),
            'features2': (True, None),
            'dense1': (False, None),
            'dense2': (False, None),
            'classfier': (False, None)
        }
        # 梯度收集机制
        self.activations = {}
        self.gradients = {}
        self._register_hooks()
        self._register_backward_hooks()

    def _register_hooks(self):
        """注册梯度钩子"""

        def forward_hook(layer_name):
            def hook(module, input, output):
                self.activations[layer_name] = output.detach()

            return hook

        def backward_hook(layer_name):
            def hook(module, grad_input, grad_output):
                self.gradients[layer_name] = grad_output[0].detach()

            return hook

        # 对需要监测的层注册钩子
        self.features1.register_forward_hook(forward_hook('conv1'))
        self.features2.register_forward_hook(forward_hook('conv2'))
        self.dense1.register_forward_hook(forward_hook('fc1'))
        self.dense2.register_forward_hook(forward_hook('fc2'))
        self.classifier.register_forward_hook(forward_hook('classifier'))

    def _register_backward_hooks(self):
        """智能梯度捕获机制"""
        def make_hook(layer_name, is_conv):
            def hook(module, grad_input, grad_output):
                # grad_output形状检测
                tensor = grad_output[0].detach()
                self.grad_info[layer_name] = (is_conv, tensor.shape)
                self.gradients[layer_name] = tensor
            return hook

        # 为卷积层注册钩子
        self.features1.register_full_backward_hook(
            make_hook('conv1_grad', is_conv=True))
        self.features2.register_full_backward_hook(
            make_hook('conv2_grad', is_conv=True))
        # 为全连接层注册钩子
        self.dense1.register_full_backward_hook(
            make_hook('fc1_grad', is_conv=False))
        self.dense2.register_full_backward_hook(
            make_hook('fc2_grad', is_conv=False))
        self.classifier.register_full_backward_hook(
            make_hook('classifier_grad', is_conv=False))

    def forward(self, x, probe=False):

        if probe:
            f1 = self.features1(x)
            p1 = self.probe1(f1)
            f2 = self.features2(f1)
            p2 = self.probe2(f2)
            f3 = self.features3(f2)
            p3 = self.probe3(f3)
            f4 = self.features4(f3)
            p4 = self.probe4(f4)
            f5 = self.features5(f4)
            f5 = f5.view(f5.size(0), -1)
            p5 = self.probe5(f5)
            d1 = F.relu(self.dense1(f5))
            p6 = self.probe6(d1)
            d2 = F.relu(self.dense2(d1))
            p7 = self.probe7(d2)
            out = d2.view(f5.size(0), -1)
            out = self.classifier(out)
            return p1, p2, p3, p4, p5, p6, p7, out
        else:
            f1 = self.features1(x)
            f2 = self.features2(f1)
            f3 = self.features3(f2)
            f4 = self.features4(f3)
            f5 = self.features5(f4)
            f5 = f5.view(f5.size(0), -1)
            d1 = F.relu(self.dense1(f5))
            d2 = F.relu(self.dense2(d1))
            out = d2.view(f5.size(0), -1)
            out = self.classifier(out)
            return out

    def _make_layers(self, cfg):
        layers = []

        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(self.in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                self.in_channels = x
        # layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    # 2. 数据加载
def get_dataloaders(batch_size=256):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    #train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    #test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True),
        DataLoader(test_set, batch_size=batch_size, shuffle=False)
    )

# 3. 主网络训练
def train_main_model(epochs=5):
    model = FlexibleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_loader, test_loader = get_dataloaders()

    for epoch in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        # 验证
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        print(f'Epoch {epoch + 1}, Acc: {100 * correct / total:.2f}%')

    torch.save(model.state_dict(), 'main_model.pth')
    return model

# 4. 探针训练
def train_probes(probe_epochs=3):
    model = FlexibleCNN().to(device)
    model.load_state_dict(torch.load('ckpt_epoch_20.pth'))

    # 冻结主网络参数
    for param in model.parameters():
        param.requires_grad = False
    for probe in model.probes:
        for param in probe.parameters():
            param.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizers = [
        optim.Adam(model.probes[0].parameters(), lr=0.001),
        optim.Adam(model.probes[1].parameters(), lr=0.001),
        optim.Adam(model.probes[2].parameters(), lr=0.001)
    ]

    train_loader, _ = get_dataloaders()
    for epoch in range(probe_epochs):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            _, probe_outs = model(inputs, probe=True)

            for i, optimizer in enumerate(optimizers):
                optimizer.zero_grad()
                loss = criterion(probe_outs[i], targets)
                loss.backward(retain_graph=(i < 2))
                optimizer.step()
        print(f'Probe Epoch {epoch + 1} completed')

# 5. 故障定位核心函数
# 这个函数看着很费脑筋
def locate_fault_neurons(model, dataloader, topk=5):
    model.eval()
    model.zero_grad()

    fault_report = defaultdict(list)

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        # 前向+反向传播

        main_out=model(inputs, probe=False)
        probe_outs = model(inputs, probe=True)
        total_loss = 0.3 * F.cross_entropy(main_out, targets) + 0.7 * sum(
            F.cross_entropy(p, targets) for p in probe_outs)
        total_loss.backward()

        # 提取梯度 - 修正维度处理
        for layer in model.gradients:
            is_conv, _ = model.grad_info.get(layer, (False, None))
            if is_conv:
                # 层梯度形状 (N,C,H,W) → 取通道均值
                grads = model.gradients[layer].abs().mean(dim=(0, 2, 3))  # [C]
            else:
                grads = model.gradients[layer].abs().mean(dim=0)  # [C]
            fault_report[layer].append(grads.cpu())

        model.zero_grad()

    final_report = {}
    for layer in fault_report:
        # 获取该层的类型信息
        is_conv, _ = model.grad_info.get(layer, (False, None))

        # 参数重要性计算
        all_grads = torch.stack(fault_report[layer])
        if is_conv:
            # 卷积层：avg over (batch, height, width)
            importance = all_grads.mean(dim=0)  # [C]
        else:
            # 全连接层：直接使用特征维度
            importance = all_grads.mean(dim=0)  # [D]

        # 选取topk神经元
        top_vals, top_indices = torch.topk(importance, topk)
        final_report[layer] = {
            'neurons': top_indices.tolist(),
            'scores': top_vals.tolist()
        }


    return final_report

# 6. 可视化
def visualize_report(report):
    plt.figure(figsize=(10, 6))
    for i, (layer, info) in enumerate(report.items()):
        plt.bar([f"{layer}-{n}" for n in info['neurons']], info['scores'], label=layer)
    plt.xticks(rotation=45)
    plt.ylabel('Importance Score')
    plt.title('Fault Neuron Distribution')
    plt.legend()
    plt.tight_layout()
    plt.savefig('fault_distribution.png')
    plt.show()

# 7. 主流程
if __name__ == "__main__":
    # # 训练主网络
    # print("Training main network...")
    # train_main_model()
    #
    # # 训练探针
    # print("\nTraining probes...")
    # train_probes()

    # 加载模型
    model = FlexibleCNN().to(device)
    model.load_state_dict(torch.load('ckpt_epoch_20.pth'))

    for name, module in model.named_modules():
        print(f"Module name: {name}, Type: {module.__class__.__name__}")



    # 获取数据
    _, test_loader = get_dataloaders(batch_size=64)

    # 执行故障定位
    print("\nLocating fault neurons...")
    fault_report = locate_fault_neurons(model, test_loader)

    # 显示结果
    print("\nFault Neurons Report:")
    for layer, info in fault_report.items():
        print(f"Layer {layer}:")
        print(f"  Neurons: {info['neurons']}")
        print(f"  Scores: {info['scores']}\n")

    # 可视化
    visualize_report(fault_report)
