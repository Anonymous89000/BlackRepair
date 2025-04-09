import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm

from FlexibleCNNFL import device

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


# 数据加载统一函数
def get_dataloaders(batch_size=256):
    #注意不同的数据集有不同的处理方式 后续都要规整起来
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # train_set = torchvision.datasets.MNIST(
    #     root='./data', train=True, download=True, transform=transform
    # )
    # test_set = torchvision.datasets.MNIST(
    #     root='./data', train=False, download=True, transform=transform
    # )
    train_set = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    test_set = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )

    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True),
        DataLoader(test_set, batch_size=batch_size, shuffle=False)
    )


# 主网络训练函数
def train_main_model(epochs=15):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 设备选择
    # 初始化
    model = FlexibleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: 'probe' not in p[0], model.named_parameters()), lr=0.001)
    print(optimizer)
    train_loader, test_loader = get_dataloaders()

    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0


        # 训练阶段
        with tqdm(train_loader, unit="batch", desc="Training", dynamic_ncols=True) as pbar:
            for inputs, targets in pbar:
                inputs=inputs.to(device)
                targets=targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                # 实时显示当前loss（保留4位小数）
                pbar.set_postfix({
                    "batch_loss": f"{loss.item():.4f}",  # 当前batch的loss
                    "avg_loss": f"{total_loss / len(pbar):.4f}"  # 平均loss（总loss/已处理batch数）
                })

        # 验证阶段
        acc = evaluate_main_model(model, test_loader)
        print(f"Epoch {epoch + 1}/{epochs} | Loss: {total_loss / len(train_loader):.4f} | Acc: {acc:.2f}%")

        # 保存最佳模型
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "best_main_model.pth")
            print(f"Saved new best model with acc {acc:.2f}%")

    print(f"Training complete, best accuracy: {best_acc:.2f}%")
    return model


def evaluate_main_model(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 设备选择
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs=inputs.to(device)
            targets=targets.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

    return correct / total * 100.0


# 探针训练函数（参数加载版）
def probe_training_with_pretrained(train_loader, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 设备选择
    model = FlexibleCNN().to(device)

    # 关键步骤：加载预训练参数
    model.load_state_dict(torch.load("best_main_model.pth"))
    print("Loaded pretrained main model weights")

    # 冻结主网络参数
    for name, param in model.named_parameters():
        if 'probe' not in name:
            param.requires_grad = False

    # 训练探针的优化器
    probe_optimizers = [
        optim.Adam(model.probe1.parameters(), lr=0.001),
        optim.Adam(model.probe2.parameters(), lr=0.005),
        optim.Adam(model.probe3.parameters(), lr=0.01),
        optim.Adam(model.probe4.parameters(), lr=0.01),
        optim.Adam(model.probe5.parameters(), lr=0.01),
        optim.Adam(model.probe6.parameters(), lr=0.01),
        optim.Adam(model.probe7.parameters(), lr=0.01)
    ]

    # 训练循环
    for epoch in range(10):
        model.train()
        loss_tracker = [0.0] * len(probe_optimizers)

        for inputs, targets in train_loader:
            inputs=inputs.to(device)
            targets=targets.to(device)
            # 探针前向
            probe_outputs = model(inputs, probe=True)

            # 依次训练各探针
            for i in range(len(probe_optimizers)):
                probe_optimizers[i].zero_grad()

                loss = nn.CrossEntropyLoss()(probe_outputs[i], targets)
                #loss.backward(retain_graph=(i < 2))
                loss.backward()

                probe_optimizers[i].step()
                loss_tracker[i] += loss.item()

        # 验证探针性能
        val_acc = validate_probes(model, test_loader)
        avg_loss = [l / len(train_loader) for l in loss_tracker]

        print(f"Probe Epoch {epoch + 1}/10")
        print(f"  Probe1 | Loss: {avg_loss[0]:.4f} | Acc: {val_acc[0]:.2f}%")
        print(f"  Probe2 | Loss: {avg_loss[1]:.4f} | Acc: {val_acc[1]:.2f}%")
        print(f"  Probe3 | Loss: {avg_loss[2]:.4f} | Acc: {val_acc[2]:.2f}%\n")
        print(f"  Probe4 | Loss: {avg_loss[3]:.4f} | Acc: {val_acc[3]:.2f}%\n")
        print(f"  Probe5 | Loss: {avg_loss[4]:.4f} | Acc: {val_acc[4]:.2f}%\n")
        print(f"  Probe6 | Loss: {avg_loss[5]:.4f} | Acc: {val_acc[5]:.2f}%\n")
        print(f"  Probe7 | Loss: {avg_loss[6]:.4f} | Acc: {val_acc[6]:.2f}%\n")


def validate_probes(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 设备选择
    model=model.to(device)
    model.eval()
    correct = [0] * 7
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs=inputs.to(device)
            targets=-targets.to(device)
            probe_outs = model(inputs, probe=True)
            for i in range(7):
                preds = probe_outs[i].argmax(dim=1)
                correct[i] += (preds == targets).sum().item()
            total += targets.size(0)

    return [c / total * 100 for c in correct]


def main():
    # 并行执行两种训练
    print("=== 主网络训练阶段 ===")
    #_ = train_main_model(epochs=15)  # 最佳模型会自动保存

    print("\n=== 探针训练阶段 ===")
    train_loader, test_loader = get_dataloaders()
    probe_training_with_pretrained(train_loader, test_loader)


if __name__ == "__main__":
    main()
