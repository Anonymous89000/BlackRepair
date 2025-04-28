import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import Compose, ToTensor, Normalize
from BadNets import *
import torch.nn.functional as F

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

# 配置参数（用户可修改区域）
CONFIG = {
    'dataset_name': 'CIFAR10',
    #'dataset_name': 'MNIST',  # 可切换为 'CIFAR10'
    'target_class': 0,  # 攻击目标类别
    'poison_rate': 0.1,  # 训练集投毒比例
    'batch_size': 128,
    'epochs': 50,
    'lr': 0.01,
    'device': 'GPU' if torch.cuda.is_available() else 'cpu'
}

# 设置随机种子保证可重复性
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# 1. 灵活数据集准备
def prepare_datasets(dataset_name):
    # 公共参数
    common_transforms = [ToTensor()]

    # 数据集特定配置
    if dataset_name == 'MNIST':
        # MNIST参数
        mean, std = (0.1307,), (0.3081,)
        dataset_class = torchvision.datasets.MNIST
        in_channels = 1
        img_size = 28
    elif dataset_name == 'CIFAR10':
        # CIFAR10参数
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        dataset_class = torchvision.datasets.CIFAR10
        in_channels = 3
        img_size = 32
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # 添加标准化
    common_transforms.append(Normalize(mean, std))

    # 训练测试集
    train_dataset = dataset_class(
        root='./data',
        train=True,
        download=True,
        transform=Compose(common_transforms)
    )

    test_dataset = dataset_class(
        root='./data',
        train=False,
        download=True,
        transform=Compose(common_transforms)
    )

    return train_dataset, test_dataset, in_channels, img_size


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
        if vgg_name=='VGG13':
            self.features3 = self._make_layers(cfg[vgg_name][6:9])
            self.features4 = self._make_layers(cfg[vgg_name][9:12])
            self.features5 = self._make_layers(cfg[vgg_name][12:])
        elif vgg_name=='VGG16':

            self.features3 = self._make_layers(cfg[vgg_name][6:10])
            self.features4 = self._make_layers(cfg[vgg_name][10:14])
            self.features5 = self._make_layers(cfg[vgg_name][14:])
        else :
            pass
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




# 主流程
def badnetattack():
    # 加载配置
    cfg = CONFIG

    # 准备数据
    train_dataset, test_dataset, in_channels, img_size = prepare_datasets(cfg['dataset_name'])
    CONFIG['img_size'] = img_size  # 更新配置

    # 初始化模型
    model_bd = FlexibleCNN(vgg_name='VGG13')
    model_raw=FlexibleCNN(vgg_name='VGG13')
    loss_bd= nn.CrossEntropyLoss()
    loss_raw=nn.CrossEntropyLoss()

    # 初始化BadNets
    attacker = BadNets(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        model=model_bd,
        loss=loss_bd,
        y_target=cfg['target_class'],
        poisoned_rate=cfg['poison_rate'],
        schedule={
            'device': cfg['device'],
            'GPU_num': 1,
            'benign_training': False,
            'batch_size': cfg['batch_size'],
            'num_workers': 4,
            'lr': cfg['lr'],
            'momentum': 0.9,
            'weight_decay': 1e-4,
            'gamma': 0.1,
            'schedule': [int(cfg['epochs'] * 0.5), int(cfg['epochs'] * 0.75)],
            'epochs': cfg['epochs'],
            'log_iteration_interval': 100,
            'test_epoch_interval': 5,
            'save_epoch_interval': 10,
            'save_dir': 'checkpoints_badnets',
            'experiment_name': f'BadNets_{cfg["dataset_name"]}'
        }
    )





    rawtrainer=BadNets(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        model=model_raw,
        loss=loss_bd,
        y_target=cfg['target_class'],
        poisoned_rate=0,
        schedule={
            'device': cfg['device'],
            'GPU_num': 1,
            'benign_training': False,
            'batch_size': cfg['batch_size'],
            'num_workers': 4,
            'lr': cfg['lr'],
            'momentum': 0.9,
            'weight_decay': 1e-4,
            'gamma': 0.1,
            'schedule': [int(cfg['epochs'] * 0.5), int(cfg['epochs'] * 0.75)],
            'epochs': cfg['epochs'],
            'log_iteration_interval': 100,
            'test_epoch_interval': 5,
            'save_epoch_interval': 10,
            'save_dir': 'checkpoints_badnets',
            'experiment_name': f'BadNets_{cfg["dataset_name"]}'
        }
    )


    print(f"Training rawnet on {cfg['dataset_name']}...")
    rawtrainer.train()

    # 训练
    print(f"Training bdnet on {cfg['dataset_name']}...")
    attacker.train()

    rawmodel_path="transpace/vgg13_raw.pth"
    bdmodel_path="transpace/vgg13_bd.pth"
    # rawtrainer.model.load_state_dict(torch.load(rawmodel_path))
    # attacker.model.load_state_dict(torch.load(bdmodel_path))

    # 测试
    print("\nTesting rawnet on clean data and poisoned data:")
    rc,rp=rawtrainer.test(test_dataset=attacker.test_dataset,poisoned_test_dataset=attacker.poisoned_test_dataset)

    print("\nTesting bdnet on clean data and poisoned data:")
    bc,bp=attacker.test(test_dataset=attacker.test_dataset,poisoned_test_dataset=attacker.poisoned_test_dataset)

    print("Extended Result: bc~rc  bp>>rp")
    print(f"rc:{rc}  bc:{bc}\nrp:{rp} bp:{bp}")


    torch.save(rawtrainer.model.state_dict(), rawmodel_path)
    torch.save(attacker.model.state_dict(),bdmodel_path)


    # 显示最终结果
    #print(f"\nFinal Results ({cfg['dataset_name']}):")
    #print(f"Clean Accuracy: {attacker.test_results['benign_accuracy']:.2f}%")
    #print(f"Attack Success Rate: {attacker.test_results['poisoned_accuracy']:.2f}%")


if __name__ == "__main__":
    badnetattack()
