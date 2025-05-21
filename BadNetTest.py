import torch
import torch.nn as nn
import torchvision
from torchvision import transforms  # <-- 添加这行
from torchvision.transforms import Compose, ToTensor, Normalize
from BadNets import *
import torch.nn.functional as F
from model.inner_vgg import VGG16_dense
from torchvision import models
from torchvision.utils import save_image


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

# 配置参数（用户可修改区域）
# CIFAR10配置: 128 20
# imagenet10

CONFIG = {
    'dataset_name': 'CIFAR10',
    #'dataset_name': 'MNIST',
    #'dataset_name': 'IMAGENET10',

    'target_class': 0,  # 攻击目标类别
    'poison_rate': 0.1,  # 训练集投毒比例
    'batch_size': 32,
    'pattern':None,
    'weight':None,
    'epochs': 10,
    'lr': 0.01,
    'device': 'GPU' if torch.cuda.is_available() else 'cpu',
    'poisoned_transform_train_index': 2,
    'poisoned_transform_test_index': 2,
    'poisoned_target_transform_index': 0
}

# 设置随机种子保证可重复性
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# 1. 灵活数据集准备
def prepare_datasets(dataset_name):
    # 公共参数
    common_transforms = []
    if dataset_name in ['MNIST', 'CIFAR10']:
        common_transforms.append(ToTensor())
    elif dataset_name == 'IMAGENET10':
        # ImageNet标准预处理流程
        common_transforms.extend([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
    elif dataset_name == 'GTSRB':  # 新增GTRSB处理
        common_transforms.extend([
            transforms.Resize((64, 64)),  # 统一调整尺寸
            transforms.ToTensor(),
        ])
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

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
    elif dataset_name == 'IMAGENET10':
        # ImageNet标准归一化参数
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        dataset_class = torchvision.datasets.ImageFolder
        in_channels = 3
        img_size = 224
    elif dataset_name == 'GTSRB':  # 新增GTRSB处理
        # GTRSB参数（使用标准ImageNet参数作为示例）
        mean = [0.3403, 0.3121, 0.3214]  # GTRSB专用均值
        std = [0.2724, 0.2608, 0.2669]  # GTRSB专用标准差
        dataset_class = torchvision.datasets.ImageFolder
        in_channels = 3
        img_size = 64
        num_classes = 43  # GTRSB有43个类别

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # 添加标准化
    common_transforms.append(Normalize(mean, std))

    # 数据集加载
    if dataset_name == 'IMAGENET10':
        train_dataset = dataset_class(
            root='./data/imagenet10/train',  # 根据实际路径修改
            transform=Compose(common_transforms)
        )
        test_dataset = dataset_class(
            root='./data/imagenet10/val',
            transform=Compose(common_transforms)
        )
    elif dataset_name=='GTRSB':
        train_dataset = dataset_class(
            root='./data/gtrsb/Training',  # 训练集路径
            transform=Compose(common_transforms)
        )
        test_dataset = dataset_class(
            root='./data/gtrsb/Testing',  # 测试集路径
            transform=Compose(common_transforms)
        )

    else:
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
        # self.probe1 = Probe(64, 2, num_class=num_class)
        # self.probe2 = Probe(128, 2, num_class=num_class)
        # self.probe3 = Probe(256, 1, num_class=num_class)
        # self.probe4 = Probe(512, 1, num_class=num_class)
        # self.probe5 = nn.Linear(512, num_class)
        # self.probe6 = nn.Linear(1024, num_class)
        # self.probe7 = nn.Linear(1024, num_class)

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
            # 此处要注意
            #f5 = F.adaptive_avg_pool2d(f5, output_size=(1, 1))
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
def badnetattacktest(pretrain=False,train=False,saveRes=False):
    # 加载配置
    cfg = CONFIG

    # 准备数据
    train_dataset, test_dataset, in_channels, img_size = prepare_datasets(cfg['dataset_name'])
    cfg['img_size'] = img_size  # 更新配置

    # 若使用IMAGENET10需要加大trigger尺寸
    if cfg['dataset_name'] == 'IMAGENET10':
        trigger_size = 20
        # 创建全0模板
        pattern = torch.zeros((3, 224, 224), dtype=torch.uint8)
        # 在右下角放白色方块
        pattern[:, -trigger_size:, -trigger_size:] = 255

        weight = torch.zeros((3, 224, 224), dtype=torch.float32)
        weight[:, -trigger_size:, -trigger_size:] = 1.0

        cfg['pattern'] = pattern
        cfg['weight'] = weight
        cfg['poisoned_transform_train_index']=2
        cfg['poisoned_transform_test_index']=2
    else:  # 原有逻辑保持不变
        cfg['pattern'] = None
        cfg['weight'] = None
        cfg['poisoned_transform_train_index']=0
        cfg['poisoned_transform_test_index']=0
    # 初始化模型
    # model_bd = FlexibleCNN(vgg_name='VGG13')
    # model_raw=FlexibleCNN(vgg_name='VGG13')

    # model_bd=VGG16_dense()
    # model_raw=VGG16_dense()
    model_bd=models.vgg16(pretrained=False)
    model_raw=models.vgg16(pretrained=False)
    num_features = model_bd.classifier[6].in_features  # 获取原层输入维度

    model_bd.classifier[6] = nn.Linear(num_features, 10)  # 修改为10类输出
    model_raw.classifier[6] = nn.Linear(num_features, 10)  # 修改为10类输出

    nn.init.kaiming_normal_(model_bd.classifier[6].weight, mode='fan_in', nonlinearity='relu')
    nn.init.constant_(model_bd.classifier[6].bias, 0.0)

    nn.init.kaiming_normal_(model_raw.classifier[6].weight, mode='fan_in', nonlinearity='relu')
    nn.init.constant_(model_raw.classifier[6].bias, 0.0)

    #参数冻结 非常有效!
    #为什么不冻结参数会产生NAN呢?为什么softmax输入需要大于1e-8
    # for param in model_bd.parameters():
    #     param.requires_grad = False
    # for param in model_bd.classifier[6].parameters():
    #     param.requires_grad = True
    for param in model_raw.parameters():
        param.requires_grad = False
    for param in model_raw.classifier[6].parameters():
        param.requires_grad = True


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
        pattern=cfg['pattern'],
        weight=cfg['weight'],
        poisoned_transform_train_index=cfg['poisoned_transform_train_index'],
        poisoned_transform_test_index=cfg['poisoned_transform_test_index'],
        poisoned_target_transform_index=cfg['poisoned_target_transform_index'],
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
        loss=loss_raw,
        y_target=cfg['target_class'],
        poisoned_rate=0,
        pattern=cfg['pattern'],
        weight=cfg['weight'],
        poisoned_transform_train_index=cfg['poisoned_transform_train_index'],
        poisoned_transform_test_index=cfg['poisoned_transform_test_index'],
        poisoned_target_transform_index=cfg['poisoned_target_transform_index'],
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

    rawmodel_src_path= "transpace/vgg16std9913_raw.pth"
    bdmodel_src_path= "transpace/vgg16std9556_bd.pth"
    if pretrain==True:
        rawtrainer.model.load_state_dict(torch.load(rawmodel_src_path))
        attacker.model.load_state_dict(torch.load(bdmodel_src_path))


    if train==True:
        print(f"Training rawnet on {cfg['dataset_name']}...")
        rawtrainer.train()

        # 训练
        print(f"Training bdnet on {cfg['dataset_name']}...")
        attacker.train()

    poisioned_data_saveroot=f"./data/poisioned_{cfg['dataset_name']}"
    clean_data_saveroot=f"./data/clean_{cfg['dataset_name']}"
    # 创建总保存目录
    os.makedirs(poisioned_data_saveroot, exist_ok=True)
    os.makedirs(clean_data_saveroot, exist_ok=True)

    # 根据数据集配置反标准化参数
    dataset_name = cfg['dataset_name']
    if dataset_name == 'IMAGENET10':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif dataset_name == 'CIFAR10':
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
    elif dataset_name == 'MNIST':
        mean = [0.1307]
        std = [0.3081]
    else:
        raise ValueError("Unsupported dataset for saving images")
    # 遍历所有样本
    for idx in range(len(attacker.test_dataset)):
        # 获取原始样本的true label
        #true_label = attacker.test_dataset[idx][1]  # 原始测试集的真实标签

        #获取干净图像
        clean_img, true_label = attacker.test_dataset[idx]

        # 获取毒化样本图像
        poisoned_img, _ = attacker.poisoned_test_dataset[idx]  # 忽略毒化数据集的标签

        # 创建按true label分类的目录
        poisoned_class_dir = os.path.join(poisioned_data_saveroot, f'true_class_{true_label}')
        os.makedirs(poisoned_class_dir, exist_ok=True)

        clean_class_dir = os.path.join(clean_data_saveroot, f'true_class_{true_label}')
        os.makedirs(clean_class_dir, exist_ok=True)
        # 反标准化处理

        img = poisoned_img.clone().detach()
        for t in range(img.shape[0]):
            img[t] = img[t] * std[t] + mean[t]

        img1 = clean_img.clone().detach()
        for t in range(img.shape[0]):
            img1[t] = img1[t] * std[t] + mean[t]

        # 生成文件名
        filename = f'poisoned_{idx}_true_{true_label}_target_{cfg["target_class"]}.png'
        save_path = os.path.join(poisoned_class_dir, filename)
        img= torch.clamp(img, 0.0, 1.0)  # <-- 关键补充步骤
        # 保存图像（禁用自动归一化）
        save_image(img, save_path, normalize=False)

        # 生成文件名
        filename = f'clean_{idx}_true_{true_label}_target_{cfg["target_class"]}.png'
        save_path = os.path.join(clean_class_dir, filename)
        img1= torch.clamp(img1, 0.0, 1.0)  # <-- 关键补充步骤
        # 保存图像（禁用自动归一化）
        save_image(img1, save_path, normalize=False)
    exit(0)

    # 测试
    print("\nTesting rawnet on clean data and poisoned data:")
    rc,rp=rawtrainer.test(test_dataset=attacker.test_dataset,poisoned_test_dataset=attacker.poisoned_test_dataset)

    print("\nTesting bdnet on clean data and poisoned data:")
    bc,bp=attacker.test(test_dataset=attacker.test_dataset,poisoned_test_dataset=attacker.poisoned_test_dataset)

    print("Extended Result: bc~rc  bp>>rp")
    print(f"rc:{rc}  bc:{bc}\nrp:{rp} bp:{bp}")


    rawmodel_tar_path= "transpace/vgg16std_raw.pth"
    bdmodel_tar_path= "transpace/vgg16std_bd.pth"
    if saveRes==True:
        torch.save(rawtrainer.model.state_dict(), rawmodel_tar_path)
        torch.save(attacker.model.state_dict(),bdmodel_tar_path)







if __name__ == "__main__":
    badnetattacktest(pretrain=False,train=False,saveRes=False)
