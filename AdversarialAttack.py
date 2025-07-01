import os
from torch.utils.data import DataLoader
import torchattacks
import torchvision.utils as vutils
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms  # <-- 添加这行
from torchvision.transforms import Compose, ToTensor, Normalize
from model.inner_vgg import VGG16_dense
from model.inner_vgg import VGG13_dense
from torchvision import models
from model.cnn import CNN6_CIFAR10,CNN6_MNIST
from tqdm import tqdm



AD_CONFIG = {
    #'dataset_name': 'CIFAR10',
    'dataset_name': 'MNIST',  # 可切换为 'CIFAR10'
    #'dataset_name': 'IMAGENET10',
    'architecture':'stdvgg16_class10',
    'ad_type':'PGD',
    'img_size':28,
    'batch_size': 64,
    #改变batch_size可以改变攻击时间
    'epochs': 20,
    'lr': 0.01,
    
}

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
    elif dataset_name=='GTSRB':
        train_dataset = dataset_class(
            root='./data/gtsrb/train',  # 训练集路径
            transform=Compose(common_transforms)
        )
        test_dataset = dataset_class(
            root='./data/gtsrb/val',  # 测试集路径
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





def adversarialattack(arg):


    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


    cfg=AD_CONFIG
    cfg['dataset_name']=arg.set
    cfg['ad_type']=arg.adtype
    cfg['architecture']=arg.arch
    train_dataset, test_dataset, in_channels, img_size = prepare_datasets(cfg['dataset_name'])
    cfg['img_size'] = img_size  # 更新配置



    loader = DataLoader(test_dataset, batch_size=cfg['batch_size'], shuffle=False)

    model_raw = None


    # 选择模型
    if arg.arch == 'stdvgg16_class10':

        model_raw = models.vgg16(pretrained=True)
        num_features = model_raw.classifier[6].in_features  # 获取原层输入维度
        model_raw.classifier[6] = nn.Linear(num_features, 10)  # 修改为10类输出

        nn.init.kaiming_normal_(model_raw.classifier[6].weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(model_raw.classifier[6].bias, 0.0)


    elif arg.arch == "resnet18_class10":
        # imagenet10

        model_raw = models.resnet18(pretrained=True)

        # 修改全连接层
        num_features = model_raw.fc.in_features
        model_raw.fc = nn.Linear(num_features, 10)

        # 初始化新层参数
        nn.init.kaiming_normal_(model_raw.fc.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(model_raw.fc.bias, 0.0)

    elif arg.arch == "resnet34_class10":
        # imagenet10

        model_raw = models.resnet34(pretrained=True)
        # 修改全连接层
        num_features = model_raw.fc.in_features
        model_raw.fc = nn.Linear(num_features, 10)

        # 初始化新层参数
        nn.init.kaiming_normal_(model_raw.fc.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(model_raw.fc.bias, 0.0)
    elif arg.arch == "resnet18_class43":

        # gtsrb

        model_raw = models.resnet18(pretrained=True)
        # 修改第一层卷积：原kernel_size=7改为3，stride=2改为1

        model_raw.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model_raw.fc = nn.Linear(512, 43)  # 调整输出层
        # 初始化新层参数
        # 迁移学习 因此提取原本的卷积核参数
        original_model = models.resnet18(pretrained=True)
        # 参数初始化关键步骤
        with torch.no_grad():
            # 截取原7x7卷积的中心3x3区域
            original_weight = original_model.conv1.weight
            center_slice = original_weight[:, :, 2:5, 2:5]  # 取中间3x3区域
            model_raw.conv1.weight.copy_(center_slice)
            # 保持BatchNorm层参数不变（重要！）
            model_raw.bn1.load_state_dict(original_model.bn1.state_dict())
        nn.init.kaiming_normal_(model_raw.fc.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(model_raw.fc.bias, 0.0)

    elif arg.arch == "innervgg16":
        # imagenet10

        model_raw = VGG16_dense()
        pass
    elif arg.arch == "innervgg13":
        # cifar10

        model_raw = VGG13_dense(vgg_name='VGG13')
    elif arg.arch == "CNN6_MNIST":

        model_raw = CNN6_MNIST()
    elif arg.arch == "CNN6_CIFAR10":

        model_raw = CNN6_CIFAR10()
    else:
        pass

    pretrain_para_src=arg.pretrainfile
    #if arg.pretrain==True:
    model_raw.load_state_dict(torch.load(pretrain_para_src))

    A=0
    B=0
    C=0
    D=0
    attack=None
    model_raw=model_raw.to(device)
    if cfg['ad_type'] == 'FGSM':
        eps=0.3
        A=eps
        attack = torchattacks.FGSM(model_raw, eps=eps)
    elif cfg['ad_type'] == 'PGD':
        eps = 0.7
        steps = 70
        alpha = 0.01
        A=eps
        B=steps
        C=alpha
        attack = torchattacks.PGD(model_raw, eps=eps, alpha=alpha, steps=steps)
        #默认配置 eps=8 / 255, alpha=2 / 255, steps=10  99%
        #eps = 0.6 steps = 60 alpha = 0.01   53.55%
        #eps = 0.9 steps = 90 alpha = 0.01   0%
        #eps = 0.7 steps = 70 alpha = 0.01   5.23%
    elif cfg['ad_type'] == 'CW':
        c=500
        kappa=20
        steps=100
        lr=0.01
        A=c
        B=kappa
        C=steps
        D=lr
        attack = torchattacks.CW(model_raw, c=c, kappa=kappa,steps=steps,lr=lr)

    # 生成并保存对抗样本
    adattacked_saveroot=f'./data/AdAttaked_{cfg["ad_type"]}_A{A}B{B}C{C}D{D}_{arg.arch}_{cfg["dataset_name"]}'
    os.makedirs(adattacked_saveroot, exist_ok=True)
    img_count=0
    for images, labels in tqdm(loader):
        images, labels = images.to(device), labels.to(device)
        adv_images = attack(images, labels)

        for i, (img, label) in enumerate(zip(adv_images, labels)):
            img_count+=1
            attacked_class_dir = os.path.join(adattacked_saveroot, str(label.item()))
            os.makedirs(attacked_class_dir, exist_ok=True)
            img_name=f"idx{img_count}_label{label}.png"
            if cfg["dataset_name"]=="MNIST":
                # 解压单通道张量 (1,28,28) -> (28,28)
                img_gray = img.squeeze(0)
                # 转换为PIL图像（自动处理归一化）
                pil_img = transforms.ToPILImage()(img_gray)

                # 显式指定保存为L模式（单通道）
                pil_img.save(os.path.join(attacked_class_dir, img_name))
            else:

                vutils.save_image(img, os.path.join(attacked_class_dir, img_name))


if __name__ == '__main__':
    pass