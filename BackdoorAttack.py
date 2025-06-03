import time
from time import sleep

import torch
import torch.nn as nn
import torchvision
from PIL.ImageChops import offset
from torchvision import transforms  # <-- 添加这行
from torchvision.transforms import Compose, ToTensor, Normalize
from attacks.BadNets import BadNets
from attacks.WaNet import WaNet
from attacks.Blended import Blended
import torch.nn.functional as F
from model.inner_vgg import VGG16_dense
from model.inner_vgg import VGG13_dense
from torchvision import models
from model.cnn import CNN6_CIFAR10,CNN6_MNIST
import os
#from utils.utils import pretrained
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 新增后门攻击参数池
BACKDOOR_PARAMS = {
    'BadNets': {
        'pattern': None,
        'weight': None,
        'trigger_size': 5  # 默认触发器大小
    },
    'WaNet': {
        'identity_grid': None,
        'noise_grid': None,
        'noise': False,
        's': 0.5,
        'grid_rescale': 1,
        'noise_rescale': 2
    },
    'Blended': {
        'alpha': 0.2  # 混合透明度
    }
}

# 配置参数（用户可修改区域）
# CIFAR10配置: 128 20
# imagenet10



CONFIG = {
    #'dataset_name': 'CIFAR10',
    'dataset_name': 'MNIST',  # 可切换为 'CIFAR10'
    #'dataset_name': 'IMAGENET10',
    'architecture':'stdvgg16_class10',
    'bd_type':'BadNets',
    'target_class': 0,  # 攻击目标类别
    'poison_rate': 0.20,  # 训练集投毒比例
    'batch_size': 32,
    'pattern':None,
    'weight':None,
    #'epochs': 50,
    'epochs': 50,
    'lr': 0.01,
    'device': 'GPU' if torch.cuda.is_available() else 'cpu',
    'poisoned_transform_train_index': 0,
    'poisoned_transform_test_index': 0,
    'poisoned_target_transform_index': 0,
    'identity_grid':None,
    'noise_grid': None,
    'noise': True,
    's': 0.5,
    'grid_rescale': 0,
    'img_size':None
    #**BACKDOOR_PARAMS['BadNets']
}

# 设置随机种子保证可重复性
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



def prepare_datasets(dataset_name):
    # 分离训练集和验证集的预处理流程
    common_transforms_train = []  # 训练集专用增强流程
    common_transforms_val = []    # 验证集专用基础流程

    # 公共参数
    if dataset_name == 'MNIST':
        common_transforms_train.append(transforms.ToTensor())
        common_transforms_val.append(transforms.ToTensor())
    elif dataset_name == 'CIFAR10':
        # 训练集增强配置（参考网页1/2/4/5/9/10/11）
        common_transforms_train.extend([
            transforms.RandomCrop(32, padding=4),  # 随机裁剪[1,4](@ref)
            transforms.RandomHorizontalFlip(p=0.5),  # 水平翻转[2,4](@ref)
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # 颜色抖动[5,10](@ref)
            transforms.ToTensor(),
        ])
        # 验证集保持基础配置
        common_transforms_val.append(transforms.ToTensor())
    elif dataset_name == 'IMAGENET10':
        # ImageNet标准预处理流程（保持原状）
        common_transforms_train.extend([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),  # 训练集用随机裁剪
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        common_transforms_val.extend([
            transforms.Resize(256),
            transforms.CenterCrop(224),  # 验证集用中心裁剪
            transforms.ToTensor(),
        ])
    elif dataset_name == 'GTRSB':
        # GTRSB增强配置（参考网页6/7/9）
        common_transforms_train.extend([
            transforms.Resize((64, 64)),
            transforms.RandomRotation(15),  # 随机旋转±15度[7,9](@ref)
            transforms.RandomAffine(0, shear=10),  # 随机仿射变换[10](@ref)
            transforms.RandomPerspective(distortion_scale=0.2),  # 透视变换[10](@ref)
            transforms.ToTensor(),
        ])
        # 验证集保持基础配置
        common_transforms_val.extend([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # 数据集特定配置（保持原有逻辑）
    if dataset_name == 'MNIST':
        mean, std = (0.1307,), (0.3081,)
        dataset_class = torchvision.datasets.MNIST
        in_channels = 1
        img_size = 28
    elif dataset_name == 'CIFAR10':
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        dataset_class = torchvision.datasets.CIFAR10
        in_channels = 3
        img_size = 32
    elif dataset_name == 'IMAGENET10':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        dataset_class = torchvision.datasets.ImageFolder
        in_channels = 3
        img_size = 224
    elif dataset_name == 'GTRSB':
        mean = [0.3403, 0.3121, 0.3214]
        std = [0.2724, 0.2608, 0.2669]
        dataset_class = torchvision.datasets.ImageFolder
        in_channels = 3
        img_size = 64
        num_classes = 43

    # 添加标准化（训练/验证集共用）
    common_transforms_train.append(transforms.Normalize(mean, std))
    common_transforms_val.append(transforms.Normalize(mean, std))

    # 数据集加载逻辑调整
    if dataset_name in ['IMAGENET10', 'GTRSB']:
        train_dataset = dataset_class(
            root=f'./data/{dataset_name.lower()}/train',
            transform=transforms.Compose(common_transforms_train)  # 训练集用增强流程
        )
        test_dataset = dataset_class(
            root=f'./data/{dataset_name.lower()}/val',
            transform=transforms.Compose(common_transforms_val)  # 验证集用基础流程
        )
    else:
        train_dataset = dataset_class(
            root='./data',
            train=True,
            download=True,
            transform=transforms.Compose(common_transforms_train)  # 训练集用增强流程
        )
        test_dataset = dataset_class(
            root='./data',
            train=False,
            download=True,
            transform=transforms.Compose(common_transforms_val)  # 验证集用基础流程
        )

    return train_dataset, test_dataset, in_channels, img_size

def generate_identity_grid(size):
    """
    生成用于WANET的identity_grid，保证经过grid_sample后图像不发生变化。
    size：图像的大小 (size x size)
    返回：shape 为 (1, size, size, 2) 的标准化网格
    """
    # 创建网格坐标（行，列）
    grid_y, grid_x = torch.meshgrid(torch.linspace(0, size-1, size), torch.linspace(0, size-1, size))

    # 将坐标标准化到 [-1, 1] 范围
    grid_x = 2 * grid_x / (size - 1) - 1  # 列坐标的标准化
    grid_y = 2 * grid_y / (size - 1) - 1  # 行坐标的标准化

    # 将标准化后的坐标堆叠成一个 4D grid，形状为 (size, size, 2)
    identity_grid = torch.stack([grid_x, grid_y], dim=-1)  # shape: (size, size, 2)

    # 扩展成 4D 张量，形状为 (1, size, size, 2)
    identity_grid = identity_grid.unsqueeze(0)  # shape: (1, size, size, 2)
    print(identity_grid.shape)

    return identity_grid


def generate_watermark_trigger(size, type="gradient"):
    """
    生成不同类型的水印触发器，用于Blended攻击。

    参数：
        size: 触发器的大小 (size x size)
        type: 水印类型，可选 "gradient"（渐变矩形）、"noise"（随机噪声）、"text"（文字水印）

    返回：
        pattern: 生成的水印触发器 (size, size)
        weight: 权重矩阵，控制水印对图像的影响 (size, size)
    """

    # 默认使用全零图案
    pattern = torch.zeros((size, size), dtype=torch.float32)  # 改为 (size, size)
    weight = torch.zeros((size, size), dtype=torch.float32)  # 改为 (size, size)

    # 根据 type 选择水印类型
    if type == "gradient":
        # 渐变矩形水印：左上到右下的渐变
        for i in range(size):
            for j in range(size):
                # 生成渐变效果
                pattern[i, j] = (i / (size - 1) + j / (size - 1)) / 2  # 平均值形成渐变
                weight[i, j] = 1.0 - pattern[i, j]  # 反向控制权重，使得透明部分权重低
    elif type == "noise":
        # 随机噪声水印：生成随机噪声图案
        pattern = torch.rand((size, size), dtype=torch.float32)
        weight = torch.ones((size, size), dtype=torch.float32)  # 全部权重为1
    elif type == "text":
        # 文字水印：生成包含文字的水印图案
        # 创建一个白色背景的图像，大小为 size x size
        img = Image.new('L', (size, size), color=255)  # 'L' 模式表示灰度图，背景设为白色
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()  # 使用默认字体
        text_scale=0.3
        # 计算文字的尺寸并根据 text_scale 调整大小
        text = "WA"  # 文字水印内容
        font_size = int(size * text_scale)  # 根据 text_scale 调整字体大小

        if font_size<=10:
            font_size=10
        elif font_size>=72:
            font_size=72
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)  # 使用较大的字体
        # except IOError:
        #     font = ImageFont.load_default()  # 如果找不到字体，则使用默认字体

        # 计算文字的尺寸并居中显示
        text_width, text_height = draw.textsize(text, font=font)
        text_position = ((size - text_width) // 2, (size - text_height) // 2)  # 居中显示文字

        # 使用黑色填充文字
        draw.text(text_position, text, fill=0, font=font)  # 文字颜色为黑色

        # 将生成的文字水印转换为张量
        pattern = torch.tensor(np.array(img), dtype=torch.float32) / 255  # 归一化到 [0, 1]

        # 为了只在文字部分影响图像，生成权重矩阵
        weight = torch.zeros((size, size), dtype=torch.float32)  # 初始化权重为 0
        # 文字部分的区域设置为 1，表示影响图像
        for i in range(size):
            for j in range(size):
                if pattern[i, j] > 0:  # 如果在部分
                    weight[i, j] = 0.0  # 设置权重为 1
                else:
                    weight[i,j]=1.0


    else:
        raise ValueError("Unsupported watermark type. Choose from 'gradient', 'noise', 'text'.")



    return pattern, weight


# 主流程
def backdoorattack(arg):
    # 加载配置
    cfg = CONFIG

    pretrain=arg.pretrain
    train=arg.train
    saveRes=arg.saveRes
    cfg['dataset_name']=arg.set
    cfg['bd_type']=arg.bdtype
    cfg['architecture']=arg.arch

    # 准备数据
    train_dataset, test_dataset, in_channels, img_size = prepare_datasets(cfg['dataset_name'])
    cfg['img_size'] = img_size  # 更新配置

    model_bd=None
    model_raw=None

    #选择模型
    if arg.arch=='stdvgg16_class10':
        # model_bd=models.vgg16(pretrained=True)
        # model_raw=models.vgg16(pretrained=True)
        model_bd=models.vgg16(pretrained=True)
        model_raw=models.vgg16(pretrained=True)
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
    elif arg.arch=="resnet18_class10":
        #imagenet10
        model_bd = models.resnet18(pretrained=True)
        model_raw = models.resnet18(pretrained=True)

        # 修改全连接层
        num_features = model_bd.fc.in_features
        model_bd.fc = nn.Linear(num_features, 10)
        model_raw.fc = nn.Linear(num_features, 10)

        # 初始化新层参数
        nn.init.kaiming_normal_(model_bd.fc.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(model_bd.fc.bias, 0.0)
        nn.init.kaiming_normal_(model_raw.fc.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(model_raw.fc.bias, 0.0)

        # 参数冻结策略
        # for param in model_bd.parameters():
        #     param.requires_grad = False
        # for param in model_bd.fc.parameters():
        #     param.requires_grad = True

        # for param in model_raw.parameters():
        #     param.requires_grad = False
        # for param in model_raw.fc.parameters():
        #     param.requires_grad = True
    elif arg.arch=="resnet34_class10":
        #imagenet10
        model_bd = models.resnet34(pretrained=True)
        model_raw = models.resnet34(pretrained=True)
        # 修改全连接层
        num_features = model_bd.fc.in_features
        model_bd.fc = nn.Linear(num_features, 10)
        model_raw.fc = nn.Linear(num_features, 10)

        # 初始化新层参数
        nn.init.kaiming_normal_(model_bd.fc.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(model_bd.fc.bias, 0.0)
        nn.init.kaiming_normal_(model_raw.fc.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(model_raw.fc.bias, 0.0)

        # 参数冻结策略
        # for param in model_bd.parameters():
        #     param.requires_grad = False
        # for param in model_bd.fc.parameters():
        #     param.requires_grad = True

        for param in model_raw.parameters():
            param.requires_grad = False
        for param in model_raw.fc.parameters():
            param.requires_grad = True
    elif arg.arch=="resnet18_class43":


        #gtsrb
        model_bd = models.resnet18(pretrained=True)
        model_raw = models.resnet18(pretrained=True)
        # 修改第一层卷积：原kernel_size=7改为3，stride=2改为1
        model_bd.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model_bd.fc = nn.Linear(512, 43)  # 调整输出层

        model_raw.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model_raw.fc = nn.Linear(512, 43)  # 调整输出层

        # 初始化新层参数
        #迁移学习 因此提取原本的卷积核参数

        original_model = models.resnet18(pretrained=True)
        # 参数初始化关键步骤
        with torch.no_grad():
            # 截取原7x7卷积的中心3x3区域
            original_weight = original_model.conv1.weight
            center_slice = original_weight[:, :, 2:5, 2:5]  # 取中间3x3区域
            model_bd.conv1.weight.copy_(center_slice)
            model_raw.conv1.weight.copy_(center_slice)

            # 保持BatchNorm层参数不变（重要！）
            model_bd.bn1.load_state_dict(original_model.bn1.state_dict())
            model_raw.bn1.load_state_dict(original_model.bn1.state_dict())

        nn.init.kaiming_normal_(model_bd.fc.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(model_bd.fc.bias, 0.0)
        nn.init.kaiming_normal_(model_raw.fc.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(model_raw.fc.bias, 0.0)



    elif arg.arch=="innervgg16":
        #imagenet10
        model_bd=VGG16_dense()
        model_raw=VGG16_dense()
        pass
    elif arg.arch=="innervgg13":
        #cifar10
        model_bd = VGG13_dense(vgg_name='VGG13')
        model_raw= VGG13_dense(vgg_name='VGG13')
    elif arg.arch=="CNN6_MNIST":
        model_bd=CNN6_MNIST()
        model_raw=CNN6_MNIST()
    elif arg.arch == "CNN6_CIFAR10":
        model_bd = CNN6_CIFAR10()
        model_raw = CNN6_CIFAR10()
    else:
        pass

    loss_bd= nn.CrossEntropyLoss()
    loss_raw=nn.CrossEntropyLoss()

    # 统一攻击参数构建
    attack_params = {
        'train_dataset': train_dataset,
        'test_dataset': test_dataset,
        'model': model_bd,
        'loss': loss_bd,
        'y_target': cfg['target_class'],
        'poisoned_rate': cfg['poison_rate']
    }

    raw_params={
        'train_dataset': train_dataset,
        'test_dataset': test_dataset,
        'model': model_raw,
        'loss': loss_raw,
        'y_target': cfg['target_class'],
        'poisoned_rate': 0
    }

    schedule_base={
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
            'save_epoch_interval': 5,
            'save_dir': f'checkpoints_{cfg["bd_type"]}',
            'experiment_name': f'{cfg["bd_type"]}_{cfg["dataset_name"]}_{cfg["architecture"]}'
        }

    # 根据攻击类型动态加载参数
    AttackMethod=None
    if cfg['bd_type'] == 'BadNets':
        AttackMethod = BadNets
        if cfg['dataset_name'] == 'IMAGENET10':
            trigger_size = 20
            offset=-1
            # 创建全0模板
            pattern = torch.zeros((3, 224, 224), dtype=torch.uint8)
            # 在右下角放白色方块
            pattern[:, -trigger_size+offset:offset, -trigger_size+offset:offset] = 255

            weight = torch.zeros((3, 224, 224), dtype=torch.float32)
            weight[:, -trigger_size:, -trigger_size:] = 1.0

            cfg['pattern'] = pattern
            cfg['weight'] = weight
            cfg['poisoned_transform_train_index']=2
            cfg['poisoned_transform_test_index']=2
        elif cfg['dataset_name'] == 'GTRSB':
            trigger_size = 5  # 更小的触发器尺寸以适应64x64分辨率
            # 创建全0模板(3通道,64x64)
            pattern = torch.zeros((3, 64, 64), dtype=torch.uint8)
            # 在右下角放置黄色方块（交通标志中更显眼）
            pattern[:, -trigger_size:, -trigger_size:] = 255

            weight = torch.zeros((3, 64, 64), dtype=torch.float32)
            weight[:, -trigger_size:, -trigger_size:] = 1.0

            cfg['pattern'] = pattern
            cfg['weight'] = weight
            #这里的攻击索引要与图像预处理相互对应
            cfg['poisoned_transform_train_index'] = 1  # 在Normalize前应用
            cfg['poisoned_transform_test_index'] = 1


        else:  # 原有逻辑保持不变
            cfg['pattern'] = None
            cfg['weight'] = None
            cfg['poisoned_transform_train_index']=0
            cfg['poisoned_transform_test_index']=0
        attack_params.update({
            'pattern': cfg['pattern'],
            'weight': cfg['weight'],
            'poisoned_transform_train_index': cfg['poisoned_transform_train_index'],
            'poisoned_transform_test_index': cfg['poisoned_transform_test_index'],
            'schedule':schedule_base
        })
        raw_params.update({
            'pattern': cfg['pattern'],
            'weight': cfg['weight'],
            'poisoned_transform_train_index': cfg['poisoned_transform_train_index'],
            'poisoned_transform_test_index': cfg['poisoned_transform_test_index'],
            'schedule':schedule_base
        })
    elif cfg['bd_type'] == 'WaNet':
        AttackMethod = WaNet
        # 根据数据集初始化网格参数
        if cfg['dataset_name'] == 'MNIST':
            size=28
            s = 0.02
            cfg['identity_grid'] = generate_identity_grid(size)
            cfg['noise_grid'] = torch.randn((1, size, size, 2)) * s * size  # 示例噪声
            cfg['poisoned_transform_train_index']=0
            cfg['poisoned_transform_test_index']=0
        elif cfg['dataset_name'] == 'CIFAR10':
            size = 32
            s = 0.03
            cfg['identity_grid'] = generate_identity_grid(size)
            cfg['noise_grid'] = torch.randn((1, size, size, 2)) * s * size  # 示例噪声
            cfg['poisoned_transform_train_index']=0
            cfg['poisoned_transform_test_index']=0
        elif cfg['dataset_name'] == 'IMAGENET10':
            size = 224
            s=0.02
            #0.01-0.1
            cfg['identity_grid'] = torch.zeros((1, size, size, 2), dtype=torch.float32)
            cfg['identity_grid']=generate_identity_grid(size)
            cfg['noise_grid'] = torch.randn((1, size, size, 2))*s*size  # 示例噪声
            cfg['poisoned_transform_train_index']=2
            cfg['poisoned_transform_test_index']=2
        elif cfg['dataset_name'] == 'GTRSB':
            size = 64
            s = 0.02
            cfg['identity_grid'] = generate_identity_grid(size)
            cfg['noise_grid'] = torch.randn((1, size, size, 2)) * s * size  # 示例噪声
            cfg['poisoned_transform_train_index']=1
            cfg['poisoned_transform_test_index']=1

        attack_params.update({
            'identity_grid': cfg['identity_grid'],
            'noise_grid': cfg['noise_grid'],
            'noise': cfg['noise'],
            #'s': cfg['s'],
            #'grid_rescale': cfg['grid_rescale'],
            'poisoned_transform_train_index': cfg['poisoned_transform_train_index'],
            'poisoned_transform_test_index': cfg['poisoned_transform_test_index'],
            'schedule': schedule_base
        })
        raw_params.update({
            'identity_grid': cfg['identity_grid'],
            'noise_grid': cfg['noise_grid'],
            'noise': cfg['noise'],
            #'s': cfg['s'],
            # 'grid_rescale': cfg['grid_rescale'],
            'poisoned_transform_train_index': cfg['poisoned_transform_train_index'],
            'poisoned_transform_test_index': cfg['poisoned_transform_test_index'],
            'schedule': schedule_base
        })

    elif cfg['bd_type'] == 'Blended':
        AttackMethod = Blended

        if cfg['dataset_name'] == 'MNIST':
            size=28
            s=0.3
            cfg['poisoned_transform_train_index'] = 0
            cfg['poisoned_transform_test_index'] = 0
            watermark_type = "text"  # 可选 "gradient", "noise", "text"
            pattern, weight = generate_watermark_trigger(size, type=watermark_type)
            cfg['pattern'] =pattern
            cfg['weight'] =weight*s

        elif cfg['dataset_name'] == 'CIFAR10':
            size=32
            s=0.3
            cfg['poisoned_transform_train_index'] = 0
            cfg['poisoned_transform_test_index'] = 0
            watermark_type = "text"  # 可选 "gradient", "noise", "text"
            pattern, weight = generate_watermark_trigger(size, type=watermark_type)
            cfg['pattern'] =pattern
            cfg['weight'] =weight*s

        elif cfg['dataset_name'] == 'IMAGENET10':
            size=224
            s=0.3
            cfg['poisoned_transform_train_index'] = 2
            cfg['poisoned_transform_test_index'] = 2
            watermark_type = "text"  # 可选 "gradient", "noise", "text"
            pattern, weight = generate_watermark_trigger(size, type=watermark_type)
            cfg['pattern'] =pattern
            cfg['weight'] =weight*s
        elif cfg['dataset_name'] == 'GTRSB':
            size=64
            s=0.3
            cfg['poisoned_transform_train_index'] = 1
            cfg['poisoned_transform_test_index'] = 1
            watermark_type = "text"  # 可选 "gradient", "noise", "text"
            pattern, weight = generate_watermark_trigger(size, type=watermark_type)
            cfg['pattern'] =pattern
            cfg['weight'] =weight*s

        attack_params.update({
            'pattern': cfg['pattern'],
            'weight': cfg['weight'],
            'poisoned_transform_train_index': cfg['poisoned_transform_train_index'],
            'poisoned_transform_test_index': cfg['poisoned_transform_test_index'],
            'schedule': schedule_base
        })
        raw_params.update({
            'pattern': cfg['pattern'],
            'weight': cfg['weight'],
            'poisoned_transform_train_index': cfg['poisoned_transform_train_index'],
            'poisoned_transform_test_index': cfg['poisoned_transform_test_index'],
            'schedule': schedule_base
        })
    else:
        pass
    # 初始化模型
    # model_bd = FlexibleCNN(vgg_name='VGG13')
    # model_raw=FlexibleCNN(vgg_name='VGG13')

    # model_bd=VGG16_dense()
    # model_raw=VGG16_dense()

    # 初始化BadNets
    attacker=AttackMethod(**attack_params)
    rawtrainer=AttackMethod(**raw_params)

    # rawmodel_src_path= "transpace/IMAGENET10_stdvgg16_class10_WaNet_rawA0.pth"
    # bdmodel_src_path= "transpace/IMAGENET10_stdvgg16_class10_WaNet_bdA0.pth"
    rawmodel_src_path= "transpace/IMAGENET10_stdvgg16_class10_BadNets_raw9913.pth"
    bdmodel_src_path= "transpace/IMAGENET10_stdvgg16_class10_BadNets_raw9913.pth"
    if pretrain==True:
        rawtrainer.model.load_state_dict(torch.load(rawmodel_src_path))
        attacker.model.load_state_dict(torch.load(bdmodel_src_path))

    if arg.pretrainfile!=None:
        rawtrainer.model.load_state_dict(torch.load(arg.pretrainfile))
        attacker.model.load_state_dict(torch.load(arg.pretrainfile))

    if train==True:
        if arg.onlybd == False:
            print(f"Training rawnet on {cfg['dataset_name']}...")
            rawtrainer.train()
        # 训练
        print(f"Training bdnet on {cfg['dataset_name']}...")
        attacker.train()
    # 测试

    print("\nTesting rawnet on clean data and poisoned data:")
    rc,rp=rawtrainer.test(test_dataset=attacker.test_dataset,poisoned_test_dataset=attacker.poisoned_test_dataset)


    print("\nTesting bdnet on clean data and poisoned data:")
    bc,bp=attacker.test(test_dataset=attacker.test_dataset,poisoned_test_dataset=attacker.poisoned_test_dataset)


    print("Extended Result: bc~rc  bp>>rp")
    print(f"rc:{rc}  bc:{bc}\nrp:{rp} bp:{bp}")

    # rawmodel_tar_path= "transpace/vgg16std_raw.pth"
    # bdmodel_tar_path= "transpace/vgg16std_bd.pth"
    rawmodel_tar_path= f"transpace/{arg.set}_{arg.arch}_{arg.bdtype}_raw.pth"
    bdmodel_tar_path= f"transpace/{arg.set}_{arg.arch}_{arg.bdtype}_bd.pth"
    if saveRes==True:
        if arg.onlybd==False:
            torch.save(rawtrainer.model.state_dict(), rawmodel_tar_path)
        torch.save(attacker.model.state_dict(),bdmodel_tar_path)

    if arg.savebdset==True:
        # poisioned_data_saveroot = f"./data/poisonedtrain_{arg.set}_{arg.arch}_{arg.bdtype}"
        # clean_data_saveroot = f"./data/cleantrain_{arg.set}_{arg.arch}_{arg.bdtype}"
        poisioned_data_saveroot = f"./data/poisonedval_{arg.set}_{arg.arch}_{arg.bdtype}"
        clean_data_saveroot = f"./data/cleanval_{arg.set}_{arg.arch}_{arg.bdtype}"
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
        #for idx in range(len(attacker.train_dataset)):

            # 获取干净图像
            clean_img, true_label = attacker.test_dataset[idx]

            # 获取毒化样本图像
            poisoned_img, _ = attacker.poisoned_test_dataset[idx]  # 忽略毒化数据集的标签
            #poisoned_img, _ = attacker.poisoned_train_dataset[idx]  # 忽略毒化数据集的标签

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
            img = torch.clamp(img, 0.0, 1.0)  # <-- 关键补充步骤
            # 保存图像（禁用自动归一化）
            torchvision.utils.save_image(img, save_path, normalize=False)

            # 生成文件名
            filename = f'clean_{idx}_true_{true_label}_target_{cfg["target_class"]}.png'
            save_path = os.path.join(clean_class_dir, filename)
            img1 = torch.clamp(img1, 0.0, 1.0)  # <-- 关键补充步骤
            # 保存图像（禁用自动归一化）
            #torchvision.utils.save_image(img1, save_path, normalize=False)




if __name__=="__main__":
    print(0)
