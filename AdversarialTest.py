import os
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
from torchvision.models import vgg16, resnet18, resnet34
from torch import nn
from model.inner_vgg import VGG16_dense, VGG13_dense
from model.cnn import CNN6_CIFAR10, CNN6_MNIST
from  LogUtil import  Log
import  time

def get_dataset_stats(dataset_name):
    """根据数据集名称获取归一化参数"""
    if dataset_name == 'MNIST':
        mean = (0.1307,)
        std = (0.3081,)
        in_channels=1
    elif dataset_name == 'CIFAR10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        in_channels=3
    elif dataset_name == 'IMAGENET10':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        in_channels=3
    elif dataset_name == 'GTSRB':
        mean = [0.3403, 0.3121, 0.3214]
        std = [0.2724, 0.2608, 0.2669]
        in_channels=3
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return mean, std,in_channels




def create_model(arch, dataset_name):
    """根据架构和数据集创建模型"""
    if arch == 'stdvgg16_class10':
        model = vgg16(pretrained=False)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, 10)
    elif arch == 'resnet18_class10':
        model = resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, 10)
    elif arch == 'resnet18_class43':
        model = resnet18(pretrained=False)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.fc = nn.Linear(512, 43)
    elif arch == 'innervgg16':
        model = VGG16_dense()
    elif arch == 'innervgg13':
        model = VGG13_dense('VGG13')
    elif arch == 'CNN6_MNIST':
        model = CNN6_MNIST()
    elif arch == 'CNN6_CIFAR10':
        model = CNN6_CIFAR10()
    else:
        raise ValueError(f"Unsupported architecture: {arch}")
    return model


def load_test_dataset(dataset_name, data_dir='./data'):
    """加载原始测试集（仅预处理部分参考文档1）"""
    transform = []
    if dataset_name == 'IMAGENET10':
        # ImageNet标准预处理流程
        transform.extend([
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ])
    elif dataset_name == 'GTSRB':  # 新增GTSRB处理
        transform.extend([
            transforms.Resize((64, 64)),  # 统一调整尺寸
        ])
    else:
        pass

    transform.append(transforms.ToTensor())
    mean, std, in_channels = get_dataset_stats(dataset_name)
    transform.append(transforms.Normalize(mean, std))

    transform = transforms.Compose(transform)


    if dataset_name == 'MNIST':
        return datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
    elif dataset_name == 'CIFAR10':
        return datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
    elif dataset_name == 'IMAGENET10':
        return datasets.ImageFolder(root=os.path.join(data_dir, 'imagenet10/val'), transform=transform)
    elif dataset_name == 'GTSRB':
        return datasets.ImageFolder(root=os.path.join(data_dir, 'gtsrb/val'), transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


def adversarialtset(arg):
    # 设备配置
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



    # 1. 加载原始测试集和对抗样本数据集
    test_dataset = load_test_dataset(arg.set)
    mean, std,in_channels = get_dataset_stats(arg.set)

    # 对抗样本路径
    #adv_root = f'./data/AdAttaked_{arg.adtype}_{arg.set}'

    adv_root=arg.advdataset

    log_filename = f"{adv_root}.log"  # 仅扩展名不同
    log_path = os.path.join('./logs', log_filename)

    # 初始化日志（自动创建目录）
    log = Log(log_path, mode='w', level='INFO', verbose=True)
    log.info(f"Adversarial Test Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    transform_steps = []
    if arg.set == 'IMAGENET10':
        # ImageNet标准预处理流程
        transform_steps.extend([
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ])
    elif arg.set == 'GTSRB':  # 新增GTRSB处理
        transform_steps.extend([
            transforms.Resize((64, 64)),  # 统一调整尺寸
        ])
    else:
        pass

    # 动态构建转换流程
    transform_steps.append(transforms.ToTensor())
    # 通道修正（MNIST单通道特殊处理）
    if in_channels == 1:
        transform_steps.append(
            transforms.Lambda(lambda x: x[:1, :, :])  # 取第一个通道
        )
    transform_steps.append(transforms.Normalize(mean, std))


    adv_transform = transforms.Compose(transform_steps)
    adv_dataset = datasets.ImageFolder(root=adv_root, transform=adv_transform)


    # 2. 加载模型
    model = create_model(arg.arch, arg.set)
    model.load_state_dict(torch.load(arg.pretrainfile))
    model = model.to(device)
    model.eval()




    batch_size=32
    # 3. 获取原始测试集的正确预测索引
    clean_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    correct_indices = []
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(clean_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct = (preds == labels).cpu().numpy()
            batch_indices = [batch_idx * batch_size + i for i, c in enumerate(correct) if c]
            correct_indices.extend(batch_indices)

    #print(correct_indices)
    # 4. 根据参数过滤对抗样本
    if  arg.includewrong==False:
        adv_dataset = Subset(adv_dataset, correct_indices)

    # 5. 测试干净数据集准确率
    clean_correct = 0
    clean_total = 0
    with torch.no_grad():
        for images, labels in clean_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            clean_correct += (outputs.argmax(dim=1) == labels).sum().item()
            clean_total += labels.size(0)
    clean_acc = clean_correct / clean_total

    # 6. 测试对抗样本准确率
    adv_loader = DataLoader(adv_dataset, batch_size=batch_size, shuffle=False)
    adv_correct = 0
    adv_total = 0
    with torch.no_grad():
        for images, labels in adv_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            adv_correct += (outputs.argmax(dim=1) == labels).sum().item()
            adv_total += labels.size(0)
    adv_acc = adv_correct / adv_total if adv_total > 0 else 0.0



    log.info("\nTest Results:")
    log.info(f"clean len:{len(test_dataset)}")
    log.info(f"adv len:{len(adv_dataset)}")
    log.info(f"| {'Clean Accuracy':<20} | {clean_acc:.4f} |")
    log.info(f"| {'Adversarial Accuracy':<20} | {adv_acc:.4f} |")
    log.info(f"| {'Accuracy Drop(ASR)':<20} | {clean_acc - adv_acc:.4f} |")
    log.info("END TEST")
    return clean_acc, adv_acc


# # 示例用法（需要配合arg对象）
# class Args:
#     def __init__(self):
#         self.set = 'MNIST'  # 数据集名称
#         self.adtype = 'PGD'  # 对抗攻击类型
#         self.arch = 'CNN6_MNIST'  # 模型架构
#         self.pretrainfile = './model_weights/mnist_cnn.pth'  # 模型权重路径
#         self.batch_size = 64
#         self.include_wrong = False  # 是否包含原预测错误样本


if __name__ == '__main__':
    pass
