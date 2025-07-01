import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import os

from model.inner_vgg import VGG16_dense
from model.inner_vgg import VGG13_dense
from model.cnn import CNN6_CIFAR10,CNN6_MNIST
import time

import torchvision.transforms as transforms
from torchvision import datasets
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau

def prepare_datasets(dataset_name):
    """
    预处理函数，根据数据集名称提供适配不同数据集的预处理。

    参数:
    - dataset_name: 数据集名称（'IMAGENET10', 'CIFAR10', 'MNIST'等）

    返回:
    - train_dataset: 训练集数据集
    - test_dataset: 测试集数据集
    - in_channels: 输入图像的通道数
    - img_size: 图像的尺寸
    """

    # 公共的预处理（每个数据集都要应用的操作）
    common_transforms = []
    train_transforms = []  # 专门为训练集设置的数据增强
    test_transforms = []  # 专门为测试集设置的标准预处理

    if dataset_name in ['MNIST', 'CIFAR10']:
        train_transforms.append(transforms.ToTensor())  # 转为Tensor
        test_transforms.append(transforms.ToTensor())
    elif dataset_name == 'IMAGENET10':
        # 训练集：增加数据增强操作
        train_transforms.extend([
            transforms.Resize(256),  # 调整为256x256
            transforms.CenterCrop(224),  # 进行中心裁剪，调整为224x224
            transforms.RandomHorizontalFlip(),  # 随机水平翻转
            transforms.RandomRotation(10),  # 随机旋转10度
            transforms.ToTensor(),  # 转为Tensor
        ])
        # 测试集：只进行标准预处理
        test_transforms.extend([
            transforms.Resize(256),  # 调整为256x256
            transforms.CenterCrop(224),  # 进行中心裁剪，调整为224x224
            transforms.ToTensor(),  # 转为Tensor
        ])
    elif dataset_name == 'GTSRB':  # 新增GTRSB处理
        train_transforms.extend([
            transforms.Resize((64, 64)),  # 统一调整尺寸
            transforms.ToTensor(),
        ])
        test_transforms.extend([
            transforms.Resize((64, 64)),  # 统一调整尺寸
            transforms.ToTensor(),
        ])
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # 数据集特定配置
    if dataset_name == 'MNIST':
        mean, std = (0.1307,), (0.3081,)
        dataset_class = datasets.MNIST
        in_channels = 1
        img_size = 28
    elif dataset_name == 'CIFAR10':
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        dataset_class = datasets.CIFAR10
        in_channels = 3
        img_size = 32
    elif dataset_name == 'IMAGENET10':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        dataset_class = datasets.ImageFolder
        in_channels = 3
        img_size = 224  # ImageNet10的标准尺寸为224
    elif dataset_name == 'GTSRB':  # GTRSB处理
        mean = [0.3403, 0.3121, 0.3214]
        std = [0.2724, 0.2608, 0.2669]
        dataset_class = datasets.ImageFolder
        in_channels = 3
        img_size = 64
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # 添加标准化步骤
    train_transforms.append(transforms.Normalize(mean, std))
    test_transforms.append(transforms.Normalize(mean, std))

    # 数据集加载
    if dataset_name == 'IMAGENET10':
        train_dataset = dataset_class(
            root='./data/imagenet10/train',  # 数据集路径需要根据实际情况修改
            transform=transforms.Compose(train_transforms)
        )
        test_dataset = dataset_class(
            root='./data/imagenet10/val',
            transform=transforms.Compose(test_transforms)
        )
        train_dataset = dataset_class(
            root='./data/imagenet10/train',  # 数据集路径需要根据实际情况修改
            transform=transforms.Compose(train_transforms)
        )
        test_dataset = dataset_class(
            root='./data/imagenet10/val',
            transform=transforms.Compose(test_transforms)
        )

    elif dataset_name == 'GTSRB':
        train_dataset = dataset_class(
            root='./data/gtsrb/train',
            transform=transforms.Compose(train_transforms)
        )
        test_dataset = dataset_class(
            root='./data/gtsrb/val',
            transform=transforms.Compose(test_transforms)
        )
    else:
        train_dataset = dataset_class(
            root='./data',
            train=True,
            download=True,
            transform=transforms.Compose(train_transforms)
        )
        test_dataset = dataset_class(
            root='./data',
            train=False,
            download=True,
            transform=transforms.Compose(test_transforms)
        )

    return train_dataset, test_dataset, in_channels, img_size


def train_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个epoch，返回本epoch的平均损失和准确率"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    start_time = time.time()  # 记录每个epoch开始时的时间

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Calculate running loss and accuracy
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total

    # 计算当前epoch的训练时间
    epoch_time = time.time() - start_time
    return epoch_loss, epoch_acc, epoch_time


def evaluate_model(model, test_loader, device):
    """评估模型在测试集上的准确率"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_acc = 100 * correct / total
    return test_acc


def trainmodel(arg):
    """训练模型并输出训练过程中的准确率和最后的测试准确率"""
    # Configuration and dataset preparation
    dataset_name = arg.set
    arch = arg.arch
    epochs = arg.epochs
    batch_size = arg.batch_size
    lr = arg.lr
    save_dir = arg.save_dir
    pretrainfile = arg.pretrainfile  # 预训练模型路径
    optimizer_type = arg.optimizer  # 选择优化器类型（Adam, SGD）
    scheduler_type = arg.scheduler  # 学习率调度器类型（StepLR, CosineAnnealingLR等）

    # Prepare datasets
    train_dataset, test_dataset, in_channels, img_size = prepare_datasets(dataset_name)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4,pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4,pin_memory=True)

    # 输出训练集和测试集大小
    print(f"Training set size: {len(train_loader.dataset)}")
    print(f"Test set size: {len(test_loader.dataset)}")

    # Model selection based on architecture argument
    if arch == 'stdvgg16_class10':
        model = models.vgg16(pretrained=False)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, 10)  # For 10 classes (ImageNet10)
    elif arch == 'resnet18_class10':
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 10)  # For 10 classes (ImageNet10)
    elif arch == 'resnet34_class10':
        model = models.resnet34(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 10)  # For 10 classes (ImageNet10)
    elif arg.arch == "resnet18_class43":
        # gtsrb
        model = models.resnet18(pretrained=False)

        # 修改第一层卷积：原kernel_size=7改为3，stride=2改为1
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.fc = nn.Linear(512, 43)  # 调整输出层

    elif arch == 'innervgg16':
        model = VGG16_dense()
    elif arch == 'innervgg13':
        model = VGG13_dense(vgg_name='VGG13')
    elif arch == 'CNN6_MNIST':
        model = CNN6_MNIST()
    elif arch == 'CNN6_CIFAR10':
        model = CNN6_CIFAR10()
    else:
        raise ValueError("Unsupported architecture")

    # Move model to device (GPU/CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()

    # Optimizer selection
    if optimizer_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    elif optimizer_type == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    else:
        raise ValueError("Unsupported optimizer")

    # 学习率调度器选择
    if scheduler_type == 'StepLR':
        scheduler = StepLR(optimizer, step_size=20, gamma=0.1)
    elif scheduler_type == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(optimizer, T_max=50)
    elif scheduler_type == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1)
    else:
        raise ValueError("Unsupported scheduler")

    # Load pre-trained weights if the pretrainfile is provided
    if pretrainfile is not None:
        print(f"Loading pre-trained model from {pretrainfile}...")
        model.load_state_dict(torch.load(pretrainfile))
    else:
        print("No pre-trained model file provided. Starting training from scratch.")

    # Training loop
    print(f"Training {arch} on {dataset_name}...")
    for epoch in range(epochs):
        # 输出当前学习率
        print(f"Epoch {epoch + 1}/{epochs} - Current Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

        epoch_loss, epoch_acc, epoch_time = train_epoch(model, train_loader, criterion, optimizer, device)

        # Print intermediate results (every epoch)
        print(
            f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%, Time: {epoch_time:.2f}s")

        # Save the model every few epochs
        if (epoch + 1) % 5 == 0:
            temp_test_acc = evaluate_model(model, test_loader, device)
            print(f"Test Accuracy on {dataset_name}: {temp_test_acc:.2f}%")
            save_path = os.path.join(save_dir, f'{arch}_epoch{epoch + 1}.pth')
            torch.save(model.state_dict(), save_path)
            print(f"Model saved at {save_path}")

        # 更新学习率
        scheduler.step()

    # Test the model after training
    final_test_acc = evaluate_model(model, test_loader, device)
    print(f"Final Test Accuracy on {dataset_name}: {final_test_acc:.2f}%")

    # Save the final model
    final_model_path = os.path.join(save_dir, f'{arch}_final.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved at {final_model_path}")


if __name__ == "__main__":
    # Assuming that the args are passed through command line or from another script
    pass