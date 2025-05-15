

import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import Compose, ToTensor, Normalize





class CNN6_CIFAR10(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super(CNN6_CIFAR10, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # 动态计算全连接层输入尺寸
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels,
                                      32, 32)
            dummy_output = self.features(dummy_input)
            fc_input = dummy_output.view(1, -1).size(1)

        self.classifier = nn.Sequential(
            nn.Linear(fc_input, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class CNN6_MNIST(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(CNN6_MNIST, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # 动态计算全连接层输入尺寸
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels,
                                      28, 28)
            dummy_output = self.features(dummy_input)
            fc_input = dummy_output.view(1, -1).size(1)

        self.classifier = nn.Sequential(
            nn.Linear(fc_input, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x