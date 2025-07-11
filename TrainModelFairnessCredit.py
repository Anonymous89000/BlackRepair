import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
import numpy as np
import time
import matplotlib.pyplot as plt  # 用于训练过程可视化
import torch.nn.functional as F

# 假设CensusNet网络结构已定义（需提前实现）

class CreditNet(nn.Module):
    def __init__(self, num_of_features):
        super().__init__()
        self.fc1 = nn.Linear(num_of_features, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 8)
        self.fc5 = nn.Linear(8, 2)

    def forward(self, x):
        x = torch.flatten(x,1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.fc5(x)
        output = x # cross entropy in pytorch already includes softmax
        return output

# class CreditNet(nn.Module):
#     def __init__(self, num_of_features):
#         super().__init__()
#         self.fc1 = nn.Linear(num_of_features, 64)
#         self.fc2 = nn.Linear(64, 32)
#         self.fc3 = nn.Linear(32, 16)
#         self.fc4 = nn.Linear(16, 8)
#         self.fc5 = nn.Linear(8, 4)
#         self.fc6 = nn.Linear(4, 2)
#
#     def forward(self, x):
#         x = torch.flatten(x,1)
#         x = self.fc1(x)
#         x = F.relu(x)
#         x = self.fc2(x)
#         x = F.relu(x)
#         x = self.fc3(x)
#         x = F.relu(x)
#         x = self.fc4(x)
#         x = F.relu(x)
#         x = self.fc5(x)
#         x = F.relu(x)
#         x = self.fc6(x)
#         output = x # cross entropy in pytorch already includes softmax
#         return output

    # def __init__(self, input_size):
    #     super(CensusNet, self).__init__()
    #     # 示例结构（需根据实际修改）
    #     self.fc1 = nn.Linear(input_size, 128)
    #     self.relu = nn.ReLU()
    #     self.fc2 = nn.Linear(128, 64)
    #     self.fc3 = nn.Linear(64, 2)  # 假设二分类
    #
    # def forward(self, x):
    #     x = self.fc1(x)
    #     x = self.relu(x)
    #     x = self.fc2(x)
    #     x = self.relu(x)
    #     x = self.fc3(x)
    #     return x


# 训练函数
def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 数据加载与预处理
    train_x = np.loadtxt('data/credit/trainx.txt')
    train_y = np.loadtxt('data/credit/trainy.txt')
    test_x = np.loadtxt('data/credit/testx.txt')
    test_y = np.loadtxt('data/credit/testy.txt')

    # 转换为PyTorch张量
    tensor_train_x = torch.FloatTensor(train_x)
    tensor_train_y = torch.FloatTensor(train_y)
    tensor_test_x = torch.FloatTensor(test_x)
    tensor_test_y = torch.FloatTensor(test_y)

    # 创建数据集和数据加载器
    train_dataset = TensorDataset(tensor_train_x, tensor_train_y)
    test_dataset = TensorDataset(tensor_test_x, tensor_test_y)

    # 合并训练集和测试集
    combined_dataset = ConcatDataset([train_dataset, test_dataset])
    # 重新划分数据集：90%用于训练，10%用于验证
    total_size = len(combined_dataset)
    train_size = int(0.9 * total_size)
    val_size = total_size - train_size

    # 随机划分数据集
    train_subset, val_subset = torch.utils.data.random_split(
        combined_dataset, [train_size, val_size]
    )

    # 创建数据加载器
    batch_size = 50
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size)



    # batch_size = 100
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # 模型初始化
    model = CreditNet(20).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # 训练记录
    train_losses = []
    val_accuracies = []

    # 训练循环
    num_epochs = 1000
    best_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        epoch_start = time.time()

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels.argmax(dim=1))  # 转换为类别索引

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # 计算epoch指标
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)

        # 验证集评估
        current_acc = recal_acc1(model.state_dict())
        val_accuracies.append(current_acc)

        # 保存最佳模型
        if current_acc > best_acc:
            best_acc = current_acc
            torch.save(model.state_dict(), 'credit_best_model.pt')
            #torch.save(model, 'best_model.pt')

        epoch_time = time.time() - epoch_start
        print(f'Epoch [{epoch + 1}/{num_epochs}] - Loss: {epoch_loss:.4f} | '
              f'Acc: {current_acc * 100:.2f}% | Time: {epoch_time:.2f}s')

    # 训练结果可视化
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.title('Training Loss Curve')
    plt.xlabel('Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    plt.savefig('training_metrics.png')

    print(f"\nTraining completed! Best accuracy: {best_acc * 100:.2f}%")
    return model


# 修改后的准确率计算函数（支持传入模型参数）
def recal_acc1(model_state):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_x = np.loadtxt('data/credit/testx.txt')
    test_y = np.loadtxt('data/credit/testy.txt')

    tensor_test_x = torch.FloatTensor(test_x)
    tensor_test_y = torch.FloatTensor(test_y)
    test_dataset = TensorDataset(tensor_test_x, tensor_test_y)
    test_loader = DataLoader(test_dataset, batch_size=100)

    model = CreditNet(20)
    model.load_state_dict(model_state)
    model = model.to(device)
    model.eval()

    correct = 0
    total = len(test_loader.dataset)

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            correct += (pred.argmax(1) == y.argmax(1)).sum().item()

    return correct / total


# 执行训练
if __name__ == "__main__":
    trained_model = train_model()