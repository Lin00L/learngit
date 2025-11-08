import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
import numpy as np

# 1. 配置参数与设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64
learning_rate = 0.001
epochs = 15
input_size = 28 * 28
num_classes = 10

# 2. 数据预处理与加载
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(
    root='./data', train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root='./data', train=False, download=True, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# 3. 模型定义
# 3.1 全连接网络（FC）
class FC_Net(nn.Module):
    def __init__(self, input_size, num_classes):
        super(FC_Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = x.view(-1, input_size)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x


# 3.2 卷积神经网络（CNN）
class CNN_Net(nn.Module):
    def __init__(self, num_classes):
        super(CNN_Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 动态计算全连接层输入维度（避免硬编码错误）
        self.fc1 = nn.Linear(64 * 5 * 5, 128)  # 修正：64*5*5=1600（之前错误为64*12*12）
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # 卷积→激活→池化：(batch,1,28,28)→(batch,32,26,26)→(batch,32,13,13)
        x = self.pool(self.relu(self.conv1(x)))
        # 卷积→激活→池化：(batch,32,13,13)→(batch,64,11,11)→(batch,64,5,5)
        x = self.pool(self.relu(self.conv2(x)))
        # 扁平化：自动适配维度（推荐用法，避免硬编码）
        x = x.view(x.size(0), -1)  # 等价于 x.view(-1, 64*5*5)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# 4. 训练与测试函数
def train_model(model, train_loader, criterion, optimizer, epochs):
    model.train()
    train_acc_list = []
    train_time = 0
    for epoch in range(epochs):
        start_time = time.time()
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_time = time.time() - start_time
        train_time += epoch_time
        train_acc = 100 * correct / total
        train_acc_list.append(train_acc)
        print(f'Epoch [{epoch + 1}/{epochs}], Train Acc: {train_acc:.2f}%, Time: {epoch_time:.2f}s')
    return train_acc_list, train_time


def runmodel(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_acc = 100 * correct / total
    return test_acc


# 5. 初始化模型与训练
fc_model = FC_Net(input_size, num_classes).to(device)
cnn_model = CNN_Net(num_classes).to(device)

criterion = nn.CrossEntropyLoss()
fc_optimizer = optim.Adam(fc_model.parameters(), lr=learning_rate)
cnn_optimizer = optim.Adam(cnn_model.parameters(), lr=learning_rate)

# 训练FC模型
print("=" * 50)
print("Training Fully Connected Network (FC)...")
fc_train_acc, fc_train_time = train_model(fc_model, train_loader, criterion, fc_optimizer, epochs)

# 训练CNN模型
print("=" * 50)
print("Training Convolutional Neural Network (CNN)...")
cnn_train_acc, cnn_train_time = train_model(cnn_model, train_loader, criterion, cnn_optimizer, epochs)

# 测试模型
fc_test_acc = runmodel(fc_model, test_loader)
cnn_test_acc = runmodel(cnn_model, test_loader)

# 6. 性能对比与可视化
print("=" * 50)
print("Performance Comparison")
print("=" * 50)
print(f"FC Network - Test Accuracy: {fc_test_acc:.2f}%, Training Time: {fc_train_time:.2f}s")
print(f"CNN Network - Test Accuracy: {cnn_test_acc:.2f}%, Training Time: {cnn_train_time:.2f}s")
print("=" * 50)

# 绘制准确率曲线
plt.figure(figsize=(10, 6))
epochs_range = np.arange(1, epochs + 1)
plt.plot(epochs_range, fc_train_acc, label=f'FC Train Acc (Final Test: {fc_test_acc:.2f}%)',
         linewidth=2, marker='o', markersize=4, color='#1f77b4')
plt.plot(epochs_range, cnn_train_acc, label=f'CNN Train Acc (Final Test: {cnn_test_acc:.2f}%)',
         linewidth=2, marker='s', markersize=4, color='#ff7f0e')

plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.title('MNIST Classification: FC vs CNN Accuracy Curve', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3, linestyle='--')
plt.xticks(epochs_range)
plt.ylim(85, 100)  # 调整y轴范围，更清晰展示差异
plt.tight_layout()
plt.savefig('mnist_fc_cnn_comparison.png', dpi=300, bbox_inches='tight')
plt.show()