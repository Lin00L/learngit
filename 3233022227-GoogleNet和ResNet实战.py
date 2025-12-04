import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time

# ==========================================
# 1. 数据集准备 (Data Preparation)
# ==========================================
# 配置参数
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MNIST 预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST 均值和标准差
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"使用的设备: {DEVICE}")


# ==========================================
# 2. 模型定义 (Model Definitions)
# ==========================================

# ------------------------------------------
# Model A: GoogleNet (Inception) - 适配版
# ------------------------------------------

# ### 关键代码 START: Inception 模块实现 ###
class InceptionBlock(nn.Module):
    def __init__(self, in_channels, c1, c2, c3, c4):
        super(InceptionBlock, self).__init__()
        # 线路 1: 1x1 卷积
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)

        # 线路 2: 1x1 卷积 -> 3x3 卷积
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)

        # 线路 3: 1x1 卷积 -> 5x5 卷积
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)

        # 线路 4: 3x3 最大池化 -> 1x1 卷积
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))

        # 在通道维度 (dim=1) 拼接
        return torch.cat((p1, p2, p3, p4), dim=1)


# ### 关键代码 END: Inception 模块实现 ###

class MiniGoogleNet(nn.Module):
    def __init__(self):
        super(MiniGoogleNet, self).__init__()
        # 初始卷积层，适应 MNIST 单通道输入
        self.pre_layers = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),  # 28 -> 24
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)  # 24 -> 12
        )

        # ### 关键代码 START: GoogleNet 网络构建 ###
        # 使用两个 Inception 块
        # 参数格式: in_c, c1, (c2_reduce, c2), (c3_reduce, c3), c4
        self.inception1 = InceptionBlock(10, 16, (16, 32), (16, 16), 16)
        # Output channels = 16 + 32 + 16 + 16 = 80

        self.inception2 = InceptionBlock(80, 32, (32, 64), (16, 32), 32)
        # Output channels = 32 + 64 + 32 + 32 = 160
        # ### 关键代码 END: GoogleNet 网络构建 ###

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(160, 10)

    def forward(self, x):
        x = self.pre_layers(x)
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# ------------------------------------------
# Model B: ResNet - 适配版
# ------------------------------------------

# ### 关键代码 START: 残差块 (Residual Block) 实现 ###
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        # 主路径 (F(x))
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 捷径 (Shortcut/Identity mapping)
        self.shortcut = nn.Sequential()
        # 如果输入输出维度不一致，需要用 1x1 卷积调整捷径的维度
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # 核心：F(x) + x
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# ### 关键代码 END: 残差块 (Residual Block) 实现 ###

class MiniResNet(nn.Module):
    def __init__(self):
        super(MiniResNet, self).__init__()
        self.in_channels = 16

        # 初始层
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        # ### 关键代码 START: ResNet 网络构建 ###
        # 堆叠残差块
        self.layer1 = self._make_layer(16, 2, stride=1)
        self.layer2 = self._make_layer(32, 2, stride=2)  # 下采样
        self.layer3 = self._make_layer(64, 2, stride=2)  # 下采样
        # ### 关键代码 END: ResNet 网络构建 ###

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 10)

    def _make_layer(self, out_channels, blocks, stride):
        layers = []
        # 第一个块可能涉及到 stride (下采样)
        layers.append(ResidualBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        # 后续块 stride 保持为 1
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


# ==========================================
# 3. 训练与评估函数 (Training & Evaluation)
# ==========================================

def train_and_evaluate(model, model_name):
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    loss_history = []
    acc_history = []

    print(f"\n--- 开始训练 {model_name} ---")
    start_time = time.time()

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # 计算该 epoch 的平均 Loss
        avg_loss = running_loss / len(train_loader)
        loss_history.append(avg_loss)

        # 验证集准确率
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        accuracy = 100 * correct / total
        acc_history.append(accuracy)

        print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

    print(f"{model_name} 训练完成, 耗时: {time.time() - start_time:.1f}s")
    return loss_history, acc_history


# ==========================================
# 4. 主程序执行与可视化 (Main & Visualization)
# ==========================================

if __name__ == "__main__":
    # 实例化模型
    googlenet = MiniGoogleNet()
    resnet = MiniResNet()

    # 训练 GoogleNet
    g_loss, g_acc = train_and_evaluate(googlenet, "GoogleNet (Inception)")

    # 训练 ResNet
    r_loss, r_acc = train_and_evaluate(resnet, "ResNet (Residual)")

    # ### 关键代码 START: 结果可视化 ###
    plt.figure(figsize=(12, 5))

    # 绘制 Loss 曲线
    plt.subplot(1, 2, 1)
    plt.plot(range(1, EPOCHS + 1), g_loss, label='GoogleNet', marker='o')
    plt.plot(range(1, EPOCHS + 1), r_loss, label='ResNet', marker='s')
    plt.title('Training Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # 绘制 Accuracy 曲线
    plt.subplot(1, 2, 2)
    plt.plot(range(1, EPOCHS + 1), g_acc, label='GoogleNet', marker='o')
    plt.plot(range(1, EPOCHS + 1), r_acc, label='ResNet', marker='s')
    plt.title('Test Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
    # ### 关键代码 END: 结果可视化 ###