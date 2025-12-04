import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# 设置随机种子
torch.manual_seed(42)

# 1. 加载MNIST数据集
data = np.load('mnist.npz')
x_train, y_train = data['x_train'], data['y_train']
x_test, y_test = data['x_test'], data['y_test']

# 转换为PyTorch张量并归一化
x_train = torch.from_numpy(x_train).float().unsqueeze(1) / 255.0
y_train = torch.from_numpy(y_train).long()
x_test = torch.from_numpy(x_test).float().unsqueeze(1) / 255.0
y_test = torch.from_numpy(y_test).long()

# 创建数据加载器
train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)


# 2. 定义GoogleNet
class InceptionA(nn.Module):
    def __init__(self, in_channels):
        super(InceptionA, self).__init__()
        # 减少通道数，适应MNIST简单特征
        self.branch1x1 = nn.Conv2d(in_channels, 8, kernel_size=1)

        self.branch5x5_1 = nn.Conv2d(in_channels, 8, kernel_size=1)
        self.branch5x5_2 = nn.Conv2d(8, 12, kernel_size=5, padding=2)

        self.branch3x3_1 = nn.Conv2d(in_channels, 8, kernel_size=1)
        self.branch3x3_2 = nn.Conv2d(8, 12, kernel_size=3, padding=1)
        self.branch3x3_3 = nn.Conv2d(12, 12, kernel_size=3, padding=1)

        self.branch_pool = nn.Conv2d(in_channels, 12, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3, branch_pool]
        return torch.cat(outputs, dim=1)


class GoogleNet(nn.Module):
    def __init__(self):
        super(GoogleNet, self).__init__()
        # 更简单的初始卷积层
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)

        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)

        self.incep1 = InceptionA(in_channels=16)
        self.incep2 = InceptionA(in_channels=44)  # 8+12+12+12 = 44

        self.mp = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.25)

        # 计算全连接层输入大小
        # 28x28 -> conv1+pool -> 14x14 -> conv2+pool -> 7x7
        # 经过两个Inception模块后仍为7x7
        self.fc1 = nn.Linear(44 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.mp(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.mp(x)

        x = self.incep1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.incep2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


# 3. 定义ResNet
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.channels = channels
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.mp = nn.MaxPool2d(2)
        self.rblock1 = ResidualBlock(16)
        self.rblock2 = ResidualBlock(32)
        self.dropout = nn.Dropout(0.25)
        # 计算全连接层输入大小
        # 28x28 -> conv1+pool -> 14x14 -> conv2+pool -> 7x7
        self.fc = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        in_size = x.size(0)
        x = self.mp(F.relu(self.bn1(self.conv1(x))))
        x = self.rblock1(x)
        x = self.dropout(x)

        x = self.mp(F.relu(self.bn2(self.conv2(x))))
        x = self.rblock2(x)
        x = self.dropout(x)

        x = x.view(in_size, -1)
        x = self.fc(x)
        return x


# 4. 训练和评估函数
def train_model(model, train_loader, test_loader, epochs=10, lr=0.01):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # 移除verbose参数以兼容旧版本PyTorch
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    train_losses = []
    test_accuracies = []
    test_f1_scores = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)

        # 评估
        model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                preds = output.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

        acc = accuracy_score(all_targets, all_preds)
        f1 = f1_score(all_targets, all_preds, average='weighted')
        test_accuracies.append(acc)
        test_f1_scores.append(f1)

        # 记录调整前的学习率
        old_lr = optimizer.param_groups[0]['lr']
        # 更新学习率
        scheduler.step(acc)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != old_lr:
            print(f'  Learning rate reduced from {old_lr} to {new_lr}')

        print(f'Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Test Acc: {acc:.4f}, Test F1: {f1:.4f}')

    return model, train_losses, test_accuracies, test_f1_scores


# 5. 训练两个模型
print("Training GoogleNet...")
googlenet = GoogleNet()
googlenet, gl_train_loss, gl_test_acc, gl_test_f1 = train_model(googlenet, train_loader, test_loader, epochs=10,
                                                                lr=0.001)

print("\nTraining ResNet...")
resnet = ResNet()
resnet, rn_train_loss, rn_test_acc, rn_test_f1 = train_model(resnet, train_loader, test_loader, epochs=10, lr=0.001)

# 6. 可视化训练过程
plt.figure(figsize=(12, 8))

# 训练损失
plt.subplot(2, 2, 1)
plt.plot(gl_train_loss, label='GoogleNet')
plt.plot(rn_train_loss, label='ResNet')
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('Training Loss Comparison')
plt.legend()
plt.grid(True)

# 测试准确率（
plt.subplot(2, 2, 2)
plt.plot(gl_test_acc, label='GoogleNet')
plt.plot(rn_test_acc, label='ResNet')
plt.xlabel('Epoch')
plt.ylabel('Test Accuracy')
plt.title('Test Accuracy Comparison')
plt.ylim(0.9, 1.0)
plt.legend()
plt.grid(True)

# 测试F1分数（
plt.subplot(2, 2, 3)
plt.plot(gl_test_f1, label='GoogleNet')
plt.plot(rn_test_f1, label='ResNet')
plt.xlabel('Epoch')
plt.ylabel('Test F1 Score')
plt.title('Test F1 Score Comparison')
plt.ylim(0.9, 1.0)  # 设置纵轴范围为0.9到1
plt.legend()
plt.grid(True)

# 最终性能比较
final_metrics = {
    'Model': ['GoogleNet', 'ResNet'],
    'Final Accuracy': [gl_test_acc[-1], rn_test_acc[-1]],
    'Final F1 Score': [gl_test_f1[-1], rn_test_f1[-1]]
}

plt.subplot(2, 2, 4)
x = np.arange(2)
width = 0.35
plt.bar(x - width / 2, [gl_test_acc[-1], rn_test_acc[-1]], width, label='Accuracy')
plt.bar(x + width / 2, [gl_test_f1[-1], rn_test_f1[-1]], width, label='F1 Score')
plt.xticks(x, ['GoogleNet', 'ResNet'])
plt.ylabel('Score')
plt.title('Final Performance Comparison')
plt.legend()
plt.grid(True, axis='y')

plt.tight_layout()
plt.savefig('performance_comparison.png', dpi=300)
plt.show()


# 7. 混淆矩阵
def plot_confusion_matrix(model, test_loader, title):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            preds = output.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {title}')
    plt.savefig(f'confusion_matrix_{title}.png', dpi=300)
    plt.show()

    return cm


print("\nGenerating confusion matrices...")
gl_cm = plot_confusion_matrix(googlenet, test_loader, 'GoogleNet')
rn_cm = plot_confusion_matrix(resnet, test_loader, 'ResNet')

# 8. 打印最终性能指标
print("\n" + "=" * 50)
print("FINAL PERFORMANCE SUMMARY")
print("=" * 50)
print(f"{'Model':<15} {'Accuracy':<12} {'F1 Score':<12}")
print("-" * 50)
print(f"{'GoogleNet':<15} {gl_test_acc[-1]:<12.4f} {gl_test_f1[-1]:<12.4f}")
print(f"{'ResNet':<15} {rn_test_acc[-1]:<12.4f} {rn_test_f1[-1]:<12.4f}")
print("=" * 50)