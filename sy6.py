import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score
import os
import warnings

warnings.filterwarnings('ignore')

# 设置环境变量解决OpenMP冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class EcologicalDataset(Dataset):
    def __init__(self, filepath):
        # 读取CSV文件
        data = pd.read_csv(filepath)
        print(f"原始数据形状: {data.shape}")
        print(f"数据列: {list(data.columns)}")

        # 清理数据 - 处理货币格式
        data_clean = data.copy()
        if 'GDP per Capita' in data_clean.columns:
            data_clean['GDP per Capita'] = data_clean['GDP per Capita'].astype(str).str.replace('$', '', regex=False)
            data_clean['GDP per Capita'] = data_clean['GDP per Capita'].str.replace(',', '', regex=False)
            data_clean['GDP per Capita'] = pd.to_numeric(data_clean['GDP per Capita'], errors='coerce')

        # 选择数值型特征
        numeric_cols = data_clean.select_dtypes(include=[np.number]).columns
        print(f"数值列: {list(numeric_cols)}")

        # 处理缺失值
        data_clean[numeric_cols] = data_clean[numeric_cols].fillna(data_clean[numeric_cols].median())

        # 选择特征列（使用所有可用的数值列，排除目标列）
        # 先创建目标变量
        if 'Total Ecological Footprint' in data_clean.columns:
            median_footprint = data_clean['Total Ecological Footprint'].median()
            data_clean['target'] = (data_clean['Total Ecological Footprint'] > median_footprint).astype(np.float32)
            print(
                f"目标变量 - 高生态足迹(1): {(data_clean['target'] == 1).sum()}, 低生态足迹(0): {(data_clean['target'] == 0).sum()}")
            # 排除目标列和相关列
            exclude_cols = ['target', 'Total Ecological Footprint', 'Biocapacity Deficit or Reserve']
        else:
            # 备用目标：使用GDP per Capita
            median_gdp = data_clean['GDP per Capita'].median()
            data_clean['target'] = (data_clean['GDP per Capita'] > median_gdp).astype(np.float32)
            exclude_cols = ['target', 'GDP per Capita']

        # 选择特征列（排除目标列和相关列）
        available_features = [col for col in numeric_cols if col not in exclude_cols]

        # 如果特征太多，选择前8个
        if len(available_features) > 8:
            available_features = available_features[:8]

        print(f"使用的特征列: {available_features}")

        # 提取特征和目标
        xy = data_clean[available_features + ['target']].values.astype(np.float32)

        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, :-1])  # 特征
        self.y_data = torch.from_numpy(xy[:, [-1]])  # 目标

        print(f"数据集加载完成: {self.len} 个样本")
        print(f"特征维度: {self.x_data.shape[1]}")
        print(f"正样本比例: {data_clean['target'].mean():.3f}")

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


class EcologicalModel(nn.Module):
    def __init__(self, input_size):
        super(EcologicalModel, self).__init__()
        # 5层神经网络：输入层 → 7 → 6 → 5 → 输出层
        self.linear1 = nn.Linear(input_size, 7)
        self.linear2 = nn.Linear(7, 6)
        self.linear3 = nn.Linear(6, 5)
        self.linear4 = nn.Linear(5, 1)
        self.sigmoid = nn.Sigmoid()

        # 添加BatchNorm层帮助训练
        self.bn1 = nn.BatchNorm1d(7)
        self.bn2 = nn.BatchNorm1d(6)
        self.bn3 = nn.BatchNorm1d(5)

    def forward(self, x):
        x = torch.relu(self.bn1(self.linear1(x)))
        x = torch.relu(self.bn2(self.linear2(x)))
        x = torch.relu(self.bn3(self.linear3(x)))
        x = self.sigmoid(self.linear4(x))
        return x


def calculate_metrics(y_true, y_pred):
    """计算准确率和精准率 - 修复版本"""
    y_pred_binary = (y_pred > 0.5).astype(int)
    y_true_binary = y_true.astype(int)

    accuracy = accuracy_score(y_true_binary, y_pred_binary)

    # 修复精准率计算
    if np.sum(y_pred_binary) > 0:
        precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
    else:
        precision = 0.0

    return accuracy, precision, np.sum(y_pred_binary), np.sum(y_true_binary)


def plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies, train_precisions, val_precisions):
    """绘制训练过程可视化图像 - 修复版本"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

    # 损失曲线
    ax1.plot(train_losses, label='训练损失', linewidth=2)
    ax1.plot(val_losses, label='验证损失', linewidth=2)
    ax1.set_title('训练和验证损失')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 准确率曲线
    ax2.plot(train_accuracies, label='训练准确率', linewidth=2)
    ax2.plot(val_accuracies, label='验证准确率', linewidth=2)
    ax2.set_title('训练和验证准确率')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim(0, 1)  # 固定y轴范围
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 精准率曲线 - 修复负值问题
    train_precisions = np.maximum(train_precisions, 0)  # 确保没有负值
    val_precisions = np.maximum(val_precisions, 0)  # 确保没有负值

    ax3.plot(train_precisions, label='训练精准率', linewidth=2)
    ax3.plot(val_precisions, label='验证精准率', linewidth=2)
    ax3.set_title('训练和验证精准率')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Precision')
    ax3.set_ylim(0, 1)  # 固定y轴范围
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ecological_training_history_fixed.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)

    # 主程序入口
    try:
        dataset = EcologicalDataset('countries.csv')
    except Exception as e:
        print(f"数据加载错误: {e}")
        print("请确保countries.csv文件存在且格式正确")
        exit(1)

    # 数据加载器配置
    # 1. 调整batch-size：基于4GB显存，最大16
    batch_size = min(16, len(dataset))

    # 2. 调整num_workers：Windows下CPU物理核心2倍，最大2
    cpu_physical_cores = os.cpu_count() // 2 if os.cpu_count() else 2
    num_workers = min(cpu_physical_cores, 2)

    print(f"\n批量大小: {batch_size}")
    print(f"工作进程数: {num_workers}")
    print(f"CPU物理核心数: {cpu_physical_cores}")

    # 划分训练集和验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)

    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers)

    # 获取输入特征维度
    input_size = dataset.x_data.shape[1]
    print(f"输入特征维度: {input_size}")

    # 初始化模型
    model = EcologicalModel(input_size)

    # 使用改进的优化器和学习率
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 改用Adam优化器

    # 记录训练过程
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    train_precisions = []
    val_precisions = []

    best_val_loss = float('inf')
    best_model_path = 'best_ecological_model.pt'

    print("\n开始训练...")

    for epoch in range(70):
        # 训练阶段
        model.train()
        epoch_train_loss = 0.0
        train_preds = []
        train_targets = []

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data

            # 梯度清零
            optimizer.zero_grad()

            # 前向传播
            y_pred = model(inputs)
            loss = criterion(y_pred, labels)

            # 反向传播
            loss.backward()

            # 更新参数
            optimizer.step()

            epoch_train_loss += loss.item()
            train_preds.extend(y_pred.detach().numpy())
            train_targets.extend(labels.numpy())

        # 计算训练指标
        train_accuracy, train_precision, train_pos_pred, train_pos_true = calculate_metrics(
            np.array(train_targets), np.array(train_preds)
        )
        avg_train_loss = epoch_train_loss / len(train_loader)

        # 验证阶段
        model.eval()
        epoch_val_loss = 0.0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for i, data in enumerate(val_loader, 0):
                inputs, labels = data
                y_pred = model(inputs)
                loss = criterion(y_pred, labels)
                epoch_val_loss += loss.item()

                val_preds.extend(y_pred.numpy())
                val_targets.extend(labels.numpy())

        # 计算验证指标
        val_accuracy, val_precision, val_pos_pred, val_pos_true = calculate_metrics(
            np.array(val_targets), np.array(val_preds)
        )
        avg_val_loss = epoch_val_loss / len(val_loader)

        # 记录指标
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        train_precisions.append(train_precision)
        val_precisions.append(val_precision)

        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
                'input_size': input_size
            }, best_model_path)

        if epoch % 10 == 0 or epoch < 5:
            print(f'Epoch {epoch:3d}:')
            print(f'  训练 - 损失: {avg_train_loss:.4f}, 准确率: {train_accuracy:.4f}, 精准率: {train_precision:.4f}')
            print(f'          预测正例: {train_pos_pred:2d}, 真实正例: {train_pos_true:2d}')
            print(f'  验证 - 损失: {avg_val_loss:.4f}, 准确率: {val_accuracy:.4f}, 精准率: {val_precision:.4f}')
            print(f'          预测正例: {val_pos_pred:2d}, 真实正例: {val_pos_true:2d}')
            if avg_val_loss == best_val_loss:
                print(f'  *** 最佳模型已保存 (损失: {best_val_loss:.4f}) ***')
            print('-' * 60)

    # 绘制训练历史
    plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies, train_precisions, val_precisions)

    # 加载最佳模型进行最终评估
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    # 最终评估
    model.eval()
    final_preds = []
    final_targets = []

    with torch.no_grad():
        for data in val_loader:
            inputs, labels = data
            y_pred = model(inputs)
            final_preds.extend(y_pred.numpy())
            final_targets.extend(labels.numpy())

    final_accuracy, final_precision, final_pos_pred, final_pos_true = calculate_metrics(
        np.array(final_targets), np.array(final_preds)
    )

    print(f"\n=== 最终模型评估 ===")
    print(f"准确率: {final_accuracy:.4f}")
    print(f"精准率: {final_precision:.4f}")
    print(f"预测正例数: {final_pos_pred}, 真实正例数: {final_pos_true}")

    # 保存最终模型
    torch.save(model.state_dict(), 'final_ecological_model.pt')
    print("最终模型已保存为: final_ecological_model.pt")