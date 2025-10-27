# 导入所需库
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import warnings

warnings.filterwarnings('ignore')


# 1. 数据预处理（统一类型：所有列→float64）
def preprocess_data(file_path):

    # 1. 读取数据
    df = pd.read_csv(file_path)

    # 2. 处理缺失值
    df['bmi'].fillna(df['bmi'].median(), inplace=True)

    # 3. 处理异常值
    df['bmi'] = df['bmi'].clip(upper=60)  # 超出一般来说肯定为异常值
    df['avg_glucose_level'] = df['avg_glucose_level'].clip(upper=300)#同上对异常值进行处理

    # 4. 处理特殊分类（如gender的"Other"）
    df['gender'] = df['gender'].replace('Other', 'Male')  # 归为多数类，不影响分布

    # 5. 独热编码分类特征
    categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    df_encoded = pd.get_dummies(
        df.drop('id', axis=1),  # 排除无意义的id列
        columns=categorical_cols,
        drop_first=True,  # 减少冗余特征
        dtype=np.float64  # 关键：独热编码直接生成float64，而非bool
    )

    # 6. 数值特征标准化
    numerical_cols = ['age', 'avg_glucose_level', 'bmi', 'hypertension', 'heart_disease']
    scaler = StandardScaler()
    df_encoded[numerical_cols] = scaler.fit_transform(df_encoded[numerical_cols])

    # 7. 强制所有列转为float64（彻底解决类型问题）
    df_encoded = df_encoded.astype(np.float64)

    # 8. 最终类型检查（ debug用，可保留确认）
    print("预处理后特征类型（确保无object/bool）：")
    print(f"所有列类型: {df_encoded.dtypes.unique()}")  # 应只显示float64
    print(f"特征维度: {df_encoded.drop('stroke', axis=1).shape[1]}")

    # 9. 返回结果
    input_dim = df_encoded.drop('stroke', axis=1).shape[1]
    return df_encoded, input_dim, scaler


# ===================== 2. 自定义数据集类（增加类型二次校验） =====================
class StrokeDataset(Dataset):
    def __init__(self, data):
        # 二次校验：确保特征列全为float64
        features_df = data.drop('stroke', axis=1)
        assert features_df.dtypes.unique().tolist() == [np.float64], \
            f"特征列存在非float64类型：{features_df.dtypes.unique()}"

        # 转换为张量（float64→float32，适配PyTorch常用类型）
        self.features = torch.tensor(features_df.values, dtype=torch.float32)
        self.labels = torch.tensor(data['stroke'].values, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# ===================== 3. 全连接神经网络模型（无修改） =====================
class StrokeNet(nn.Module):
    def __init__(self, input_dim):
        super(StrokeNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.BatchNorm1d(32),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


# ===================== 4. 模型训练与评估（无修改） =====================
def train_model(model, train_loader, test_loader, device, epochs=50, lr=1e-3):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    train_history = {'loss': [], 'acc': []}
    test_history = {'loss': [], 'acc': []}
    best_acc = 0.0
    best_model_state = None

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        total_train = 0

        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * features.size(0)
            predicted = (outputs > 0.5).float()
            train_correct += (predicted == labels).sum().item()
            total_train += labels.size(0)

        avg_train_loss = train_loss / total_train
        train_acc = train_correct / total_train
        train_history['loss'].append(avg_train_loss)
        train_history['acc'].append(train_acc)

        # 测试阶段
        model.eval()
        test_loss = 0.0
        test_correct = 0
        total_test = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for features, labels in test_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)

                test_loss += loss.item() * features.size(0)
                predicted = (outputs > 0.5).float()
                test_correct += (predicted == labels).sum().item()
                total_test += labels.size(0)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_test_loss = test_loss / total_test
        test_acc = test_correct / total_test
        test_history['loss'].append(avg_test_loss)
        test_history['acc'].append(test_acc)

        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            best_model_state = model.state_dict().copy()

        # 打印日志
        print(f"Epoch [{epoch + 1}/{epochs}] | "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Test Loss: {avg_test_loss:.4f}, Test Acc: {test_acc:.4f} | "
              f"Best Test Acc: {best_acc:.4f}")

    model.load_state_dict(best_model_state)
    return model, train_history, test_history, all_preds, all_labels


# ===================== 5. 训练过程可视化（无修改） =====================
def plot_train_history(train_history, test_history, save_path='train_history.png'):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(figsize=(12, 5))

    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_history['loss'], label='训练损失', color='#1f77b4', linewidth=2)
    plt.plot(test_history['loss'], label='测试损失', color='#ff7f0e', linewidth=2)
    plt.xlabel('训练轮次', fontsize=11)
    plt.ylabel('损失值', fontsize=11)
    plt.title('训练与测试损失曲线', fontsize=12, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)

    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_history['acc'], label='训练准确率', color='#1f77b4', linewidth=2)
    plt.plot(test_history['acc'], label='测试准确率', color='#ff7f0e', linewidth=2)
    plt.xlabel('训练轮次', fontsize=11)
    plt.ylabel('准确率', fontsize=11)
    plt.title('训练与测试准确率曲线', fontsize=12, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n训练历史图已保存至: {save_path}")


# ===================== 6. 模型保存与加载（无修改） =====================
def save_model(model, save_path='best_stroke_model.pt', scaler=None):
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler': scaler
    }, save_path)
    print(f"最佳模型已保存至: {save_path}")


def load_model(load_path, input_dim, device):
    checkpoint = torch.load(load_path, map_location=device)
    model = StrokeNet(input_dim).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    scaler = checkpoint['scaler']
    print(f"模型已从 {load_path} 加载完成")
    return model, scaler


# ===================== 7. 主函数（确认数据路径+类型校验） =====================
def main():
    # 1. 配置参数（务必修改为你的数据实际路径！）
    DATA_PATH = 'healthcare-dataset-stroke-data.csv'
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 1e-3
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {DEVICE}")

    # 2. 数据预处理（已强制全float64）
    print("\n=== 开始数据预处理 ===")
    processed_data, input_dim, scaler = preprocess_data(DATA_PATH)
    print(
        f"总样本数: {len(processed_data)}, 中风样本数: {processed_data['stroke'].sum():.0f}, 非中风样本数: {len(processed_data) - processed_data['stroke'].sum():.0f}")

    # 3. 划分训练集/测试集
    print("\n=== 划分训练集与测试集 ===")
    train_data, test_data = train_test_split(
        processed_data,
        test_size=0.2,
        random_state=42,
        stratify=processed_data['stroke']  # 分层抽样，保证类别比例
    )
    print(f"训练集样本数: {len(train_data)}, 测试集样本数: {len(test_data)}")

    # 4. 二次检查训练集类型（最终确认）
    print("\n训练集特征类型检查（最终确认）：")
    train_features = train_data.drop('stroke', axis=1)
    print(f"训练集特征列类型: {train_features.dtypes.unique()}")  # 必须只显示float64
    print(f"训练集特征维度: {train_features.shape[1]}")

    # 5. 创建数据加载器（此时可安全转换张量）
    print("\n=== 创建数据加载器 ===")
    train_dataset = StrokeDataset(train_data)
    test_dataset = StrokeDataset(test_data)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True  # 加速GPU数据传输（可选）
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    print(f"训练集批次数量: {len(train_loader)}, 测试集批次数量: {len(test_loader)}")

    # 6. 初始化模型
    print("\n=== 初始化模型 ===")
    model = StrokeNet(input_dim).to(DEVICE)
    print(model)  # 打印模型结构

    # 7. 模型训练
    print("\n=== 开始模型训练 ===")
    model, train_history, test_history, all_preds, all_labels = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=DEVICE,
        epochs=EPOCHS,
        lr=LEARNING_RATE
    )

    # 8. 可视化训练历史
    print("\n=== 可视化训练历史 ===")
    plot_train_history(train_history, test_history)

    # 9. 模型评估
    print("\n=== 模型评估报告 ===")
    print("混淆矩阵:")
    print(confusion_matrix(all_labels, all_preds))
    print("\n分类报告（精确率/召回率/F1）:")
    print(classification_report(
        all_labels,
        all_preds,
        target_names=['无中风', '有中风'],
        digits=4
    ))

    # 10. 保存模型
    print("\n=== 保存最佳模型 ===")
    save_model(model, scaler=scaler)

    # 11. 验证模型加载功能
    print("\n=== 验证模型加载 ===")
    loaded_model, loaded_scaler = load_model('best_stroke_model.pt', input_dim, DEVICE)
    loaded_model.eval()
    with torch.no_grad():
        # 加载后测试准确率
        test_features_tensor = torch.tensor(
            test_data.drop('stroke', axis=1).values,
            dtype=torch.float32
        ).to(DEVICE)
        test_labels_tensor = torch.tensor(
            test_data['stroke'].values,
            dtype=torch.float32
        ).unsqueeze(1).to(DEVICE)
        loaded_outputs = loaded_model(test_features_tensor)
        loaded_acc = ((loaded_outputs > 0.5).float() == test_labels_tensor).sum().item() / len(test_labels_tensor)
    print(f"加载后模型测试准确率: {loaded_acc:.4f}")


if __name__ == "__main__":
    main()