import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

class EcoFootprintDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)

        # 增强GDP清洗函数
        def clean_gdp(value):
            if isinstance(value, str):
                # 移除所有非数字字符（保留小数点）
                cleaned = ''.join(c for c in value if c.isdigit() or c == '.')
                return float(cleaned) if cleaned else 0.0
            return float(value) if not pd.isna(value) else 0.0

        df['GDP per Capita'] = df['GDP per Capita'].apply(clean_gdp)

        # 选择特征列和目标列
        feature_columns = ['Population (millions)', 'HDI', 'GDP per Capita',
                           'Cropland Footprint', 'Grazing Footprint',
                           'Forest Footprint', 'Carbon Footprint', 'Fish Footprint']

        # 修复Pandas警告：使用正确的方法填充缺失值
        for col in feature_columns:
            median_val = df[col].median()
            # 使用正确的方法避免链式赋值警告
            df.loc[:, col] = df[col].fillna(median_val)

        # 异常值处理（截断到第5-95百分位）
        for col in feature_columns:
            q5 = df[col].quantile(0.05)
            q95 = df[col].quantile(0.95)
            df.loc[:, col] = df[col].clip(lower=q5, upper=q95)

        # 稳健标准化（防止除零错误）
        self.features = df[feature_columns].values.astype(np.float32)
        self.target = df['Total Ecological Footprint'].values.astype(np.float32)

        self.feature_mean = np.nanmean(self.features, axis=0)
        self.feature_std = np.nanstd(self.features, axis=0)
        self.feature_std[self.feature_std == 0] = 1.0  # 防止除零错误

        self.target_mean = np.nanmean(self.target)
        self.target_std = np.nanstd(self.target)
        if self.target_std == 0:
            self.target_std = 1.0

        self.features = (self.features - self.feature_mean) / self.feature_std
        self.target = (self.target - self.target_mean) / self.target_std

        # 最终数据完整性检查
        assert not np.isnan(self.features).any(), "特征中存在NaN值!"
        assert not np.isnan(self.target).any(), "目标中存在NaN值!"

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.features[idx], dtype=torch.float32),
            torch.tensor(self.target[idx], dtype=torch.float32)
        )


class EcoFootprintModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 7),
            nn.BatchNorm1d(7),
            nn.ReLU(),

            nn.Linear(7, 6),
            nn.BatchNorm1d(6),
            nn.ReLU(),

            nn.Linear(6, 5),
            nn.BatchNorm1d(5),
            nn.ReLU(),

            nn.Linear(5, 1)
        )

        # 自定义权重初始化
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0.1)

    def forward(self, x):
        return self.layers(x)


def main():
    # 数据加载
    dataset = EcoFootprintDataset('D:/xunlei/countries.csv')

    # 打印数据统计信息
    print("特征统计:")
    print(f"样本数量: {len(dataset)}")
    print(f"特征均值: {dataset.feature_mean}")
    print(f"特征标准差: {dataset.feature_std}")
    print(f"目标均值: {dataset.target_mean}")
    print(f"目标标准差: {dataset.target_std}")

    train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)

    # 模型初始化
    model = EcoFootprintModel(input_size=8)

    # 自定义安全损失函数
    class SafeMSE(nn.Module):
        def __init__(self):
            super().__init__()
            self.eps = 1e-6

        def forward(self, pred, target):
            loss = (pred - target) ** 2
            loss[torch.isnan(loss)] = self.eps
            return loss.mean()

    criterion = SafeMSE()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5
    )

    # 梯度裁剪
    max_grad_norm = 1.0

    # 训练循环
    train_losses = []
    best_loss = float('inf')

    for epoch in range(400):
        model.train()
        epoch_loss = 0.0

        for i, (features, target) in enumerate(train_loader):
            optimizer.zero_grad()

            outputs = model(features)
            loss = criterion(outputs, target.unsqueeze(1))

            # 检查NaN
            if torch.isnan(loss):
                print(f"检测到NaN损失! 批次 {i}")
                continue

            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()
            epoch_loss += loss.item()

            # 每5个批次打印一次
            if i % 5 == 0:
                print(f'Epoch [{epoch + 1}/400], Batch [{i}/{len(train_loader)}], Loss: {loss.item():.4f}')

        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        scheduler.step(avg_loss)

        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 'best_model.pt')
            print(f'保存最佳模型，损失: {best_loss:.4f}')

        #print(f'Epoch [{epoch + 1}/200], 平均损失: {avg_loss:.4f}, 学习率: {optimizer.param_groups[0]["lr"]:.6f}')
        print(f'Epoch [{epoch + 1}/400], 平均损失: {avg_loss:.4f}, 学习率: {optimizer.param_groups[0]["lr"]:.8e}')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # 解决负号显示问题（可选）
    plt.rcParams['axes.unicode_minus'] = False
    # 可视化
    plt.figure(figsize=(12, 8))
    plt.plot(train_losses, label='训练损失')
    plt.title('训练损失曲线', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('MSE损失', fontsize=12)
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.savefig('training_loss.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("训练完成! 最佳模型已保存为 'best_model.pt'")

if __name__ == '__main__':
    main()