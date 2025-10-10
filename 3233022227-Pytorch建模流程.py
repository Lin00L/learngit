import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.font_manager as fm

# 设置中文字体（更完善的字体配置）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['mathtext.fontset'] = 'cm'  # 使用计算机现代字体显示数学符号


# 加载数据
df = pd.read_csv('train.csv')
# 清洗空白数据
df.dropna(inplace=True)
# 数据归一化处理
scaler_x = StandardScaler()
scaler_y = StandardScaler()
x_scaled = scaler_x.fit_transform(df['x'].values.reshape(-1, 1))
y_scaled = scaler_y.fit_transform(df['y'].values.reshape(-1, 1))
x_data = torch.tensor(x_scaled, dtype=torch.float32)
y_data = torch.tensor(y_scaled, dtype=torch.float32)


# 设计模型
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)
        # 保守的权重初始化
        nn.init.normal_(self.linear.weight, mean=0, std=0.1)
        nn.init.normal_(self.linear.bias, mean=0, std=0.1)

    def forward(self, x):
        return self.linear(x)


def train_with_optimizer(optimizer_name, lr=0.001, epochs=1000):
    """使用不同优化器训练模型"""
    model = LinearRegressionModel()
    # 选择优化器
    if optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr)

    criterion = nn.MSELoss()

    # 记录训练过程
    losses = []
    weights = []
    biases = []

    for epoch in range(epochs):
        # 前向传播
        y_pred = model(x_data)
        loss = criterion(y_pred, y_data)

        # 检查是否出现NaN，及时停止训练
        if torch.isnan(loss):
            print(f"{optimizer_name}在第{epoch}轮出现NaN，停止训练")
            break

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 记录参数
        losses.append(loss.item())
        weights.append(model.linear.weight.item())
        biases.append(model.linear.bias.item())

        if epoch % 200 == 0:
            print(f'{optimizer_name} - Epoch {epoch}, Loss: {loss.item():.4f}')

    return losses, weights, biases, model


# 使用三种不同的优化器并可视化性能
optimizers = ['SGD', 'Adam', 'RMSprop']
results = {}

plt.figure(figsize=(15, 10))

# 训练并绘制损失曲线
for i, opt in enumerate(optimizers):
    # 为SGD使用更小的学习率
    lr = 0.001 if opt == 'SGD' else 0.01
    losses, weights, biases, model = train_with_optimizer(opt, lr=lr, epochs=1000)
    results[opt] = {
        'losses': losses,
        'weights': weights,
        'biases': biases,
        'model': model
    }

    # 绘制损失曲线
    plt.subplot(2, 3, i + 1)
    plt.plot(losses)
    plt.title(f'{opt}优化器 - 损失曲线')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)

# 调节参数w和b的可视化
for i, opt in enumerate(optimizers):
    plt.subplot(2, 3, i + 4)
    plt.plot(results[opt]['weights'], label='权重 w')
    plt.plot(results[opt]['biases'], label='偏置 b')
    plt.title(f'{opt}优化器 - 参数变化')
    plt.xlabel('Epoch')
    plt.ylabel('参数值')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()

# 调节epoch和学习率η的可视化
learning_rates = [0.0001, 0.001, 0.01]
epochs_list = [500, 1000, 2000]

plt.figure(figsize=(15, 10))

# 不同学习率的影响
for i, lr in enumerate(learning_rates):
    losses, _, _, _ = train_with_optimizer('SGD', lr=lr, epochs=1000)
    plt.subplot(2, 3, i + 1)
    plt.plot(losses)
    plt.title(f'学习率 η = {lr} - 损失曲线')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.grid(True)

# 不同epoch数量的影响
fixed_lr = 0.001
for i, epochs in enumerate(epochs_list):
    losses, weights, biases, _ = train_with_optimizer('SGD', lr=fixed_lr, epochs=epochs)
    plt.subplot(2, 3, i + 4)
    plt.plot(weights, label='权重 w')
    plt.plot(biases, label='偏置 b')
    plt.title(f'Epochs = {epochs} - 参数收敛')
    plt.xlabel('Epoch')
    plt.ylabel('参数值')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()

# 最终模型评估
print("\n最终模型参数:")
for opt in optimizers:
    model = results[opt]['model']
    w = model.linear.weight.item()
    b = model.linear.bias.item()
    final_loss = results[opt]['losses'][-1] if len(results[opt]['losses']) > 0 else float('nan')
    print(f"{opt}优化器: w = {w:.4f}, b = {b:.4f}, 最终损失 = {final_loss:.4f}")

# 显示真实数据与预测结果
plt.figure(figsize=(15, 5))
x_test = torch.linspace(x_data.min(), x_data.max(), 100).reshape(-1, 1)

for i, opt in enumerate(optimizers):
    plt.subplot(1, 3, i + 1)
    model = results[opt]['model']
    with torch.no_grad():
        y_pred = model(x_test)

    # 反归一化以显示原始数据范围
    x_original = scaler_x.inverse_transform(x_data.numpy())
    y_original = scaler_y.inverse_transform(y_data.numpy())
    x_test_original = scaler_x.inverse_transform(x_test.numpy())
    y_pred_original = scaler_y.inverse_transform(y_pred.numpy())

    plt.scatter(x_original, y_original, alpha=0.3, label='真实数据')
    plt.plot(x_test_original, y_pred_original, 'r-', linewidth=2, label='预测直线')
    plt.title(f'{opt}优化器拟合结果')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()
