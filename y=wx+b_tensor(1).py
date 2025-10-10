import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn

# 1. 读取并清洗数据
df = pd.read_csv('train.csv')
# 删除包含缺失值的行（处理y列空值）
df.dropna(inplace=True)
# 提取特征x和标签y，转为PyTorch张量
x_data = torch.tensor(df['x'].values, dtype=torch.float32).view(-1, 1)
y_data = torch.tensor(df['y'].values, dtype=torch.float32).view(-1, 1)
print(f"清洗后数据量：{len(x_data)} 条")

# 2. 定义模型参数（使用PyTorch张量，需要梯度）
w = torch.tensor([1.0], dtype=torch.float32, requires_grad=True)
b = torch.tensor([0.0], dtype=torch.float32, requires_grad=True)

# 3. 定义前向传播和损失函数
def forward(x):
    """前向传播：计算线性预测值 y_pred = w*x + b"""
    return x * w + b

def compute_mse(x, y):
    """计算均方误差（MSE）作为损失指标"""
    y_pred = forward(x)
    return torch.mean((y_pred - y) ** 2)

# 4. 训练模型（梯度下降）
learning_rate = 0.0001  # 学习率需要调小，因为数据量较大
epochs = 100  # 训练轮数

# 记录训练过程中的参数和损失变化
w_history = []
b_history = []
loss_history = []

print("开始训练...")
print(f"predict (before training): x=4, y_pred={forward(torch.tensor([4.0])).item():.4f}")

for epoch in range(epochs):
    # 清零梯度
    if w.grad is not None:
        w.grad.zero_()
    if b.grad is not None:
        b.grad.zero_()

    # 前向传播计算整个数据集的损失
    loss = compute_mse(x_data, y_data)

    # 记录当前参数和损失
    w_history.append(w.item())
    b_history.append(b.item())
    loss_history.append(loss.item())

    # 反向传播计算梯度
    loss.backward()

    # 更新参数（梯度下降）
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad

    # 打印训练进度
    if epoch % 10 == 0 or epoch < 5:
        print(f"progress: epoch={epoch}, loss={loss.item():.6f}, "
              f"w={w.item():.4f}, b={b.item():.4f}")

print(f"predict (after training): x=4, y_pred={forward(torch.tensor([4.0])).item():.4f}")

# 5. 输出最终参数
print(f"\n训练完成！")
print(f"最终参数：w = {w.item():.4f}, b = {b.item():.4f}")
print(f"最终损失：{loss.item():.4f}")

# 6. 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 7. 绘制线性回归图
plt.figure(figsize=(10, 8))

# 绘制原始数据点
plt.scatter(x_data.detach().numpy(), y_data.detach().numpy(),
           alpha=0.6, color='#2E86AB', label='原始数据', s=50)

# 绘制拟合直线
x_plot = np.linspace(x_data.min(), x_data.max(), 100)
x_plot_tensor = torch.tensor(x_plot, dtype=torch.float32).view(-1, 1)
y_plot = forward(x_plot_tensor).detach().numpy()

plt.plot(x_plot, y_plot, color='#A23B72', linewidth=3,
        label=f'拟合直线: y = {w.item():.4f}x + {b.item():.4f}')

plt.xlabel('x', fontsize=14)
plt.ylabel('y', fontsize=14)
plt.title('线性回归拟合结果', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3, linestyle='--')

# 根据您提供的图像，调整坐标轴范围
plt.xlim(0, 100)
plt.ylim(0, 80)

plt.tight_layout()
plt.savefig('linear_regression_result.png', dpi=300, bbox_inches='tight')
print("线性回归拟合结果图已保存为：linear_regression_result.png")

# 关闭当前图形
plt.close()

# 8. 绘制w与loss的关系图（简化版）
plt.figure(figsize=(10, 8))

# 直接绘制连线，不使用复杂的颜色映射
plt.plot(w_history, loss_history, color='blue', linewidth=2, alpha=0.7)

# 标记起点和终点
plt.scatter(w_history[0], loss_history[0], color='red', s=100,
           label=f'起点: w={w_history[0]:.4f}', marker='o')
plt.scatter(w_history[-1], loss_history[-1], color='green', s=100,
           label=f'终点: w={w_history[-1]:.4f}', marker='s')

plt.xlabel('参数 w 的值', fontsize=14)
plt.ylabel('损失函数 Loss (MSE)', fontsize=14)
plt.title('参数 w 与损失函数的关系', fontsize=16)
plt.grid(True, alpha=0.3, linestyle='--')
plt.legend(fontsize=12)

plt.tight_layout()
plt.savefig('w_loss_relationship.png', dpi=300, bbox_inches='tight')
print("w-loss关系图已保存为：w_loss_relationship.png")

# 关闭图形
plt.close()

print("\n所有图表已保存完成！")