import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 1. 读取并清洗数据
df = pd.read_csv('train.csv')
df.dropna(inplace=True)
x_data = df['x'].values
y_data = df['y'].values

# 2. 定义模型与损失函数
def forward(x, w, b):
    """前向传播：计算预测值 y_pred = w*x + b"""
    return x * w + b

def mse_loss(y_pred, y_true):
    """计算均方误差"""
    return np.mean((y_pred - y_true) ** 2)

# 3. 网格搜索训练（优化版）
w_range = np.arange(0.5, 1.51, 0.05)  # w：0.5~1.5（步长0.05）
b_range = np.arange(-3.0, 3.01, 0.05)  # b：-3~3（步长0.05）

# 初始化MSE网格
mse_grid = np.zeros((len(w_range), len(b_range)))

# 使用向量化计算加速
for i, w in enumerate(w_range):
    y_pred_base = w * x_data
    for j, b in enumerate(b_range):
        mse_grid[i, j] = mse_loss(y_pred_base + b, y_data)

# 找到最优参数
min_idx = np.unravel_index(np.argmin(mse_grid), mse_grid.shape)
best_w = w_range[min_idx[0]]
best_b = b_range[min_idx[1]]
best_mse = mse_grid[min_idx]

print("="*30)
print("训练结果")
print("="*30)
print(f"最优参数：w={best_w:.4f}, b={best_b:.4f}")
print(f"最优参数对应的MSE：{best_mse:.4f}")
print("="*30)

# 4. 可视化
plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示问题
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# 4.1 创建图形
plt.figure(figsize=(16, 10))

# 4.2 w与MSE的关系（固定b为最优b）
plt.subplot(2, 2, 1)
best_b_idx = min_idx[1]
plt.plot(w_range, mse_grid[:, best_b_idx], color='#1f77b4', linewidth=2)
plt.scatter(best_w, best_mse, color='red', s=80, zorder=5,
            label=f'最优w：{best_w:.4f}\nMSE：{best_mse:.4f}')
plt.xlabel('参数 w（斜率）', fontsize=12)
plt.ylabel('MSE（均方误差）', fontsize=12)
plt.title(f'w与MSE的关系（固定b={best_b:.4f}）', fontsize=14)
plt.legend()
plt.grid(alpha=0.3)

# 4.3 b与MSE的关系（固定w为最优w）
plt.subplot(2, 2, 2)
best_w_idx = min_idx[0]
plt.plot(b_range, mse_grid[best_w_idx, :], color='#ff7f0e', linewidth=2)
plt.scatter(best_b, best_mse, color='red', s=80, zorder=5,
            label=f'最优b：{best_b:.4f}\nMSE：{best_mse:.4f}')
plt.xlabel('参数 b（截距）', fontsize=12)
plt.ylabel('MSE（均方误差）', fontsize=12)
plt.title(f'b与MSE的关系（固定w={best_w:.4f}）', fontsize=14)
plt.legend()
plt.grid(alpha=0.3)

# 4.4 3D可视化：w、b与MSE的关系
ax = plt.subplot(2, 2, 3, projection='3d')
W, B = np.meshgrid(w_range, b_range, indexing='ij')
surf = ax.plot_surface(W, B, mse_grid, cmap='viridis', alpha=0.8)
ax.scatter(best_w, best_b, best_mse, color='red', s=100, marker='*',
           label=f'最优解\nw={best_w:.4f}\nb={best_b:.4f}\nMSE={best_mse:.4f}')
ax.set_xlabel('w（斜率）', fontsize=10)
ax.set_ylabel('b（截距）', fontsize=10)
ax.set_zlabel('MSE（均方误差）', fontsize=10)
ax.set_title('w、b与MSE的关系', fontsize=14)
ax.legend()
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

# 4.5 拟合结果可视化
plt.subplot(2, 2, 4)
y_pred = forward(x_data, best_w, best_b)
plt.scatter(x_data, y_data, color='#1f77b4', alpha=0.6, label='原始数据')
plt.plot(x_data, y_pred, color='red', linewidth=2, label=f'拟合直线：y = {best_w:.4f}x + {best_b:.4f}')
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title('线性回归拟合结果', fontsize=14)
plt.legend()
plt.grid(alpha=0.3)

# 保存图片
plt.tight_layout()
plt.savefig('linear_regression_results.png', dpi=300, bbox_inches='tight')
print("\n可视化图已保存为：linear_regression_results.png")