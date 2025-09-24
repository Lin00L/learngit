import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 1. 读取并清洗数据
df = pd.read_csv('train.csv')
# 删除包含缺失值的行（处理y列空值）
df.dropna(inplace=True)
# 提取特征x和标签y，转为numpy数组
x_data = df['x'].values
y_data = df['y'].values
print(f"清洗后数据量：{len(x_data)} 条")

# 2. 定义前向传播（预测）和损失计算函数
def forward(x, w, b):
    """前向传播：计算线性预测值 y_pred = w*x + b"""
    return x * w + b

def compute_mse(x, y, w, b):
    """计算均方误差（MSE）作为损失指标"""
    y_pred = forward(x, w, b)
    # 计算所有样本的损失并取均值
    return np.mean((y_pred - y) ** 2)

# 3. 网格搜索：遍历w和b的取值范围，计算对应MSE
# 定义w和b的搜索范围（根据数据分布合理设置）
w_range = np.arange(0.8, 1.21, 0.01)  # w从0.8到1.2，步长0.01（41个值）
b_range = np.arange(-1.5, 1.51, 0.01)  # b从-1.5到1.5，步长0.01（301个值）

# 存储参数与对应损失
w_list = []
b_list = []
mse_list = []

# 遍历所有w和b的组合计算MSE（双重循环实现网格搜索）
print("开始网格搜索计算损失...")
for w in w_range:
    for b in b_range:
        mse = compute_mse(x_data, y_data, w, b)
        w_list.append(w)
        b_list.append(b)
        mse_list.append(mse)

# 转换为numpy数组便于后续处理
w_array = np.array(w_list)
b_array = np.array(b_list)
mse_array = np.array(mse_list)

# 找到最优参数（MSE最小对应的w和b）
min_mse_idx = np.argmin(mse_array)
best_w = w_array[min_mse_idx]
best_b = b_array[min_mse_idx]
best_mse = mse_array[min_mse_idx]

print(f"\n最优参数：w = {best_w:.4f}, b = {best_b:.4f}")
print(f"最优参数对应的MSE：{best_mse:.4f}")

# 4. 数据筛选：提取绘制所需的参数-损失对
# 4.1 筛选最优b对应的所有(w, MSE)（用于绘制w与loss关系）
best_b_mask = np.isclose(b_array, best_b)  # 匹配最优b的掩码
w_best_b = w_array[best_b_mask]
mse_best_b = mse_array[best_b_mask]

# 4.2 筛选最优w对应的所有(b, MSE)（用于绘制b与loss关系）
best_w_mask = np.isclose(w_array, best_w)  # 匹配最优w的掩码
b_best_w = b_array[best_w_mask]
mse_best_w = mse_array[best_w_mask]

# 5. 绘制w与loss、b与loss的关系图
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 支持中文和英文
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 创建2个子图（1行2列）
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# 子图1：w与MSE的关系（固定b为最优值）
ax1.plot(w_best_b, mse_best_b, color='#2E86AB', linewidth=2.5, label='损失曲线')
# 标记最优w对应的点
ax1.scatter(best_w, best_mse, color='#A23B72', s=100, marker='*',
            label=f'最优w: {best_w:.4f}\nMSE: {best_mse:.4f}')
ax1.set_xlabel('参数 w（斜率）', fontsize=12)
ax1.set_ylabel('MSE（均方误差）', fontsize=12)
ax1.set_title(f'w与损失的关系（固定b={best_b:.4f}）', fontsize=14, pad=15)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3, linestyle='--')

# 子图2：b与MSE的关系（固定w为最优值）
ax2.plot(b_best_w, mse_best_w, color='#F18F01', linewidth=2.5, label='损失曲线')
# 标记最优b对应的点
ax2.scatter(best_b, best_mse, color='#C73E1D', s=100, marker='*',
            label=f'最优b: {best_b:.4f}\nMSE: {best_mse:.4f}')
ax2.set_xlabel('参数 b（截距）', fontsize=12)
ax2.set_ylabel('MSE（均方误差）', fontsize=12)
ax2.set_title(f'b与损失的关系（固定w={best_w:.4f}）', fontsize=14, pad=15)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, linestyle='--')

# 调整布局并保存图片
plt.tight_layout()
plt.savefig('w_b_loss_relation.png', dpi=300, bbox_inches='tight')
print("\n图表已保存为：w_b_loss_relation.png")