import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 1. 读取并清洗数据
df = pd.read_csv('train.csv')
# 清洗空白数据
df.dropna(inplace=True)
x_data = df['x'].values  # 特征数据（转为numpy数组）
y_data = df['y'].values  # 标签数据（转为numpy数组）

# 2. 定义模型与损失函数
def forward(x, w, b):
    """前向传播：计算预测值 y_pred = w*x + b"""
    return x * w + b

def loss(x, y, w, b):
    """计算损失"""
    y_pred = forward(x, w, b)
    return (y_pred-y)*(y_pred-y)


# 3. 网格搜索训练：遍历w、b计算MSE
w_range = np.arange(0.5, 1.51, 0.05)  # w：0.5~1.5（步长0.05，共21个值）
b_range = np.arange(-3.0, 3.01, 0.05)  # b：-3~3（步长0.05，共121个值）

# 存储训练结果（w、b、对应MSE）
w_list = []
b_list = []
mse_list = []

# 遍历所有w和b的组合，计算MSE
for w in w_range:
    for b in b_range:
        l_sum=0
        for x_val, y_val in zip(x_data, y_data):
            y_pred_val = forward(x_val, w, b)
            loss_val = loss(x_val, y_val, w, b)
            l_sum += loss_val
            print('x_val=',x_val,'y_val=', y_val,'y_pred_val=', y_pred_val, 'loss_val=',loss_val,'\t')
        print('MSE=',l_sum/len(x_data))  # 计算当前参数的MSE
        w_list.append(w)
        b_list.append(b)
        mse_list.append(l_sum/len(x_data))

# 找到最优参数（MSE最小对应的w和b）
min_mse_idx = np.argmin(mse_list)
best_w = w_list[min_mse_idx]
best_b = b_list[min_mse_idx]
best_mse = mse_list[min_mse_idx]
print(f"\n最优参数：w={best_w:.4f}, b={best_b:.4f}")
print(f"最优参数对应的MSE：{best_mse:.4f}")


# 4. 可视化：w与loss、b与loss的关系
plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示问题
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# 4.1 可视化：w与MSE的关系（固定b为最优b）
# 提取“最优b”对应的所有(w, MSE)对
best_b_mask = np.isclose(b_list, best_b)  # 筛选b=best_b的记录
w_best_b = np.array(w_list)[best_b_mask]
mse_best_b = np.array(mse_list)[best_b_mask]

plt.figure(figsize=(12, 5))

# 子图1：w与MSE的关系
plt.subplot(1, 2, 1)
plt.plot(w_best_b, mse_best_b, color='#1f77b4', linewidth=2)
plt.scatter(best_w, best_mse, color='red', s=50, label=f'最优w：{best_w:.4f}')
plt.xlabel('参数 w（斜率）', fontsize=12)
plt.ylabel('MSE（均方误差）', fontsize=12)
plt.title('w与MSE的关系（固定b={:.4f}）'.format(best_b), fontsize=14)
plt.legend()
plt.grid(alpha=0.3)

# 4.2 可视化：b与MSE的关系（固定w为最优w）
# 提取“最优w”对应的所有(b, MSE)对
best_w_mask = np.isclose(w_list, best_w)  # 筛选w=best_w的记录
b_best_w = np.array(b_list)[best_w_mask]
mse_best_w = np.array(mse_list)[best_w_mask]

plt.subplot(1, 2, 2)
plt.plot(b_best_w, mse_best_w, color='#ff7f0e', linewidth=2)
plt.scatter(best_b, best_mse, color='red', s=50, label=f'最优b：{best_b:.4f}')
plt.xlabel('参数 b（截距）', fontsize=12)
plt.ylabel('MSE（均方误差）', fontsize=12)
plt.title('b与MSE的关系（固定w={:.4f}）'.format(best_w), fontsize=14)
plt.legend()
plt.grid(alpha=0.3)

# 保存图片（避免运行时弹窗阻塞，可直接查看文件）
plt.tight_layout()
plt.savefig('w_b_loss_relation.png', dpi=300, bbox_inches='tight')
print("\n可视化图已保存为：w_b_loss_relation.png")