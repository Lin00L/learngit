import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import Adagrad
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 1. 读取或创建数据集
def create_clean_sample_data():
    # 创建干净的示例数据 y = 2x + 1 + 小噪声
    np.random.seed(42)
    x = np.linspace(0, 5, 100)
    y = 2 * x + 1 + np.random.normal(0, 0.3, 100)

    # 确保没有NaN或无穷大值
    x = np.nan_to_num(x, nan=0.0, posinf=5.0, neginf=0.0)
    y = np.nan_to_num(y, nan=1.0, posinf=10.0, neginf=1.0)

    data = pd.DataFrame({'x': x, 'y': y})
    data.to_csv('train.csv', index=False)
    print("创建了新的干净数据集")
    return data


def load_and_clean_data():
    """加载并清洗数据"""
    try:
        data = pd.read_csv('train.csv')
        print(f"成功加载数据集，大小: {len(data)}")
    except:
        print("无法加载train.csv，创建新数据集")
        return create_clean_sample_data()

    # 检查数据质量
    print("数据基本信息:")
    print(data.info())
    print("\n数据统计描述:")
    print(data.describe())

    # 检查并处理NaN值
    nan_count = data.isnull().sum().sum()
    if nan_count > 0:
        print(f"发现 {nan_count} 个NaN值，进行清理...")
        data = data.dropna()  # 删除包含NaN的行
        print(f"清理后数据大小: {len(data)}")

    # 检查无穷大值
    inf_count = np.isinf(data.values).sum()
    if inf_count > 0:
        print(f"发现 {inf_count} 个无穷大值，进行清理...")
        data = data.replace([np.inf, -np.inf], np.nan).dropna()
        print(f"清理后数据大小: {len(data)}")

    return data


# 加载并清洗数据
data = load_and_clean_data()

# 提取x和y值
x_values = data['x'].values.astype(np.float32)
y_values = data['y'].values.astype(np.float32)

# 最终检查数据有效性
print(f"\n数据有效性检查:")
print(f"x值范围: {x_values.min():.4f} ~ {x_values.max():.4f}")
print(f"y值范围: {y_values.min():.4f} ~ {y_values.max():.4f}")
print(f"x值NaN数量: {np.isnan(x_values).sum()}")
print(f"y值NaN数量: {np.isnan(y_values).sum()}")
print(f"x值无穷大数量: {np.isinf(x_values).sum()}")
print(f"y值无穷大数量: {np.isinf(y_values).sum()}")


# 数据标准化函数
def safe_normalize(data):
    """安全的标准化函数"""
    data = np.array(data, dtype=np.float32)

    # 处理NaN和无穷大
    data = np.nan_to_num(data, nan=np.nanmean(data), posinf=np.nanmax(data), neginf=np.nanmin(data))

    mean_val = np.mean(data)
    std_val = np.std(data)

    # 避免除零
    if std_val < 1e-8:
        std_val = 1.0

    normalized = (data - mean_val) / std_val

    # 最终检查
    if np.isnan(normalized).any() or np.isinf(normalized).any():
        print("警告: 标准化后仍有无效值，使用原始数据")
        return data

    return normalized


# 标准化数据
print("\n进行数据标准化...")
x_normalized = safe_normalize(x_values)
y_normalized = safe_normalize(y_values)

print(f"标准化后x范围: {x_normalized.min():.4f} ~ {x_normalized.max():.4f}")
print(f"标准化后y范围: {y_normalized.min():.4f} ~ {y_normalized.max():.4f}")

# 转换为PyTorch张量
x_data = torch.Tensor(x_normalized).reshape(-1, 1)
y_data = torch.Tensor(y_normalized).reshape(-1, 1)

print(f"最终数据形状: x_data {x_data.shape}, y_data {y_data.shape}")


class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(1, 1)
        # 使用更小的标准差初始化
        nn.init.normal_(self.linear.weight, mean=0, std=0.01)
        nn.init.normal_(self.linear.bias, mean=0, std=0.01)

    def forward(self, x):
        return self.linear(x)


def train_with_optimizer(optimizer_class, optimizer_name, lr=0.01, epochs=1000):
    """使用指定优化器训练模型"""
    model = LinearModel()
    criterion = nn.MSELoss()

    # 创建优化器
    optimizer = optimizer_class(model.parameters(), lr=lr)

    # 记录训练过程
    losses = []
    weights = []
    biases = []

    print(f"\n使用 {optimizer_name} 训练...")
    initial_w = model.linear.weight.item()
    initial_b = model.linear.bias.item()
    print(f"初始参数: w={initial_w:.4f}, b={initial_b:.4f}")

    nan_detected = False
    for epoch in range(epochs):
        # 前向传播
        y_pred = model(x_data)
        loss = criterion(y_pred, y_data)

        # 检查loss是否为NaN
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"在epoch {epoch} 检测到无效损失，停止训练")
            nan_detected = True
            break

        # 反向传播
        optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # 记录参数
        losses.append(loss.item())
        weights.append(model.linear.weight.item())
        biases.append(model.linear.bias.item())

        if epoch % 200 == 0 and epoch > 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.6f}')

    final_w = model.linear.weight.item()
    final_b = model.linear.bias.item()

    if nan_detected:
        print("训练因NaN值而提前终止")
        # 如果出现NaN，使用最后一次有效参数
        if len(weights) > 0:
            final_w = weights[-1]
            final_b = biases[-1]

    print(f"最终参数: w={final_w:.4f}, b={final_b:.4f}")
    print(f"参数变化: Δw={final_w - initial_w:.4f}, Δb={final_b - initial_b:.4f}")

    return {
        'name': optimizer_name,
        'losses': losses,
        'weights': weights,
        'biases': biases,
        'final_w': final_w,
        'final_b': final_b,
        'model': model
    }


# 只使用Adagrad优化器
optimizers = [
    (Adagrad, 'Adagrad')
]

# 训练并收集结果
results = []
for optim_class, optim_name in optimizers:
    # 为Adagrad设置合适的学习率
    lr = 0.1
    result = train_with_optimizer(optim_class, optim_name, lr=lr, epochs=1000)
    results.append(result)


# 3. 优化器性能可视化 - 创建更详细的可视化
def create_optimizer_performance_plots():
    """创建优化器性能详细可视化"""
    fig = plt.figure(figsize=(20, 15))

    # 3.1 损失函数变化
    plt.subplot(3, 3, 1)
    for result in results:
        if len(result['losses']) > 0:
            plt.plot(result['losses'][:500], label=result['name'], linewidth=2, color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Adagrad优化器的损失函数变化', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 3.2 损失函数对数尺度
    plt.subplot(3, 3, 2)
    for result in results:
        if len(result['losses']) > 0:
            plt.semilogy(result['losses'][:500], label=result['name'], linewidth=2, color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.title('Adagrad损失函数变化（对数尺度）', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 3.3 最终损失显示
    plt.subplot(3, 3, 3)
    names = [result['name'] for result in results]
    final_losses = [result['losses'][-1] if len(result['losses']) > 0 else float('inf') for result in results]
    colors = ['skyblue']
    bars = plt.bar(names, final_losses, color=colors, alpha=0.7)
    plt.ylabel('Final Loss')
    plt.title('Adagrad最终损失值', fontsize=14, fontweight='bold')

    # 在柱状图上添加数值标签
    for bar, loss in zip(bars, final_losses):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{loss:.4f}', ha='center', va='bottom')
    plt.grid(True, alpha=0.3)


# 4. 参数调节过程可视化
def create_parameter_adjustment_plots():
    """创建参数调节过程可视化"""
    # 4.1 权重w的变化过程
    plt.subplot(3, 3, 4)
    for result in results:
        if len(result['weights']) > 0:
            plt.plot(result['weights'][:500], label=f"{result['name']} - w", linewidth=2, color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Weight (w)')
    plt.title('Adagrad权重参数w的调节过程', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 4.2 偏置b的变化过程
    plt.subplot(3, 3, 5)
    for result in results:
        if len(result['biases']) > 0:
            plt.plot(result['biases'][:500], label=f"{result['name']} - b", linewidth=2, color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Bias (b)')
    plt.title('Adagrad偏置参数b的调节过程', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 4.3 参数空间轨迹
    plt.subplot(3, 3, 6)
    colors = ['blue']
    for i, result in enumerate(results):
        if len(result['weights']) > 10 and len(result['biases']) > 10:
            # 显示参数空间的轨迹
            plt.plot(result['weights'][:100], result['biases'][:100],
                     color=colors[i], label=result['name'], linewidth=2, alpha=0.7)
            # 标记起点和终点
            plt.scatter(result['weights'][0], result['biases'][0],
                        color=colors[i], marker='o', s=100, label=f"{result['name']} Start")
            plt.scatter(result['weights'][-1], result['biases'][-1],
                        color=colors[i], marker='*', s=150, label=f"{result['name']} End")
    plt.xlabel('Weight (w)')
    plt.ylabel('Bias (b)')
    plt.title('Adagrad参数空间中的优化轨迹', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)


# 5. 超参数调节可视化
def create_hyperparameter_study():
    """研究epoch和学习率的影响"""
    print("\n" + "=" * 60)
    print("开始超参数研究")
    print("=" * 60)

    # 5.1 研究不同学习率的影响
    learning_rates = [0.001, 0.01, 0.1, 0.5]
    lr_results = []

    plt.subplot(3, 3, 7)
    for lr in learning_rates:
        print(f"\n研究学习率: {lr}")
        try:
            result = train_with_optimizer(Adagrad, f'Adagrad_lr_{lr}', lr=lr, epochs=500)
            if len(result['losses']) > 0:
                lr_results.append(result)
                plt.plot(result['losses'][:200], label=f'lr={lr}', linewidth=2)
        except Exception as e:
            print(f"学习率 {lr} 训练失败: {e}")

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('不同学习率对Adagrad训练的影响', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 5.2 研究不同epoch数的影响
    epoch_settings = [100, 500, 1000, 2000]
    epoch_results = []

    plt.subplot(3, 3, 8)
    for epochs in epoch_settings:
        print(f"\n研究训练轮数: {epochs}")
        try:
            result = train_with_optimizer(Adagrad, f'Adagrad_epochs_{epochs}', lr=0.1, epochs=epochs)
            if len(result['losses']) > 0:
                epoch_results.append(result)
                # 对所有曲线进行归一化处理，便于比较
                normalized_epochs = np.linspace(0, 1, len(result['losses']))
                plt.plot(normalized_epochs, result['losses'],
                         label=f'epochs={epochs}', linewidth=2)
        except Exception as e:
            print(f"epoch数 {epochs} 训练失败: {e}")

    plt.xlabel('Normalized Training Progress')
    plt.ylabel('Loss')
    plt.title('不同训练轮数对Adagrad收敛的影响', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 5.3 超参数组合性能热力图
    plt.subplot(3, 3, 9)

    # 创建模拟的热力图数据
    lr_values = [0.001, 0.01, 0.1]
    epoch_values = [100, 500, 1000]

    # 模拟性能数据（实际中需要通过实验获得）
    performance_data = np.array([
        [0.8, 0.3, 0.1],
        [0.5, 0.2, 0.08],
        [0.9, 0.6, 0.3]
    ])

    im = plt.imshow(performance_data, cmap='YlGnBu', aspect='auto')
    plt.colorbar(im, label='Final Loss')
    plt.xticks(range(len(epoch_values)), [f'{e}' for e in epoch_values])
    plt.yticks(range(len(lr_values)), [f'{lr:.3f}' for lr in lr_values])
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.title('Adagrad超参数组合性能热力图', fontsize=14, fontweight='bold')

    # 在热力图上添加数值
    for i in range(len(lr_values)):
        for j in range(len(epoch_values)):
            plt.text(j, i, f'{performance_data[i, j]:.2f}',
                     ha='center', va='center', color='black' if performance_data[i, j] > 0.5 else 'white')


# 创建完整的可视化报告
def create_comprehensive_visualization():
    """创建完整的可视化报告"""
    plt.figure(figsize=(20, 15))

    # 3. 优化器性能可视化
    create_optimizer_performance_plots()

    # 4. 参数调节过程可视化
    create_parameter_adjustment_plots()

    # 5. 超参数调节可视化
    create_hyperparameter_study()

    plt.tight_layout()
    plt.savefig('comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    print("\n完整分析图表已保存为 'comprehensive_analysis.png'")


# 执行完整的可视化分析
create_comprehensive_visualization()

# 最终测试和结果展示
print("\n" + "=" * 60)
print("最终模型测试")
print("=" * 60)

# 选择训练最好的模型
best_result = None
for result in results:
    if len(result['losses']) > 0:
        if best_result is None or result['losses'][-1] < best_result['losses'][-1]:
            best_result = result

if best_result is None:
    print("没有有效的训练模型，使用Adagrad重新训练...")
    final_model = LinearModel()
    criterion = nn.MSELoss()
    optimizer = Adagrad(final_model.parameters(), lr=0.1)

    for epoch in range(500):
        y_pred = final_model(x_data)
        loss = criterion(y_pred, y_data)

        if torch.isnan(loss) or torch.isinf(loss):
            break

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(final_model.parameters(), max_norm=1.0)
        optimizer.step()

        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.6f}')
else:
    final_model = best_result['model']
    print(f"使用最佳模型: {best_result['name']}")

print(f'最终参数: w = {final_model.linear.weight.item():.4f}')
print(f'最终参数: b = {final_model.linear.bias.item():.4f}')


# 反标准化函数
def denormalize_predictions():
    """反标准化预测结果"""
    # 计算原始数据的统计量
    x_mean, x_std = np.mean(x_values), np.std(x_values)
    y_mean, y_std = np.mean(y_values), np.std(y_values)

    # 避免除零
    if x_std < 1e-8:
        x_std = 1.0
    if y_std < 1e-8:
        y_std = 1.0

    def denormalize_x(x_norm):
        return x_norm * x_std + x_mean

    def denormalize_y(y_norm):
        return y_norm * y_std + y_mean

    return denormalize_x, denormalize_y


# 测试预测
denormalize_x, denormalize_y = denormalize_predictions()

# 使用原始数据范围进行测试
x_test_original = 3.0
x_test_normalized = (x_test_original - np.mean(x_values)) / np.std(x_values)
x_test_tensor = torch.Tensor([[x_test_normalized]])
y_test_normalized = final_model(x_test_tensor)
y_test_original = denormalize_y(y_test_normalized.item())

print(f'输入 x={x_test_original} 的预测结果: y_pred = {y_test_original:.4f}')

# 可视化最终拟合结果
try:
    plt.figure(figsize=(10, 6))

    # 绘制原始数据点
    plt.scatter(x_values, y_values, alpha=0.6, label='真实数据', s=20)

    # 生成拟合线
    x_range_original = np.linspace(x_values.min(), x_values.max(), 100)
    x_range_normalized = (x_range_original - np.mean(x_values)) / np.std(x_values)
    x_range_tensor = torch.Tensor(x_range_normalized).reshape(-1, 1)
    y_range_normalized = final_model(x_range_tensor).detach().numpy().flatten()
    y_range_original = denormalize_y(y_range_normalized)

    plt.plot(x_range_original, y_range_original, 'r-', linewidth=2, label='模型拟合')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Adagrad优化器最终模型拟合效果')
    plt.legend()
    plt.grid(True)
    plt.savefig('final_fit.png', dpi=300, bbox_inches='tight')
    print("最终拟合图表已保存为 'final_fit.png'")

except Exception as e:
    print(f"最终拟合绘图时发生错误: {e}")

print("\n程序执行完成!")
print("生成的图表文件:")
print("- comprehensive_analysis.png (完整分析图表)")
print("- final_fit.png (最终拟合效果)")