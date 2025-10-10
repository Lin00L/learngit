import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler  # 数据标准化（避免量纲影响）

# -------------------------- 1. 数据准备与预处理 --------------------------
# 读取train.csv数据集（请确保文件路径正确）
# 数据集格式要求：包含"x"列（特征）和"y"列（标签）
df = pd.read_csv("train.csv")
# 处理可能的缺失值（删除或填充）
df = df.dropna(subset=["x", "y"])
# 提取特征和标签并转换为Tensor
x_raw = df["x"].values.reshape(-1, 1)  # 形状：(n_samples, 1)
y_raw = df["y"].values.reshape(-1, 1)

# 数据标准化（提升模型训练稳定性）
scaler_x = StandardScaler()
scaler_y = StandardScaler()
x_scaled = scaler_x.fit_transform(x_raw)
y_scaled = scaler_y.fit_transform(y_raw)

# 转换为PyTorch Tensor
x_data = torch.tensor(x_scaled, dtype=torch.float32)
y_data = torch.tensor(y_scaled, dtype=torch.float32)


# -------------------------- 2. 自定义线性回归模型（正态分布初始化） --------------------------
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        # 定义线性层（in_features=1个特征，out_features=1个输出）
        self.linear = nn.Linear(1, 1)
        # 正态分布初始化权重和偏置（mean=0, std=0.1）
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.linear.bias, mean=0.0, std=0.1)

    def forward(self, x):
        # 前向传播：计算预测值
        y_pred = self.linear(x)
        return y_pred


# -------------------------- 3. 模型训练通用函数（支持不同优化器） --------------------------
def train_model(optimizer_name, lr=0.01, epochs=1000):
    """
    训练模型并记录训练过程
    参数：
        optimizer_name: 优化器名称（"SGD"、"Adam"、"Adagrad"）
        lr: 学习率
        epochs: 训练轮次
    返回：
        train_history: 训练历史（包含每轮的loss、w、b）
    """
    # 初始化模型、损失函数
    model = LinearRegressionModel()
    criterion = nn.MSELoss()  # 均方误差损失
    # 根据名称选择优化器
    if optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "Adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=lr)
    else:
        raise ValueError("不支持的优化器，请选择'SGD'、'Adam'或'Adagrad'")

    # 记录训练历史
    train_history = {
        "loss": [],
        "w": [],
        "b": []
    }

    # 开始训练
    for epoch in range(epochs):
        # 1. 前向传播：计算预测值和损失
        y_pred = model(x_data)
        loss = criterion(y_pred, y_data)

        # 2. 反向传播：梯度清零 + 计算梯度 + 更新参数
        optimizer.zero_grad()  # 梯度清零（避免累加）
        loss.backward()  # 自动求导
        optimizer.step()  # 更新参数

        # 3. 记录当前轮次的信息
        train_history["loss"].append(loss.item())
        train_history["w"].append(model.linear.weight.item())
        train_history["b"].append(model.linear.bias.item())

        # 每100轮打印一次训练信息
        if (epoch + 1) % 100 == 0:
            print(f"优化器: {optimizer_name}, 轮次: {epoch + 1:4d}, 损失: {loss.item():.6f}, "
                  f"权重w: {model.linear.weight.item():.6f}, 偏置b: {model.linear.bias.item():.6f}")

    return model, train_history


# -------------------------- 4. 多优化器训练与性能对比 --------------------------
# 超参数设置（可根据需求调整）
LEARNING_RATE = 0.01
EPOCHS = 1000
OPTIMIZERS = ["SGD", "Adam", "Adagrad"]  # 选择3种优化器

# 存储所有优化器的训练结果
all_histories = {}
best_model = None
best_loss = float("inf")  # 初始化为无穷大

# 训练不同优化器的模型
for opt_name in OPTIMIZERS:
    print(f"\n========== 开始使用 {opt_name} 优化器训练 ==========")
    model, history = train_model(opt_name, lr=LEARNING_RATE, epochs=EPOCHS)
    all_histories[opt_name] = history

    # 保存损失最小的最优模型
    final_loss = history["loss"][-1]
    if final_loss < best_loss:
        best_loss = final_loss
        best_model = model

# -------------------------- 5. 可视化功能实现 --------------------------
plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文显示
plt.rcParams['axes.unicode_minus'] = False  # 支持负号显示

# 5.1 不同优化器的损失曲线对比
plt.figure(figsize=(12, 8))
for opt_name, history in all_histories.items():
    plt.plot(range(1, EPOCHS + 1), history["loss"], label=f"{opt_name} (最终损失: {history['loss'][-1]:.6f})")
plt.xlabel("训练轮次 (Epoch)", fontsize=12)
plt.ylabel("均方误差损失 (MSE Loss)", fontsize=12)
plt.title("不同优化器的损失曲线对比", fontsize=14, fontweight="bold")
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.savefig("optimizer_loss_comparison.png", dpi=300, bbox_inches="tight")
plt.close()

# 5.2 最优模型的权重（w）和偏置（b）更新过程
best_opt = min(all_histories.keys(), key=lambda k: all_histories[k]["loss"][-1])
best_history = all_histories[best_opt]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
# 权重w的更新曲线
ax1.plot(range(1, EPOCHS + 1), best_history["w"], color="#FF6B6B", linewidth=2)
ax1.set_xlabel("训练轮次 (Epoch)", fontsize=12)
ax1.set_ylabel("权重 (w)", fontsize=12)
ax1.set_title(f"{best_opt}优化器 - 权重w的更新过程", fontsize=14, fontweight="bold")
ax1.grid(alpha=0.3)
# 偏置b的更新曲线
ax2.plot(range(1, EPOCHS + 1), best_history["b"], color="#4ECDC4", linewidth=2)
ax2.set_xlabel("训练轮次 (Epoch)", fontsize=12)
ax2.set_ylabel("偏置 (b)", fontsize=12)
ax2.set_title(f"{best_opt}优化器 - 偏置b的更新过程", fontsize=14, fontweight="bold")
ax2.grid(alpha=0.3)
plt.savefig("w_b_update_process.png", dpi=300, bbox_inches="tight")
plt.close()

# 5.3 不同学习率对模型性能的影响（以最优优化器为例）
LEARNING_RATES = [0.001, 0.01, 0.1, 0.5]  # 测试4种学习率
lr_histories = {}

plt.figure(figsize=(12, 8))
for lr in LEARNING_RATES:
    print(f"\n========== 测试学习率 {lr} ==========")
    _, history = train_model(best_opt, lr=lr, epochs=EPOCHS)
    lr_histories[lr] = history
    plt.plot(range(1, EPOCHS + 1), history["loss"], label=f"学习率 {lr} (最终损失: {history['loss'][-1]:.6f})")

plt.xlabel("训练轮次 (Epoch)", fontsize=12)
plt.ylabel("均方误差损失 (MSE Loss)", fontsize=12)
plt.title(f"{best_opt}优化器 - 不同学习率的损失曲线对比", fontsize=14, fontweight="bold")
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.savefig("learning_rate_comparison.png", dpi=300, bbox_inches="tight")
plt.close()

# 5.4 不同训练轮次对模型性能的影响（以最优优化器和学习率为例）
BEST_LR = min(lr_histories.keys(), key=lambda k: lr_histories[k]["loss"][-1])
epoch_histories = {}
EPOCH_LIST = [100, 300, 500, 1000, 2000]  # 测试5种轮次

plt.figure(figsize=(12, 8))
for epochs in EPOCH_LIST:
    print(f"\n========== 测试训练轮次 {epochs} ==========")
    _, history = train_model(best_opt, lr=BEST_LR, epochs=epochs)
    epoch_histories[epochs] = history
    # 补全到2000轮（便于对比）
    full_loss = history["loss"] + [history["loss"][-1]] * (2000 - epochs)
    plt.plot(range(1, 2001), full_loss, label=f"轮次 {epochs} (最终损失: {history['loss'][-1]:.6f})")

plt.xlabel("训练轮次 (Epoch)", fontsize=12)
plt.ylabel("均方误差损失 (MSE Loss)", fontsize=12)
plt.title(f"{best_opt}优化器 (学习率{best_LR}) - 不同训练轮次的损失曲线对比", fontsize=14, fontweight="bold")
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.savefig("epoch_comparison.png", dpi=300, bbox_inches="tight")
plt.close()

# -------------------------- 6. 最优模型评估与参数输出 --------------------------
print("\n========== 最优模型信息 ==========")
print(f"最优优化器: {best_opt}")
print(f"最优学习率: {BEST_LR}")
print(f"最优训练轮次: {min(epoch_histories.keys(), key=lambda k: epoch_histories[k]['loss'][-1])}")
print(f"最小损失值: {best_loss:.6f}")
# 输出原始尺度的参数（反标准化）
# 由于数据标准化，需将模型参数转换回原始尺度
w_scaled = best_model.linear.weight.item()
b_scaled = best_model.linear.bias.item()
# 反标准化公式：y = (y_scaled * std_y) + mean_y = w_scaled * (x_scaled) + b_scaled
# 代入x_scaled = (x - mean_x) / std_x，推导得：y = (w_scaled / std_x) * x + (b_scaled * std_y + mean_y - w_scaled * mean_x / std_x)
w_original = w_scaled * (scaler_y.scale_[0] / scaler_x.scale_[0])
b_original = b_scaled * scaler_y.scale_[0] + scaler_y.mean_[0] - w_scaled * (scaler_x.mean_[0] / scaler_x.scale_[0]) * \
             scaler_y.scale_[0]
print(f"原始尺度参数 - 权重w: {w_original:.6f}, 偏置b: {b_original:.6f}")

# 测试最优模型（预测示例）
x_test_scaled = torch.tensor([[scaler_x.transform([[df["x"].min()]]).item()],  # 最小x
                              [[scaler_x.transform([[df["x"].mean()]]).item()],  # 平均x
                               [[scaler_x.transform([[df["x"].max()]]).item()]]  # 最大x
                               y_test_scaled = best_model(x_test_scaled)
y_test_original = scaler_y.inverse_transform(y_test_scaled.detach().numpy())
print("\n========== 模型预测示例 ==========")
print(
    f"输入x（原始）: {scaler_x.inverse_transform([[x_test_scaled[0][0].item()]])[0][0]:.2f} → 预测y: {y_test_original[0][0]:.2f}")
print(
    f"输入x（原始）: {scaler_x.inverse_transform([[x_test_scaled[1][0].item()]])[0][0]:.2f} → 预测y: {y_test_original[1][0]:.2f}")
print(
    f"输入x（原始）: {scaler_x.inverse_transform([[x_test_scaled[2][0].item()]])[0][0]:.2f} → 预测y: {y_test_original[2][0]:.2f}")

# -------------------------- 7. 模型保存（便于提交Git） --------------------------
# 保存最优模型参数
torch.save(best_model.state_dict(), "best_linear_regression_model.pth")
# 保存数据标准化器参数（便于后续预测）
np.savez("scaler_params.npz",
         mean_x=scaler_x.mean_, std_x=scaler_x.scale_,
         mean_y=scaler_y.mean_, std_y=scaler_y.scale_)
print("\n========== 文件保存完成 ==========")
print("1. 最优模型参数: best_linear_regression_model.pth")
print("2. 数据标准化参数: scaler_params.npz")
print(
    "3. 可视化图表: optimizer_loss_comparison.png, w_b_update_process.png, learning_rate_comparison.png, epoch_comparison.png")

# -------------------------- 8. Git提交备注（参考） --------------------------
print("\n========== Git提交备注建议 ==========")
print(f"学号: XXXXXX（替换为你的学号）")
print(f"任务: PyTorch线性回归练习5-1&5-2")
print(
    f"最优配置: 优化器={best_opt}, 学习率={BEST_LR}, 训练轮次={min(epoch_histories.keys(), key=lambda k: epoch_histories[k]['loss'][-1])}")
print(f"模型性能: 最终MSE损失={best_loss:.6f}, 原始尺度参数w={w_original:.6f}, b={b_original:.6f}")
print(f"提交文件: 本代码文件、模型参数文件、标准化参数文件、4张可视化图表")