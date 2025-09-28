import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn, optim

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
torch.seed()
# 加载数据
data = pd.read_csv('train.csv')
# 清洗空白数据
data.dropna(inplace=True)
x_data = data['x'].values
y_data = data['y'].values

# 转换为Tensor
x_train = torch.tensor(x_data, dtype=torch.float32).view(-1, 1)
y_train = torch.tensor(y_data, dtype=torch.float32).view(-1, 1)


# 定义线性模型
class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(1, 1)  # 输入维度为1，输出维度为1

    def forward(self, x):
        return self.linear(x)


# 初始化模型、损失函数和优化器
model = LinearModel()
criterion = nn.MSELoss()  # 均方误差损失
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 记录训练过程中的w和loss变化
w_values = []
loss_values = []
epochs = 1200 # 训练轮数

# 训练模型
for epoch in range(epochs):
    # 前向传播
    y_pred = model(x_train)
    loss = criterion(y_pred, y_train)

    # 记录w和loss
    w = model.linear.weight.item()
    w_values.append(w)
    loss_values.append(loss.item())

    # 反向传播和优化
    optimizer.zero_grad()  # 梯度清零
    loss.backward()  # 反向传播计算梯度
    optimizer.step()  # 更新参数

    # 每100轮打印一次信息
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.6f}, W: {w:.6f}')

# 可视化w的变化
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(epochs), w_values, 'b-')
plt.title('训练期间的权重（w）')
plt.xlabel('Epoch')
plt.ylabel('w的值')
plt.grid(True)

# 可视化loss的变化
plt.subplot(1, 2, 2)
plt.plot(range(epochs), loss_values, 'r-')
plt.title('训练期间的损失')
plt.xlabel('Epoch')
plt.ylabel('损失值')
plt.grid(True)

plt.tight_layout()
plt.show()

# 可视化最终拟合结果
plt.figure(figsize=(8, 6))
plt.scatter(x_data, y_data, color='blue', label='原始数据')
plt.plot(x_data, model(x_train).detach().numpy(), 'r-', label='拟合线')
plt.title('线性回归结果')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
