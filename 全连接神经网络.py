# 导入必要的库
import numpy as np                  # 用于数值计算和数据处理
import torch                         # PyTorch深度学习框架核心库
import torch.nn as nn                # 构建神经网络模块
from torch.utils.data import Dataset, DataLoader, random_split  # 数据加载和处理工具
import matplotlib.pyplot as plt      # 用于数据可视化
import os                            # 用于操作系统相关操作
# 设置环境变量以避免KMP库重复加载的问题（Windows系统常见）
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 定义自定义数据集类，继承自PyTorch的Dataset基类
class DiabetesDataset(Dataset):
    def __init__(self, filepath):
        """
        初始化数据集
        参数:
            filepath: 数据文件的路径
        """
        # 加载CSV数据文件，使用逗号分隔，数据类型为float32
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        # 计算数据集的样本数量（行数）
        self.len = xy.shape[0]
        # 提取特征数据（所有行，除了最后一列），并转换为PyTorch张量
        self.x_data = torch.from_numpy(xy[:, :-1])
        # 提取标签数据（所有行，仅最后一列），并转换为PyTorch张量，保持二维结构
        self.y_data = torch.from_numpy(xy[:, [-1]])

    def __getitem__(self, index):
        """
        通过索引获取样本
        参数:
            index: 样本索引
        返回:
            对应索引的特征和标签
        """
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        """
        返回数据集的样本数量
        返回:
            数据集长度
        """
        return self.len


# 定义神经网络模型，继承自PyTorch的Module基类
class DiabetesModel(nn.Module):
    def __init__(self):
        """初始化神经网络层"""
        # 调用父类的构造函数
        super(DiabetesModel, self).__init__()
        # 定义第一个线性层：输入特征数8，输出特征数6
        self.linear1 = nn.Linear(8, 6)
        # 定义第二个线性层：输入特征数6，输出特征数4
        self.linear2 = nn.Linear(6, 4)
        # 定义第三个线性层：输入特征数4，输出特征数1（二分类问题）
        self.linear3 = nn.Linear(4, 1)
        # 定义sigmoid激活函数，用于将输出转换为概率
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        定义正向传播过程
        参数:
            x: 输入特征张量
        返回:
            模型输出（预测概率）
        """
        # 第一层线性变换后经过sigmoid激活
        x = self.sigmoid(self.linear1(x))
        # 第二层线性变换后经过sigmoid激活
        x = self.sigmoid(self.linear2(x))
        # 第三层线性变换后经过sigmoid激活，得到最终输出（0-1之间的概率）
        x = self.sigmoid(self.linear3(x))
        return x


# 1. 加载数据集并划分训练集和测试集
# 创建自定义数据集实例，加载糖尿病数据集
dataset = DiabetesDataset('diabetes.csv.gz')

# 计算训练集和测试集的大小（80%训练，20%测试）
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
# 随机划分数据集为训练集和测试集
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])


# 2. 创建数据加载器
# 训练集数据加载器
train_loader = DataLoader(
    dataset=train_dataset,    # 使用训练数据集
    batch_size=32,            # 每次加载32个样本，平衡内存使用和训练效率
    shuffle=True,             # 每个epoch前打乱数据顺序，防止模型过拟合数据顺序
    num_workers=0             # Windows系统下设为0避免多进程问题，Linux/Mac可设为CPU核心数
)

# 测试集数据加载器
test_loader = DataLoader(
    dataset=test_dataset,     # 使用测试数据集
    batch_size=32,            # 测试时同样使用批量加载
    shuffle=False,            # 测试集不需要打乱顺序
    num_workers=0             # 同上
)


# 3. 创建模型、损失函数和优化器
model = DiabetesModel()                          # 实例化神经网络模型
criterion = nn.BCELoss(reduction='mean')         # 二分类交叉熵损失函数（BCELoss）
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # 随机梯度下降优化器，学习率0.01


# 4. 初始化变量用于记录训练过程
train_losses = []       # 存储每个epoch的训练损失
train_accuracies = []   # 存储每个epoch的训练准确率
test_losses = []        # 存储每个epoch的测试损失
test_accuracies = []    # 存储每个epoch的测试准确率
best_accuracy = 0.0     # 记录最佳测试准确率
best_model_path = 'best_diabetes_model.pt'  # 最佳模型保存路径


# 5. 训练模型（共100个epoch）
for epoch in range(100):
    # 训练阶段
    total_train_loss = 0.0  # 累计训练损失
    correct_train = 0       # 训练集正确预测数
    total_train = 0         # 训练集总样本数

    # 遍历训练数据加载器中的每个批次
    for i, (inputs, labels) in enumerate(train_loader):
        # 正向传播：将输入数据传入模型得到输出
        outputs = model(inputs)
        # 计算损失：比较模型输出与真实标签
        loss = criterion(outputs, labels)

        # 反向传播和参数优化
        optimizer.zero_grad()  # 清空梯度缓存，防止梯度累积
        loss.backward()        # 反向传播计算梯度
        optimizer.step()       # 更新模型参数

        # 统计训练损失和准确率
        total_train_loss += loss.item()  # 累加损失值（将张量转换为Python数值）
        # 将输出概率转换为预测标签（大于等于0.5视为1，否则为0）
        predicted = (outputs >= 0.5).float()
        total_train += labels.size(0)    # 累加总样本数
        # 累加正确预测的样本数
        correct_train += (predicted == labels).sum().item()

        # 每10个批次打印一次训练信息
        if i % 10 == 0:
            print(f'Epoch [{epoch + 1}/100], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    # 计算并记录训练集的平均损失和准确率
    avg_train_loss = total_train_loss / len(train_loader)
    train_accuracy = 100 * correct_train / total_train
    train_losses.append(avg_train_loss)
    train_accuracies.append(train_accuracy)

    # 测试阶段（不计算梯度，节省内存和计算资源）
    total_test_loss = 0.0   # 累计测试损失
    correct_test = 0        # 测试集正确预测数
    total_test = 0          # 测试集总样本数

    # 遍历测试数据加载器中的每个批次
    for inputs, labels in test_loader:
        outputs = model(inputs)          # 模型预测
        loss = criterion(outputs, labels)# 计算测试损失
        total_test_loss += loss.item()   # 累加测试损失

        # 计算测试准确率
        predicted = (outputs >= 0.5).float()
        total_test += labels.size(0)
        correct_test += (predicted == labels).sum().item()

    # 计算并记录测试集的平均损失和准确率
    avg_test_loss = total_test_loss / len(test_loader)
    test_accuracy = 100 * correct_test / total_test
    test_losses.append(avg_test_loss)
    test_accuracies.append(test_accuracy)

    # 打印当前epoch的训练和测试指标
    print(f'Epoch [{epoch + 1}/100], Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, '
          f'Test Loss: {avg_test_loss:.4f}, Test Acc: {test_accuracy:.2f}%')

    # 6. 保存性能最好的模型（当测试准确率高于当前最佳时）
    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        torch.save(model.state_dict(), best_model_path)  # 保存模型参数
        print(f'Best model saved with test accuracy: {best_accuracy:.2f}%')


# 7. 可视化训练过程中的损失和准确率变化
plt.figure(figsize=(12, 5))  # 创建一个12x5英寸的图形

# 绘制损失曲线（左子图）
plt.subplot(1, 2, 1)  # 1行2列中的第1个
plt.plot(train_losses, label='Train Loss')    # 训练损失曲线
plt.plot(test_losses, label='Test Loss')      # 测试损失曲线
plt.xlabel('Epoch')                           # x轴标签：迭代次数
plt.ylabel('Loss')                            # y轴标签：损失值
plt.title('Training and Test Loss')           # 图表标题
plt.legend()                                  # 显示图例
plt.grid(True)                                # 显示网格线

# 绘制准确率曲线（右子图）
plt.subplot(1, 2, 2)  # 1行2列中的第2个
plt.plot(train_accuracies, label='Train Accuracy')  # 训练准确率曲线
plt.plot(test_accuracies, label='Test Accuracy')    # 测试准确率曲线
plt.xlabel('Epoch')                                 # x轴标签：迭代次数
plt.ylabel('Accuracy (%)')                          # y轴标签：准确率（百分比）
plt.title('Training and Test Accuracy')             # 图表标题
plt.legend()                                        # 显示图例
plt.grid(True)                                      # 显示网格线

plt.tight_layout()               # 自动调整子图间距
plt.savefig('training_metrics.png')  # 保存图表为PNG文件
plt.show()                       # 显示图表


# 8. 使用最佳模型进行最终测试
final_model = DiabetesModel()    # 创建新的模型实例
# 加载保存的最佳模型参数
final_model.load_state_dict(torch.load(best_model_path))

correct = 0  # 正确预测数
total = 0    # 总样本数

# 在测试集上评估最佳模型
for inputs, labels in test_loader:
    outputs = final_model(inputs)          # 模型预测
    predicted = (outputs >= 0.5).float()   # 转换为标签
    total += labels.size(0)                # 累加总样本数
    correct += (predicted == labels).sum().item()  # 累加正确预测数

# 计算并打印最终准确率
final_accuracy = 100 * correct / total
print(f'Final test accuracy of the best model: {final_accuracy:.2f}%')