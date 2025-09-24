import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号


class LinearRegression:
    """线性回归模型 y = wx + b"""

    def __init__(self, learning_rate=0.01, epochs=1000):
        self.w = np.random.randn()  # 随机初始化权重
        self.b = np.random.randn()  # 随机初始化偏置
        self.learning_rate = learning_rate  # 学习率
        self.epochs = epochs  # 训练轮数

    def forward(self, x):
        """前向传播计算预测值"""
        return self.w * x + self.b

    def compute_loss(self, y_pred, y_true):
        """计算均方误差损失"""
        return np.mean((y_pred - y_true) ** 2)

    def backward(self, x, y_pred, y_true):
        """反向传播计算梯度"""
        n = len(x)
        dw = (2 / n) * np.sum((y_pred - y_true) * x)  # w的梯度
        db = (2 / n) * np.sum(y_pred - y_true)  # b的梯度
        return dw, db

    def update_parameters(self, dw, db):
        """更新参数"""
        self.w -= self.learning_rate * dw
        self.b -= self.learning_rate * db

    def train(self, x, y, track_params=False):
        """训练模型"""
        losses = []
        weights = []
        biases = []

        for epoch in range(self.epochs):
            # 前向传播
            y_pred = self.forward(x)

            # 计算损失
            loss = self.compute_loss(y_pred, y)
            losses.append(loss)

            # 记录参数（如果需要）
            if track_params:
                weights.append(self.w)
                biases.append(self.b)

            # 反向传播和参数更新
            dw, db = self.backward(x, y_pred, y)
            self.update_parameters(dw, db)

            # 每100轮打印一次损失
            if (epoch + 1) % 100 == 0:
                print(f"Epoch [{epoch + 1}/{self.epochs}], Loss: {loss:.4f}")

        # 返回训练历史
        return {
            'losses': losses,
            'weights': weights,
            'biases': biases
        }


def visualize_relationships(model, history, x, y):
    """可视化w与loss、b与loss的关系"""
    # 绘制w和loss的关系
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history['weights'], history['losses'], 'b-')
    plt.scatter(history['weights'], history['losses'], c='r', s=10)
    plt.xlabel('权重 w')
    plt.ylabel('损失 Loss')
    plt.title('w与loss的关系')
    plt.grid(True, linestyle='--', alpha=0.7)

    # 绘制b和loss的关系
    plt.subplot(1, 2, 2)
    plt.plot(history['biases'], history['losses'], 'g-')
    plt.scatter(history['biases'], history['losses'], c='r', s=10)
    plt.xlabel('偏置 b')
    plt.ylabel('损失 Loss')
    plt.title('b与loss的关系')
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

    # 绘制最终拟合曲线
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, color='blue', label='原始数据')
    plt.plot(x, model.forward(x), 'r-', label=f'拟合直线: y = {model.w:.4f}x + {model.b:.4f}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('线性回归拟合结果')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()


def main():
    try:
        # 1. 使用pandas读取数据
        print("读取训练数据...")
        df = pd.read_csv('D:/Git/train.csv')

        # 假设数据集中有'x'和'y'两列
        if 'x' not in df.columns or 'y' not in df.columns:
            raise ValueError("train.csv中必须包含'x'和'y'列")

        x = df['x'].values
        y = df['y'].values

        print(f"成功读取数据，共{len(df)}个样本")

        # 2. 初始化并训练模型
        print("开始训练线性回归模型...")
        model = LinearRegression(learning_rate=0.01, epochs=1000)
        history = model.train(x, y, track_params=True)

        # 3. 输出最终参数
        print(f"\n训练完成，最终参数:")
        print(f"w = {model.w:.4f}")
        print(f"b = {model.b:.4f}")
        print(f"最终损失: {history['losses'][-1]:.4f}")

        # 4. 可视化结果
        visualize_relationships(model, history, x, y)

    except FileNotFoundError:
        print("错误: 找不到train.csv文件，请确保文件存在于当前目录下")
    except Exception as e:
        print(f"发生错误: {str(e)}")


if __name__ == "__main__":
    main()