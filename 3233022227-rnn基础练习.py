import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.table import Table



# 1. 全局配置与数据预处理
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

chars = ['d', 'l', 'e', 'a', 'r', 'n']
idx2char = chars
char2idx = {char: idx for idx, char in enumerate(chars)}

x_str = "dlearn"
y_str = "lanrla"
x_data = [char2idx[char] for char in x_str]
y_data = [char2idx[char] for char in y_str]

# 超参数
input_size = len(chars)
hidden_size = 8
num_classes = len(chars)  # 分类数=字符集大小（0-5）
batch_size = 1
seq_len = len(x_str)
num_epochs = 20
learning_rate = 0.1

# 数据格式转换
one_hot_lookup = torch.eye(input_size)
x_one_hot = one_hot_lookup[x_data]
inputs = x_one_hot.view(seq_len, batch_size, input_size)
labels_rnncell = torch.LongTensor(y_data).view(-1, 1)
labels_rnn = torch.LongTensor(y_data)


# 2. 模型定义
class RNNCellModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, batch_size):
        super(RNNCellModel, self).__init__()
        self.batch_size = batch_size
        self.rnncell = torch.nn.RNNCell(input_size=input_size, hidden_size=hidden_size)
        self.fc = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, input_t, hidden):
        hidden = self.rnncell(input_t, hidden)
        output = self.fc(hidden)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(self.batch_size, hidden_size)


class RNNModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=1):
        super(RNNModel, self).__init__()
        self.num_layers = num_layers
        self.rnn = torch.nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=False
        )
        self.fc = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, input_seq):
        hidden_init = torch.zeros(self.num_layers, batch_size, hidden_size)
        out, _ = self.rnn(input_seq, hidden_init)
        out = self.fc(out)
        return out.view(-1, num_classes)


# 3. 模型训练
def train_rnncell(model, inputs, labels, criterion, optimizer, num_epochs):
    train_log = {"epoch": [], "loss": [], "predicted_str": []}
    for epoch in range(num_epochs):
        loss_total = 0.0
        optimizer.zero_grad()
        hidden = model.init_hidden()
        predicted_chars = []

        for input_t, label_t in zip(inputs, labels):
            output, hidden = model(input_t, hidden)
            loss_total += criterion(output, label_t)
            _, idx = output.max(dim=1)
            predicted_chars.append(idx2char[idx.item()])

        loss_total.backward()
        optimizer.step()

        predicted_str = ''.join(predicted_chars)
        train_log["epoch"].append(epoch + 1)
        train_log["loss"].append(loss_total.item())
        train_log["predicted_str"].append(predicted_str)

        print(f"RNNCell | Epoch [{epoch + 1}/{num_epochs}] | Loss: {loss_total.item():.4f} | Pred: {predicted_str}")
    return train_log


def train_rnn(model, inputs, labels, criterion, optimizer, num_epochs):
    train_log = {"epoch": [], "loss": [], "predicted_str": []}
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        _, idx = outputs.max(dim=1)
        predicted_chars = [idx2char[i.item()] for i in idx]
        predicted_str = ''.join(predicted_chars)

        train_log["epoch"].append(epoch + 1)
        train_log["loss"].append(loss.item())
        train_log["predicted_str"].append(predicted_str)

        print(f"RNN     | Epoch [{epoch + 1}/{num_epochs}] | Loss: {loss.item():.4f} | Pred: {predicted_str}")
    return train_log


# 初始化模型并训练
rnncell_net = RNNCellModel(input_size, hidden_size, num_classes, batch_size)
criterion = torch.nn.CrossEntropyLoss()
optimizer_rnncell = torch.optim.Adam(rnncell_net.parameters(), lr=learning_rate)

rnn_net = RNNModel(input_size, hidden_size, num_classes)
optimizer_rnn = torch.optim.Adam(rnn_net.parameters(), lr=learning_rate)

print("\n" + "=" * 50 + " 开始训练RNNCell " + "=" * 50)
rnncell_log = train_rnncell(rnncell_net, inputs, labels_rnncell, criterion, optimizer_rnncell, num_epochs)

print("\n" + "=" * 50 + " 开始训练RNN " + "=" * 50)
rnn_log = train_rnn(rnn_net, inputs, labels_rnn, criterion, optimizer_rnn, num_epochs)


# 4. 训练过程可视化
def plot_loss_comparison(rnncell_log, rnn_log):
    """损失对比曲线"""
    plt.figure(figsize=(10, 6))
    plt.plot(rnncell_log["epoch"], rnncell_log["loss"],
             color="#FF6B6B", linewidth=2.5, marker="o", markersize=4, label="RNNCell")
    plt.plot(rnn_log["epoch"], rnn_log["loss"],
             color="#4ECDC4", linewidth=2.5, marker="s", markersize=4, label="RNN")

    plt.xlabel("Epoch（训练轮次）", fontsize=12)
    plt.ylabel("Loss（损失值）", fontsize=12)
    plt.title("RNNCell vs RNN 损失变化对比", fontsize=14, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(rnncell_log["epoch"][::2])
    plt.tight_layout()
    plt.savefig("loss_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_prediction_table(rnncell_log, rnn_log):
    """关键Epoch预测结果表格"""
    key_epochs = [0, len(rnncell_log["epoch"]) // 2, len(rnncell_log["epoch"]) - 1]
    table_data = []
    for idx in key_epochs:
        epoch = rnncell_log["epoch"][idx]
        rnncell_pred = rnncell_log["predicted_str"][idx]
        rnn_pred = rnn_log["predicted_str"][idx]
        rnncell_loss = f"{rnncell_log['loss'][idx]:.4f}"
        rnn_loss = f"{rnn_log['loss'][idx]:.4f}"
        table_data.append([epoch, rnncell_pred, rnncell_loss, rnn_pred, rnn_loss])

    # 创建表格图
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_axis_off()  # 隐藏坐标轴
    columns = ["Epoch", "RNNCell预测结果", "RNNCell损失", "RNN预测结果", "RNN损失"]
    table = Table(ax, bbox=[0, 0, 1, 1])  # 表格占满整个图

    for i, col in enumerate(columns):
        # 添加表头单元格，捕获返回的Cell对象
        cell = table.add_cell(
            row=0, col=i, width=1 / 5, height=0.2,
            text=col, loc="center", facecolor="#4ECDC4"
        )
        # 单独设置表头文本加粗（兼容所有matplotlib版本）
        cell.set_text_props(fontweight="bold")

    for row_idx, row_data in enumerate(table_data):
        for col_idx, cell_data in enumerate(row_data):
            # 定义单元格背景色
            if col_idx in [1, 3] and cell_data == y_str:
                facecolor = "#FFE66D"  # 目标字符串背景色（黄色）
            else:
                facecolor = "#F7F9F9"  # 普通单元格背景色（浅灰）

            # 添加数据单元格，捕获返回的Cell对象
            cell = table.add_cell(
                row=row_idx + 1, col=col_idx, width=1 / 5, height=0.2,
                text=cell_data, loc="center", facecolor=facecolor
            )

            # 对目标字符串设置文本属性（红色+加粗）
            if col_idx in [1, 3] and cell_data == y_str:
                cell.set_text_props(color="#FF0000", fontweight="bold")

    # 将表格添加到坐标轴
    ax.add_table(table)
    # 设置标题
    plt.title(f"关键Epoch预测结果对比（目标：{y_str}）", fontsize=14, fontweight="bold", pad=20)
    # 保存表格图片
    plt.savefig("prediction_table.png", dpi=300, bbox_inches="tight")
    plt.show()


# 执行可视化
plot_loss_comparison(rnncell_log, rnn_log)
plot_prediction_table(rnncell_log, rnn_log)