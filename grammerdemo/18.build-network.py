import numpy as np
import random
#from matplotlib import pyplot as plt
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

# 升级随机训练集
w = 2
b = 3
xlim = [-10, 10]
x_train = np.random.randint(low=xlim[0], high=xlim[1], size=30)
# y_train 对应数据 label
y_train = [w * x + b + random.randint(0, 2) for x in x_train]
# plt.plot(x_train, y_train)

# 构建模型
class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1))
        self.bais = nn.Parameter(torch.randn(1))

    def forward(self, input):
        return (input * self.weight) + self.bais

# 创建模型
model = LinearModel()
# 优化器
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, weight_decay=1e-2, momentum=0.9)

# Tensorboard
writer = SummaryWriter()

# 数据转为 tensor
y_train = torch.tensor(y_train, dtype=torch.float32)
# 训练
for _ in range(1000):
    # 定义输入
    input = torch.from_numpy(x_train)
    # 获得模型的输出结果，也即是当前模型学到的效果
    output = model(input)
    # 获得输出结果和数据真正类别的损失函数
    loss = nn.MSELoss()(output, y_train)
    # 首先要通过zero_grad()函数把梯度清零，不然PyTorch每次计算梯度会累加，不清零的话第二次算的梯度等于第一次加第二次的
    model.zero_grad()
    # 算完 loss 之后进行反向梯度传播，这个过程之后梯度会记录在变量中
    loss.backward()
    # 用计算的梯度去做优化
    optimizer.step()
    # use Tensorboard
    writer.add_scalar('Loss/train', loss, n_iter)

for parameter in model.parameters():
    print(parameter)