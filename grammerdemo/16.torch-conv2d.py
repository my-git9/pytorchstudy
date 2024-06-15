import torch
import torch.nn as nn


# 1.创建输入特征图
input_feat = torch.tensor([[4,5,2,5],[2,4,5,1],[6,2,8,1],[4,6,2,7]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

# 2.创建一个 2*2 的卷积
conv2d = nn.Conv2d(1, 1, (2, 2), stride=1, padding='same', bias=True)

# 卷积核要有四个维度(输入通道数，输出通道数，高，宽)
kernels = torch.tensor([[[[1, 0], [2, 1]]]], dtype=torch.float32)
conv2d.weight = nn.Parameter(kernels, requires_grad=False)

# 3.计算
output = conv2d(input_feat)
print(output)
