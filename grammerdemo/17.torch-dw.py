import torch
import torch.nn as nn

# 生成一个三通道的 5x5 特征图
x = torch.randn((3, 5, 5)).unsqueeze(0)

# 请注意 DW 中，输入特征通道数与输出通道数是一样的
in_channels_dw = x.shape[1]
out_channels_dw = x.shape[1]

# 一般来讲DW卷积的kernel size为3
kernel_size = 3
stride = 1

# DW 卷积 groups 参数与输入参数通道一致
dw = nn.Conv2d(in_channels_dw, out_channels_dw, kernel_size, stride, groups=in_channels_dw)

# PW 卷积
in_channels_pw = out_channels_dw
out_channels_pw = 4
kernel_size_pw = 1
pw = nn.Conv2d(in_channels_pw, out_channels_pw, kernel_size_pw, stride)

# 输出
out = pw(dw(x))
print(out.shape)

