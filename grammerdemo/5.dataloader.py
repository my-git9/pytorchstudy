import torch
from torch.utils.data import DataLoader, Dataset

class MyDataset(Dataset):
    # 构造函数
    def __init__(self, data_tensor, target_tensor):
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor
    # 返回数据集大小
    def __len__(self):
        return self.data_tensor.size(0)
    # 返回索引的数据与标签
    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index]

# 生成数据
# torch.randn 用于生成数据类型为浮点型且维度指定的随机 Tensor，随机生成的浮点数的取值满足均值为 0、方差为 1 的标准正态分布
# 10 * 3
data_tensor = torch.randn(10, 3)
# torch.randint(low, high, size) 用于生成随机整数的 Tensor，其内部填充的是在[low,high)均匀生成的随机整数
target_tensor = torch.randint(2,(10,)) # 标签是 0 或 1，
# 将数据封装成 Dataset
my_dataset = MyDataset(data_tensor, target_tensor)

tenser_dataloader = DataLoader(dataset= my_dataset, # 传入数据集
                               batch_size=5, # 输出 batch 大小
                               shuffle=True, # 数据是否打乱
                               num_workers=0) # 进程数，0 表示只有主进程

for data, target in tenser_dataloader:
    print(data, target)
