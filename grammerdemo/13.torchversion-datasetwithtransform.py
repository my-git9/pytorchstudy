from torchvision import transforms, datasets

# 定义一个 transform
# 1.转换为 tensor
# 2.标准化
my_transform = transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize((0.5),(0.5))
                                   ])

# 读取 MINSET 数据集
mniset_dataset = datasets.MNIST(root='./data',
                                train=False,
                                transform=my_transform,
                                target_transform=None,
                                download=False)

# 查看变换后的数据类型
item = mniset_dataset.__getitem__(0)
print(type(item[0]))
