from IPython.core.display_functions import display
from PIL import Image
from torchvision import transforms

# 定义标准化操作
norm_oper = transforms.Normalize((0.5,0.5,0.5,0.5),(0.5,0.5,0.5,0.5))

# 原图
orig_img = Image.open('img/jk.jpg')
display(orig_img)

# 图像转化为 Tensor
img_tensor = transforms.ToTensor()(orig_img)

# 标准化
tensor_norm = norm_oper(img_tensor)

# Tensor 转为图像
img_norm = transforms.ToPILImage()(tensor_norm)
display(img_norm)
