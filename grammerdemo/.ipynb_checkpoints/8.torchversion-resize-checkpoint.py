from IPython.core.display_functions import display
from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt

# 定义一个 Resize 操作
resize_img_oper = transforms.Resize((200,200), interpolation=2)

# 原图
orig_img = Image.open('img/jk.jpg')
display(orig_img)

# Resize 操作后的图
img = resize_img_oper(orig_img)
display(img)
