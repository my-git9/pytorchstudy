from IPython.core.display_functions import display
from PIL import Image
from torchvision import transforms

# 定义翻转动作
# 水平翻转
h_flip_oper = transforms.RandomHorizontalFlip(p=1)
# 垂直翻转
v_flip_oper = transforms.RandomVerticalFlip(p=1)

# 原图
orig_img = Image.open('img/jk.jpg')
display(orig_img)

# 翻转
img1 = h_flip_oper(orig_img)
display(img1)

img2 = v_flip_oper(orig_img)
display(img2)
