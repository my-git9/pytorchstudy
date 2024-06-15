from IPython.core.display_functions import display
from PIL import Image
from torchvision import transforms

# 中心剪裁：在中心裁剪指定的 PIL Image 或 Tensor
center_crop_oper = transforms.CenterCrop((60,70))
# 随机剪裁：在一个随机位置剪裁指定的 PIL Image 或 Tensor
random_crop_oper = transforms.RandomCrop((80,80))
# FiveCrop：分别从四角和中心进行剪裁，共剪裁成五块
five_crop_oper = transforms.FiveCrop((60,70))

# 原图
orig_img = Image.open('img/jk.jpg')
display(orig_img)

# 中心剪裁
img1 = center_crop_oper(orig_img)
display(img1)

# 随机剪裁
img2 = random_crop_oper(orig_img)
display(img2)

# 四角和中心剪裁
img3 = five_crop_oper(orig_img)
for img in img3:
    display(img)
