from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

# 读取图片
im = Image.open('img/jk.webp')
# 打印图片大小
print(im.size) # (318, 116)

# 将 Pillow 的数据转换为 NumPy 的数组格式
im_pillow = np.asarray(im)
# 打印数组形状
print(im_pillow.shape) # (116, 318, 3)

# 获取对应通道的数据
im_pillow_c1 = im_pillow[:, :, 0]
im_pillow_c2 = im_pillow[:, :, 1]
im_pillow_c3 = im_pillow[:, :, 2]
print(im_pillow_c1.shape) # (116, 318)

# 生成一个全 0 数组，该数组要与 im_pillow 具有相同的宽高
zeros = np.zeros((im_pillow.shape[0], im_pillow.shape[1], 1)) #
print(zeros.shape) # (116, 318, 1)

# 数组拼接：只需要将全 0 的数组与 im_pillow_c1、im_pillow_c2、im_pillow_c3 进行拼接
# zeros 于 im_pillow_c1 的维度不一样 (116, 318, 0) / (116, 318)，此处需要将 im_pillow_c1 转换为 3 个维度的
im_pillow_c1 = im_pillow_c1[:,:,np.newaxis]
im_pillow_c2 = im_pillow_c2[:,:,np.newaxis]
im_pillow_c3 = im_pillow_c3[:,:,np.newaxis]
print(im_pillow_c1.shape)

# 数组拼接
# axis：沿 2 轴方向
im_pillow_c1_3ch = np.concatenate((im_pillow_c1,zeros,zeros),axis=2)
im_pillow_c2_3ch = np.concatenate((im_pillow_c2,zeros,zeros),axis=2)
im_pillow_c3_3ch = np.concatenate((im_pillow_c2,zeros,zeros),axis=2)

# 绘图
plt.subplot(2, 2, 1)
plt.title('Origin Image')
plt.imshow(im_pillow)
plt.axis('off')
plt.subplot(2, 2, 2)
plt.title('Red Channel')
plt.imshow(im_pillow_c1_3ch.astype(np.uint8))
plt.axis('off')
plt.subplot(2, 2, 3)
plt.title('Green Channel')
plt.imshow(im_pillow_c2_3ch.astype(np.uint8))
plt.axis('off')
plt.subplot(2, 2, 4)
plt.title('Blue Channel')
plt.imshow(im_pillow_c3_3ch.astype(np.uint8))
plt.axis('off')
plt.savefig('./rgb_pillow.png', dpi=150)


