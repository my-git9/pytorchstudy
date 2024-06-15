from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

im = Image.open('img/jk.webp')
im_pillow = np.array(im)
im_pillow1 = np.array(im)
im_pillow2 = np.array(im)
im_pillow3 = np.array(im)
im_pillow1[:,:,1:]=0
im_pillow2[:,:,2:]=0
im_pillow3[:,:,0]=0

# 绘图
plt.subplot(2, 2, 1)
plt.title('Origin Image')
plt.imshow(im_pillow)
plt.axis('off')
plt.subplot(2, 2, 2)
plt.title('Red Channel')
plt.imshow(im_pillow1.astype(np.uint8))
plt.axis('off')
plt.subplot(2, 2, 3)
plt.title('Green Channel')
plt.imshow(im_pillow2.astype(np.uint8))
plt.axis('off')
plt.subplot(2, 2, 4)
plt.title('Blue Channel')
plt.imshow(im_pillow3.astype(np.uint8))
plt.axis('off')
plt.savefig('./rgb_pillow.png', dpi=150)
