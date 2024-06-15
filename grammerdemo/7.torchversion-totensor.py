from PIL import Image
from torchvision import transforms
from IPython.display import display

img = Image.open('img/jk.webp')
# display: 用于显示（渲染）Python 对象的输出，使用 display 函数可以在 Notebook 中以更友好的形式显示数据
display(img)

print(type(img))
# output --> <class 'PIL.WebPImagePlugin.WebPImageFile'>

img1 = transforms.ToTensor()(img)
print(type(img1))
# output --> <class 'torch.Tensor'>

# Tensor 转换为 PIL.Image
img2 = transforms.ToPILImage()(img1)
print(type(img2))
# output --> <class 'PIL.Image.Image'>
