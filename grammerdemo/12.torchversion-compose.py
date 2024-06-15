from IPython.core.display_functions import display
from PIL import Image
from torchvision import transforms

# 原图
orig_img = Image.open('img/jk.jpg')
display(orig_img)

# 定义组合操作
composed = transforms.Compose([transforms.Resize((200,200)),
                               transforms.RandomCrop(80)])

img = composed(orig_img)
display(img)