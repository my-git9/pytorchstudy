import torchvision
from IPython.core.display_functions import display

mnist_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    transform=None,
    target_transform=None,
    download=True)
mnist_dataset_list = list(mnist_dataset)
print(mnist_dataset_list)

display(mnist_dataset_list[0][0])
print("Image label is:", mnist_dataset_list[0][1])