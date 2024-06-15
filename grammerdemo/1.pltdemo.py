# 绘画一个 y = x ** 2 的图像向量图
import numpy as np
import matplotlib.pyplot as plt

# 生成一个 -50 到 51，step 为 2 的数组
X = np.arange(-50, 51, 2)
Y = X ** 2
print(X)

# 绘图
plt.plot(X, Y, color='blue')
plt.legend()
plt.show()