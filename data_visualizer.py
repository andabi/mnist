import numpy as np
import gzip as gz
from struct import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm

fp_image = gz.open('MNIST_data/train-images-idx3-ubyte.gz', 'rb')
fp_label = gz.open('MNIST_data/train-labels-idx1-ubyte.gz', 'rb')

img = np.zeros((28, 28))
lbl = [[] for i in range(10)]
d = 0
l = 0
index = 0

fp_image.read(16)
fp_label.read(8)

while True:
    s = fp_image.read(784)
    l = fp_label.read(1)

    if not s:
        break
    if not l:
        break

    index = int(l[0])

    img = np.reshape(unpack(len(s) * 'B', s), (28, 28))
    lbl[index].append(img)

m_img = []
for i in range(0,10):
    m_img.append(np.mean(lbl[i], axis=0))

for i in range(0,10):
    plt.imshow(m_img[i], cmap=cm.binary)
    plt.show()