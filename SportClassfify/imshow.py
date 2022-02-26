import matplotlib.pyplot as plt

import os

for i in range(1, 11):
    plt.subplot(2, 5, i)
    filename = os.path.join('/home/linxu/Desktop/dataset/train/table tennis', str(i).zfill(3) + '.jpg')
    print('filename', filename)
    img = plt.imread(filename)
    plt.imshow(img)
plt.show()