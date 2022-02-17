
import os
import paddle
from PIL import Image
import numpy as np

# 读取数据
ROOT = "/home/linxu/Downloads/GoogleDownload/CaptchaDataset-master/CaptchaDataset-master/Classify_Dataset/"
with open(os.path.join(ROOT, "label_dict.txt")) as file:
    labels = eval(file.read())
    pass

# 定义网络
class Net(paddle.nn.Layer):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = paddle.nn.Linear(in_features=3*15*30,out_features=128)
        self.fc2 = paddle.nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# 实例化网络对象
net = Net()
net.set_state_dict(paddle.load("param"))

# 开始预测
img_path = os.path.join(ROOT, "2000.jpg")
# print('img_path', img_path)
_img = Image.open(img_path)
_img = np.array(_img).astype("float32").flatten() / 255
_img = paddle.to_tensor([_img], dtype="float32")
infer_number = net(_img)
print("预测2000.jpg的概率分布为：",paddle.nn.functional.softmax(infer_number))
print("模型认为，概率最大的是：",np.argmax(infer_number.numpy()[0]))

