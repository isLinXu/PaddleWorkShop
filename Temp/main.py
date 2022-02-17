
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

# 定义损失函数
loss_func = paddle.nn.CrossEntropyLoss()

# 优化器
opt = paddle.optimizer.SGD(parameters=net.parameters())

# 组建训练程序
for epoch in range(10):
    for img_id in range(1, 1000):
        img_name = f"{img_id}.jpg"
        img_path = os.path.join(ROOT,img_name)

        _img = Image.open(img_path)
        _img = np.array(_img).astype("float32").flatten() / 255
        _img = paddle.to_tensor([_img], dtype="float32")

        _label = labels[img_name]
        _label = paddle.to_tensor([_label], dtype="int64")
        number_prob = net(_img)

        loss = loss_func(number_prob, _label)
        loss.backward()

        opt.step()
        opt.clear_gradients()
        print(f"Epoch:{epoch}\t loss:{loss.numpy()}")


# 保存网络模型参数
paddle.save(net.state_dict(), "param")


