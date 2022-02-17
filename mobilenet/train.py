import os
import shutil
# import mobilenet_v1
import paddle as paddle
# import reader
import paddle.fluid as fluid
from . import mobilenet_v1
from . import reader



crop_size = 224
resize_size = 250

# 定义输入层
image = fluid.layers.data(name='image', shape=[3, crop_size, crop_size], dtype='float32')
label = fluid.layers.data(name='label', shape=[1], dtype='int64')


# 获取分类器，因为这次只爬取了6个类别的图片，所以分类器的类别大小为6
model = mobilenet_v1.net(image, 6)


# 获取损失函数和准确率函数
cost = fluid.layers.cross_entropy(input=model, label=label)
avg_cost = fluid.layers.mean(cost)
acc = fluid.layers.accuracy(input=model, label=label)

# 获取训练和测试程序
test_program = fluid.default_main_program().clone(for_test=True)

# 定义优化方法
optimizer = fluid.optimizer.AdamOptimizer(learning_rate=1e-3,
                                          regularization=fluid.regularizer.L2DecayRegularizer(1e-4))
opts = optimizer.minimize(avg_cost)

# 获取自定义数据
train_reader = paddle.batch(reader=reader.train_reader('images/train.list', crop_size, resize_size), batch_size=32)
test_reader = paddle.batch(reader=reader.test_reader('images/test.list', crop_size), batch_size=32)


