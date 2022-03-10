import paddle
from paddle.vision.transforms import ToTensor

# 加载数据集
train_dataset = paddle.vision.datasets.MNIST(mode='train', transform=ToTensor())
val_dataset =  paddle.vision.datasets.MNIST(mode='test', transform=ToTensor())

# 模型组网
mnist = paddle.nn.Sequential(
    paddle.nn.Flatten(),
    paddle.nn.Linear(784, 512),
    paddle.nn.ReLU(),
    paddle.nn.Dropout(0.2),
    paddle.nn.Linear(512, 10)
)

# 封装模型
model = paddle.Model(mnist)

# 配置训练模型
model.prepare(paddle.optimizer.Adam(parameters=model.parameters()),
              paddle.nn.CrossEntropyLoss(),
              paddle.metric.Accuracy())

# 模型训练
model.fit(train_dataset,
          epochs=5,
          batch_size=64,
          verbose=1)
# 模型评估
model.evaluate(val_dataset, verbose=0)

# 模型测试
model.predict(val_dataset)
