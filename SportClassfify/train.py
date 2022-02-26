# 载入模型
from paddle.vision.models import MobileNetV2
import paddle.vision.transforms as T
from SportClassfify.model import SportsDataset

import paddle.nn as nn

import paddle


model = MobileNetV2(num_classes=100)


model = paddle.Model(model)
model.prepare(optimizer=paddle.optimizer.Adam(parameters=model.parameters()),
              loss=paddle.nn.CrossEntropyLoss(),
              metrics=paddle.metric.Accuracy())

train_transform = T.Compose([
    T.RandomHorizontalFlip(),  # 随机水平反转,默认0.5概率
    T.ColorJitter(0.4, 0.4, 0.4, 0.4),  # 随机调整图像的亮度，对比度，饱和度和色调。
    T.ToTensor(),  # 数据的格式转换和标准化 HWC => CHW
    T.Normalize()  # 图像归一化
])

eval_transform = T.Compose([
    T.ToTensor(),  # 数据的格式转换和标准化 HWC => CHW
    T.Normalize()  # 图像归一化
])

test_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),  # 数据的格式转换和标准化 HWC => CHW
    T.Normalize()  # 图像归一化
])

train_dataset = SportsDataset(path='/home/linxu/Desktop/dataset/train_list.txt', mode='train',
                              transform=train_transform)
test_dataset = SportsDataset(path='/home/linxu/Desktop/dataset/test_list.txt', mode='test', transform=test_transform)
eval_dataset = SportsDataset(path='/home/linxu/Desktop/dataset/eval_list.txt', mode='eval', transform=eval_transform)


# 训练
model.fit(train_dataset,
          test_dataset,
          epochs=100,
          batch_size=32,
          drop_last=True,
          log_freq=1,
          shuffle=True,
          verbose=2)

# save for training
# model.save("training_final.pdparams")

# save for inference
model.save('inference_model', training=False)

eval_result = model.evaluate(test_dataset, verbose=1)
print(eval_result)
