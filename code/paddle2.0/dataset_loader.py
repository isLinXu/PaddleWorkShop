import paddle

# 方法一：飞桨框架自带数据集
print('视觉相关数据集：', paddle.vision.datasets.__all__)
print('自然语言相关数据集：', paddle.text.__all__)

# 使用飞桨框架自带数据集
from paddle.vision.transforms import ToTensor
# 训练数据集 用ToTensor将数据格式转为Tensor
train_dataset = paddle.vision.datasets.MNIST(mode='train', transform=ToTensor())

# 验证数据集
val_dataset = paddle.vision.datasets.MNIST(mode='test', transform=ToTensor())

# 方法二：自定义数据集
from paddle.io import Dataset

BATCH_SIZE = 64
BATCH_NUM = 20

IMAGE_SIZE = (28, 28)
CLASS_NUM = 10


class MyDataset(Dataset):
    """
    步骤一：继承paddle.io.Dataset类
    """
    def __init__(self, num_samples):
        """
        步骤二：实现构造函数，定义数据集大小
        """
        super(MyDataset, self).__init__()
        self.num_samples = num_samples

    def __getitem__(self, index):
        """
        步骤三：实现__getitem__方法，定义指定index时如何获取数据，并返回单条数据（训练数据，对应的标签）
        """
        data = paddle.uniform(IMAGE_SIZE, dtype='float32')
        label = paddle.randint(0, CLASS_NUM-1, dtype='int64')

        return data, label

    def __len__(self):
        """
        步骤四：实现__len__方法，返回数据集总数目
        """
        return self.num_samples

# 测试定义的数据集
custom_dataset = MyDataset(BATCH_SIZE * BATCH_NUM)

print('=============custom dataset=============')
for data, label in custom_dataset:
    print(data.shape, label.shape)
    break

# 数据加载

train_loader = paddle.io.DataLoader(custom_dataset, batch_size=BATCH_SIZE, shuffle=True)
# 如果要加载内置数据集，将 custom_dataset 换为 train_dataset 即可

for batch_id, data in enumerate(train_loader()):
    x_data = data[0]
    y_data = data[1]

    print(x_data.shape)
    print(y_data.shape)
    break
