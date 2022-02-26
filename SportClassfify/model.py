from paddle.io import Dataset
import paddle.vision.transforms as T
import numpy as np
from PIL import Image
import os


class SportsDataset(Dataset):
    def __init__(self, path,mode, transform):
        self.transform = transform
        super(SportsDataset, self).__init__()
        """
        初始化函数
        """
        self.data = []
        self.mode = mode
        # with open('{}_list.txt'.format(mode)) as f:
        #     for line in f.readlines():
        #         info = line.strip().split('\t')
        #         if len(info) > 0:
        #             # print(info[0].strip())
        #             self.data.append([info[0].strip(), info[1].strip()])
        with open(path) as f:
            for line in f.readlines():
                info = line.strip().split('\t')
                if len(info) > 0:
                    # print(info[0].strip())
                    self.data.append([info[0].strip(), info[1].strip()])


    def get_origin_data(self):
        return self.data

    def __getitem__(self, index):
        """
        根据索引获取单个样本
        """
        image_file, label = self.data[index]

        image = Image.open(os.path.join('dataset/{}'.format(self.mode), image_file))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = self.transform(image)
        return image, np.array(label, dtype='int64')

    def __len__(self):
        return len(self.data)


# train_transform = T.Compose([
#     T.RandomHorizontalFlip(),  # 随机水平反转,默认0.5概率
#     T.ColorJitter(0.4, 0.4, 0.4, 0.4),  # 随机调整图像的亮度，对比度，饱和度和色调。
#     T.ToTensor(),  # 数据的格式转换和标准化 HWC => CHW
#     T.Normalize()  # 图像归一化
# ])
#
# eval_transform = T.Compose([
#     T.ToTensor(),  # 数据的格式转换和标准化 HWC => CHW
#     T.Normalize()  # 图像归一化
# ])
#
# test_transform = T.Compose([
#     T.Resize((224, 224)),
#     T.ToTensor(),  # 数据的格式转换和标准化 HWC => CHW
#     T.Normalize()  # 图像归一化
# ])
#
# train_dataset = SportsDataset(mode='train', transform=train_transform)
# test_dataset = SportsDataset(mode='test', transform=test_transform)
# eval_dataset = SportsDataset(mode='eval', transform=eval_transform)


