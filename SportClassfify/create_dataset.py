# 预处理数据，将其转化为标准格式。同时将数据拆分成两份，以便训练和计算预估准确率
import codecs
import os
import random
import shutil
from PIL import Image

# 获取项目
# all_file_dir = 'dataset/train'
all_file_dir = '/home/linxu/Desktop/dataset/train'
class_list = [c for c in os.listdir(all_file_dir) if
              os.path.isdir(os.path.join(all_file_dir, c)) and not c.endswith('Set') and not c.startswith('.')]
# 重新排序
class_list.sort()

# 配置数据目录
train_image_dir = "/home/linxu/Desktop/dataset/train"
test_image_dir = "/home/linxu/Desktop/dataset/test"
eval_image_dir = "/home/linxu/Desktop/dataset/valid"


# 生成列表
def generate_list(save_dir, dir_name):
    file_list = []
    for sub_dir in class_list:
        label = class_list.index(sub_dir)
        img_paths = os.listdir(os.path.join(dir_name, sub_dir))
        print('img_paths', img_paths)
        for img_path in img_paths:
            context = dir_name + '/'+ os.path.join(sub_dir, img_path) + '\t%d' % label + '\n'
            print('context',context)
            file_list.append(context)
    return file_list


# 保存列表
def save_list2file(mylist, filename):
    with open(filename, 'w') as f:
        f.writelines(mylist)


save_dir = '/home/linxu/Desktop/dataset/'

train_list = generate_list(save_dir, train_image_dir)
test_list = generate_list(save_dir, test_image_dir)
eval_list = generate_list(save_dir, eval_image_dir)
print('train_list', train_list)

save_list2file(train_list, save_dir + 'train_list.txt')
save_list2file(test_list, save_dir + 'test_list.txt')
save_list2file(eval_list, save_dir + 'eval_list.txt')

# 打印索引及对应项目
for item in class_list:
    print(class_list.index(item), item)
