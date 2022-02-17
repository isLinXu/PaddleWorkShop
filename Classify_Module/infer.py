# Author: Acer Zhang
# Datetime: 2020/10/19 
# Copyright belongs to the author.
# Please indicate the source for reprinting.
from paddle.fluid.reader import DataLoader

from Classify_Module.reader import InferReader
from Classify_Module.train import *

if __name__ == '__main__':
    model = pp.Model(Net(is_infer=True), inputs=input_define)
    model.load("/home/linxu/Documents/PaddleWorkShop/output/final")
    model.prepare()
    infer_reader = InferReader(DATA_PATH)
    # loader = DataLoader(infer_reader,batch_size=1)
    # data = next(loader())
    # infer_reader = model.predict(test_data=loader)[0]
    result = model.predict(test_data=infer_reader)[0]

    img_list = infer_reader.get_names()
    img_index = 0
    for mini_batch in result:
        for sample in mini_batch:
            print(f"{img_list[img_index]}的推理结果为:{sample}")
            img_index += 1
