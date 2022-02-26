import paddle
import paddle.fluid as fluid
import numpy
# from paddle.utils import Ploter
from PIL import Image
import matplotlib.pyplot as plt
# % matplotlibinline

# 全局变量
use_cuda = 1  # 是否使用GPU
batch_size = 128  # 每批读取数据
epoch_num = 50  # 训练迭代周期
save_dirname = "./model/Image_Classification.vggnet.model"  # 模型保存路径
file_dirname = "./image/dog.png"  # 预测图像路径


# vggnet
def conv_block(input, num_filter, groups, dropouts):
    return fluid.nets.img_conv_group(
        input=input,
        conv_num_filter=[num_filter] * groups,
        conv_filter_size=3,
        conv_with_batchnorm=True,
        conv_batchnorm_drop_rate=dropouts,
        conv_act='relu',
        pool_size=2,
        pool_stride=2,
        pool_type='max')


def vggnet(image):
    # 输入图像: N*C*H*W=N*3*32*32, H/W=(H/W-F+2*P)/S+1, 网络层数: 16=13+1+1+1
    conv1 = conv_block(input=image, num_filter=64, groups=2, dropouts=[0.3, 0])  # 输出:N*64*16*16
    conv2 = conv_block(input=conv1, num_filter=128, groups=2, dropouts=[0.4, 0])  # 输出:N*128*8*8
    conv3 = conv_block(input=conv2, num_filter=256, groups=3, dropouts=[0.4, 0.4, 0])  # 输出:N*256*4*4
    conv4 = conv_block(input=conv3, num_filter=512, groups=3, dropouts=[0.4, 0.4, 0])  # 输出:N*512*2*2
    conv5 = conv_block(input=conv4, num_filter=512, groups=3, dropouts=[0.4, 0.4, 0])  # 输出:N*512*1*1

    drop1 = fluid.layers.dropout(x=conv5, dropout_prob=0.5)
    fc1 = fluid.layers.fc(input=drop1, size=512, act=None)  # 输出: N*512*1*1
    bn1 = fluid.layers.batch_norm(input=fc1, act='relu')
    drop2 = fluid.layers.dropout(x=bn1, dropout_prob=0.5)
    fc2 = fluid.layers.fc(input=drop2, size=512, act=None)  # 输出: N*512*1*1

    # 输出概率: N*L=N*10
    prediction = fluid.layers.fc(input=fc2, size=10, act='softmax')

    return prediction


# 预测网络
def infer_network():
    image = fluid.layers.data(name='image', shape=[None, 3, 32, 32], dtype='float32')
    prediction = vggnet(image)
    return prediction


# 训练网络
def train_network(prediction):
    label = fluid.layers.data(name='label', shape=[None, 1], dtype='int64')

    loss = fluid.layers.cross_entropy(input=prediction, label=label)
    avg_loss = fluid.layers.mean(loss)
    accuracy = fluid.layers.accuracy(input=prediction, label=label)

    return [avg_loss, accuracy]


# 测试模型
def test(executor, program, reader, feeder, fetch_list):
    avg_loss_set = []  # 平均损失值集
    accuracy_set = []  # 分类准确率集
    for test_data in reader():  # 将测试数据输出的每一个数据传入网络进行训练
        metrics = executor.run(
            program=program,
            feed=feeder.feed(test_data),
            fetch_list=fetch_list)
        avg_loss_set.append(float(metrics[0]))
        accuracy_set.append(float(metrics[1]))
    avg_loss_mean = numpy.array(avg_loss_set).mean()  # 计算平均损失值
    accuracy_mean = numpy.array(accuracy_set).mean()  # 计算平均准确率
    return avg_loss_mean, accuracy_mean  # 返回平均损失值和平均准确率


# 训练模型
def train():
    # 读取数据
    train_reader = paddle.batch(
        paddle.reader.shuffle(paddle.dataset.cifar.train10(), buf_size=50000),
        batch_size=batch_size)  # 读取训练数据
    test_reader = paddle.batch(
        paddle.dataset.cifar.test10(),
        batch_size=batch_size)  # 读取测试数据

    # 配置网络
    prediction = infer_network()  # 配置预测网络
    avg_loss, accuracy = train_network(prediction)  # 配置训练网络

    # 获取网络
    main_program = fluid.default_main_program()  # 获取默认主程序
    startup_program = fluid.default_startup_program()  # 获取默认启动程序
    test_program = main_program.clone(for_test=True)  # 克隆测试主程序

    # 优化方法
    optimizer = fluid.optimizer.Adam(learning_rate=0.001)  # Adam算法
    optimizer.minimize(avg_loss)  # 最小化平均损失值

    # 启动程序
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()  # 获取执行器设备
    train_exe = fluid.Executor(place)  # 获取训练执行器
    train_exe.run(startup_program)  # 运行启动程序
    test_exe = fluid.Executor(place)  # 获取测试执行器

    # 训练模型
    step = 0  # 周期计数器
    feed_order = ['image', 'label']
    feed_var_list_loop = [main_program.global_block().var(var_name) for var_name in feed_order]
    feeder = fluid.DataFeeder(feed_list=feed_var_list_loop, place=place)  # 获取数据读取器

    train_prompt = "Train loss"
    test_prompt = "Test loss"
    # loss_ploter = Ploter(train_prompt, test_prompt)  # 绘制损失值图

    for epoch in range(epoch_num):
        # 训练模型
        for train_data in train_reader():
            train_metrics = train_exe.run(
                program=main_program,
                feed=feeder.feed(train_data),
                fetch_list=[avg_loss, accuracy])

            if step % 100 == 0:
                # loss_ploter.append(train_prompt, step, train_metrics[0])
                # loss_ploter.plot()
                print("Pass %d, Epoch %d, avg_loss: %f" % (step, epoch, train_metrics[0]))
            step += 1

        # 测试模型
        test_metrics = test(
            executor=test_exe,
            program=test_program,
            reader=test_reader,
            feeder=feeder,
            fetch_list=[avg_loss, accuracy])

        # loss_ploter.append(test_prompt, step, test_metrics[0])
        # loss_ploter.plot()
        print("Test with Epoch %d, avg_loss: %f, accuracy: %f" % (epoch, test_metrics[0], test_metrics[1]))

        # 保存模型
        if save_dirname is not None:
            fluid.io.save_inference_model(save_dirname, ['image'], [prediction], train_exe)

        if test_metrics[0] < 0.4:  # 如果平均损失值达到要求，停止训练
            break


# 加载图像
def load_image(file):
    im = Image.open(file)  # 打开图像文件
    im = im.resize((32, 32), Image.ANTIALIAS)  # 调整图像大小
    im = numpy.array(im).astype(numpy.float32)  # 转换数据类型
    im = im.transpose((2, 0, 1))  # WHC转为CWH
    im = im / 255.0  # 归一化处理(0,1)
    im = numpy.expand_dims(im, axis=0)  # 增加数据维度
    return im


# 预测图像
def infer():
    # 加载图像
    image = load_image(file_dirname)

    # 预测图像
    place = fluid.CUDAPlace(0)  # 获取GPU设备
    infer_exe = fluid.Executor(place)  # 获取预测执行器
    inference_scope = fluid.core.Scope()  # 获取预测作用域

    with fluid.scope_guard(inference_scope):
        # 加载模型
        [inference_program, feed_target_names, fetch_targets
         ] = fluid.io.load_inference_model(save_dirname, infer_exe)

        # 预测图像
        results = infer_exe.run(
            program=inference_program,
            feed={feed_target_names[0]: image},
            fetch_list=fetch_targets)

        # 显示结果
        infer_label = ["ariplane", "automobile", "bird", "cat", "deer",
                       "dog", "frog", "horse", "ship", "truck"]  # 预测图像标签
        print("Infer image: %s" % infer_label[numpy.argmax(results[0])])
        infer_image = Image.open(file_dirname)  # 打开预测图像
        plt.imshow(infer_image)  # 显示预测图像


# 主函数
def main():
    paddle.enable_static()
    # 训练模型
    train()

    # 预测图像
    infer()


# 主函数
if __name__ == '__main__':
    main()
