
import paddle

# 里程数据
data_x = [3, 12, 4, 7, 9]
# 打车费用
data_y = [19, 40, 21, 27, 32]


net = paddle.nn.Linear(in_features=1, out_features=1)

# 定义损失函数
loss_func = paddle.nn.MSELoss()

# 定义优化器
opt = paddle.optimizer.SGD(parameters=net.parameters())

# 组建训练程序
for epoch in range(10):
    for x, y in zip(data_x, data_y):
        x = paddle.to_tensor([x], dtype="float32")
        y = paddle.to_tensor([y], dtype="float32")
        # 预测结果
        infer_y = net(x)
        # 计算Loss
        loss = loss_func(infer_y, y)
        loss.backward()
        opt.step()
        opt.clear_gradients()
        print(f"Epoch:{epoch}\t loss:{loss.numpy()}")


x = paddle.to_tensor([8], dtype="float32")
infer_y = net(x)
print(f"8km对应的打车价格为：{infer_y.numpy()}元。")


