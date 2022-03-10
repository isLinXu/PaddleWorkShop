import numpy as np


def copy_array(a):

    return h


if __name__ == '__main__':

    # 复制数组
    a = np.array([1, 2, 3])
    h = a.view()
    np.copy(a)
    h = a.copy()

    # 数组创建
    a = np.array([1, 2, 3])  # 创建数组
    b = np.array([(1.5, 2, 3), (4, 5, 6)],dtype=float)
    c = np.array([[(1.5, 2, 3), (4, 5, 6)],
                 [(3, 2, 1), (4, 5, 6)]],dtype = float)

    np.zeros((3, 4)) # 创建0数组
    np.ones((2, 3, 4), dtype=np.int16)  # 创建1数组
    # d = np.arrange(10, 25, 5)  # 创建相同步数数组
    np.linspace(0, 2, 9)  # 创建等差数组

    e = np.full((2, 2), 7)  # 创建常数数组
    f = np.eye(2)  # 创建2x2矩阵
    np.random.random((2, 2))  # 创建随机数组
    np.empty((3, 2))  # 创建空数组

    # 打印数组
    my_array = np.array([1, 2, 3])

    print(my_array)  # 打印数组
    # saving &Loading on disk保存到磁盘
    np.save('my_array', a)
    np.savez('array.npz', a, b)
    np.load('my_array.npy')

    # saving &Loading Text files保存到文件
    # np.loadtxt("my file.txt")
    # np.genfromtxt("my_file.csv", delimiter=',')
    # np.savetxt("marry.txt", a, delimiter=" ")

    # arithmetic operation算术运算
    g = a - b
    np.subtract(a, b)  # 减法
    b + a
    np.add(b, a)  # 加法
    a / b
    np.divide(a, b)  # 除法
    a * b
    # np.multiple(a, b)  # 乘法
    np.exp(b)  # 指数
    np.sqrt(b)  # 开方
    np.sin(a)  # sin函数
    np.cos(b)  # cos函数
    np.log(a)  # log函数
    e.dot(f)  # 内积

    # Comparison比较
    # a == b  # 元素
    # a < 2  # 元素
    np.array_equal(a, b)  # 数组

    # Aggregate Functions 函数
    a.sum()  # 求和
    b.min()  # 最小值
    b.max(axis=0)  # 最大值数组列
    b.cumsum(axis=1)  # 元素累加和
    a.mean()  # 平均值
    # b.median()  # 中位数
    # a.corrcoef()  # 相关系数
    # np.std(b)  # 标准差

    import numpy as np

    # array处理
    # Transposing Array
    I = np.transpose(b)  # 转置矩阵
    # i.T  # 转置矩阵

    # Changing Array Shape
    b.ravel()  # 降为一维数组
    g.reshape(3, -2)  # 重组

    # Adding/Removing Elements
    h.resize((2, 6))  # 返回shape(2,6)
    np.append(h, g)  # 添加
    np.insert(a, 1, 5)  # 插入
    np.delete(a, [1])  # 删除

    # Combining Arrays
    # np.concatenate((a, d), axis=0)  # 连结
    np.vstack((a, b))  # 垂直堆叠
    # np.r_[e, f]  # 垂直堆叠
    np.hstack((e, f))  # 水平堆叠
    # np.column_stack((a, d))  # 创建水平堆叠
    # np.c_[a, d]  ##创建水平堆叠

    # splitting arrays
    np.hsplit(a, 3)  # 水平分离
    np.vsplit(c, 2)  # 垂直分离


    # 矩阵索引操作
    # subsetting
    # a[2]  # 选取数组第三个元素
    # b[1, 2]  # 选取2行3列元素
    #
    # # slicing
    # a[0:2]  # 选1到3元素
    # b[0:2, 1]  # 选1到2行的2列元素
    # b[:1]  # 选所有1行的元素
    # c[1, ...]  # c[1,:,:]
    # a[::-1]  # 反转数组
    #
    # # Boolean Indexing
    # a[a < 2]  # 选取数组中元素<2的
    #
    # # Fancy Indexing
    # b[[1, 0, 1, 0], [0, 1, 2, 0]]
    # # 选取[1,0],[0,1],[1,2],[0,0]
    # b[[1, 0, 1, 0][:, [0, 1, 2, 0]]]
    # # 选取矩阵的一部分

    # 数据类型
    # np.int64  # 64位整数
    # np.float32  # 标准双精度浮点
    # np.complex  # 复杂树已浮点128为代表
    # np.bool  # true&false
    # np.object  # python object
    # np.string_  # 固定长度字符串
    # np.unicode_  # 固定长度统一码

    # a.shape  # 数组维度
    # len(a)  # 数组长度
    # b.ndim  # 数组维度数量
    # e.size  # 数组元素数量
    # b.dtype  # 元素数据类型
    # b.dtype.name  # 数据类型名
    # b.astype(int)  # 改变数组类型
    #
    # # asking for help更多信息
    # np.info(np.ndarray.dtype)