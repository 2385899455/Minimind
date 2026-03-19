import torch
import torch.nn as nn

# dropout
# 按照1-p的概率保留, 被保留的元素除以 (1-p) 期望不变
dropout_layer = nn.Dropout(p=0.5)

t1 = torch.tensor([1.0, 2.0, 3.0])
t2 = dropout_layer(t1)
print(f"dropout后", t2)

# linear
# 线性变换 就是对应用的张量乘以一个W矩阵然后+b
# in_features: 输入特征的维度
# out_features: 输出特征的维度
# bias: y = xW^T + b中的b是否存在
layer = nn.Linear(in_features=3, out_features=5, bias=True)
t1 = torch.tensor([1.0, 2.0, 3.0])
t2 = torch.tensor([[1.0, 2.0, 3.0]])
output2 = layer(t2)
print(f"维度变换后", output2)

# view
# 改变张量形状
t = torch.tensor([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]])
# 将t变成3行4列的张量, 但是元素个数不变
t_view1 = t.view(3, 4)
print(f"t_view1:", t_view1)
t_view2 = t.view(4, 3)
print(f"t_view2:", t_view2)

# transpose
# 交换维度
t1 = torch.tensor([[1, 2, 3], [4, 5, 6]])
# 维度0和维度1交换
t_transpose = t1.transpose(0, 1)
print(f"t_transpose:", t_transpose)

# triu
# 上三角矩阵
x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"上三角矩阵:", x.triu())
# 移动对角线
print(f"上三角矩阵（k=1）:", x.triu(diagonal=1))

# reshape
# 改变张量形状 类似于view 可以处理不连续的内存
x = torch.arange(1, 7)
y = torch.reshape(x, (2, 3))
print(f"reshape后的张量:", y)
# 使用 -1 自动推断维度
z = torch.reshape(x, (3, -1))
print(f"使用-1自动推断后的张量:", z)