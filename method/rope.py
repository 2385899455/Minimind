import torch

x = torch.tensor([1,2,3,4,5])
y = torch.tensor([10,20,30,40,50])

# 判断 x是否大于3 把结果赋值给condition
condition = x > 3

# 筛选过滤 条件为condition, x张量中符合条件的保留, 不符合的由y对应位置填充
result = torch.where(condition, x, y)

print(f"条件过滤后 =", result)

# 生成一个等差序列 0开始到10 间隔为2 [0, 2, 4, 6, 8]
t = torch.arange(0, 10, 2)
print(f"t =",t)
t2 = torch.arange(5, 0, -1)
print(f"t2 =",t2)

# 外积
# v1 * v2^T
v1 = torch.tensor([1, 2, 3])
v2 = torch.tensor([4, 5, 6])
result = torch.outer(v1, v2)
print(f"外积 =",result)

# 拼接
# 三维张量 a1[0] = [[1,2,3],[4,5,6]], a1[0,0] = [1, 2, 3]
a1 = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[13, 14, 15], [16, 17, 18]]])
a2 = torch.tensor([[[7, 8, 9], [10, 11, 12]], [[19, 20, 21], [22, 23, 24]]])
print(f"a1形状为",a1.shape)
result = torch.cat((a1, a2), dim=0)
print(f"拼接维度0", result)
result2 = torch.cat((a1, a2), dim=1)
print(f"拼接维度1", result2)
result3 = torch.cat((a1, a2), dim=-1)
print(f"最后一维拼接", result3)

# 增加维度
b1 = torch.tensor([1, 2, 3])
# 自适应 在dim维度上在增加一个维度
b2 = b1.unsqueeze(0)
print(f"b1维度", b1.shape)
print(f"b2", b2)
print(f"b2维度", b2.shape)

