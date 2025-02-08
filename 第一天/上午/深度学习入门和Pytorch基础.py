import torch
import torch.nn as nn
import torch.nn.functional as Fun

## 1.张量的创建
# 创建一个未初始化的张量
x = torch.empty(3, 3)  # 2 行 3 列
print("未初始化张量:",x)

# 创建一个随机初始化的张量
x = torch.rand(3, 3)
print("随机初始化的张量:",x)

# 创建一个全零张量
x = torch.zeros(3, 3, dtype=torch.float32)
print("全零张量:",x)

# 创建一个全一张量
x = torch.ones(3, 3)
print("全一张量:",x)

## 2.张量的操作
# 张量加法
x = torch.rand(2, 3)
y = torch.rand(2, 3)
z = x + y
print("张量x和y的加法结果:",z)

# 张量乘法
z = x * y
print("张量x和y的乘法结果:",z)

# 张量切片
print("张量x和y的切片",x[:, 1])  # 获取第 2 列

## 3.自动求导
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 2
z = y.sum()
z.backward()  # 计算梯度
print("x 的梯度:",x.grad)  # 输出 x 的梯度

## 4.张量数据类型的转换
x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
x = x.to(torch.float64)
print("x 的数据类型:",x.dtype)
x = x.to(torch.int32)
print("x 的数据类型:",x.dtype)
x = x.to(torch.int64)
print("x 的数据类型:",x.dtype)
x = x.to(torch.float32)

## 5.张量的形状转换
x = torch.tensor([[1, 2, 3], [4, 5, 6]])
print("x 的形状:",x.size())
# 将其数据展平
x_flatten = x.view(-1)
print("展平后 x 的:",x_flatten)
# 张量转置
x_transposed  = x.t()
print("展平且转置后的 x:",x_transposed)


## 6.神经网络的创建
class Net(nn.Module):
    def __init__(self,):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(10,5)
        print("fc1's shape of weights:",self.fc1.weight.size())
        print("fc1's weights:",self.fc1.weight)
        self.fc2 = nn.Linear(5,2)
    def forward(self, x):
        x =Fun.relu(self.fc1(x))
        x = self.fc2(x)
        return x

net=Net()
x=torch.tensor(list(range(10)),dtype=torch.float32)
print("简单的神经网络的创建:",net(x))
