import torch
import torch.nn as nn
import torch.nn.functional as Fun
print("## 1.张量的创建:")
## 1.张量的创建
# 创建一个未初始化的张量
x = torch.empty(3, 3)  # 2 行 3 列
print("未初始化张量:", x)

# 创建一个随机初始化的张量
x = torch.rand(3, 3)
print("随机初始化的张量:", x)

# 创建一个全零张量
x = torch.zeros(3, 3, dtype=torch.float32)
print("全零张量:", x)

# 创建一个全一张量
x = torch.ones(3, 3)
print("全一张量:", x)

print("\n## 2.张量的操作:")
## 2.张量的操作
# 张量加法
x = torch.rand(2, 3)
y = torch.rand(2, 3)
z = x + y
print("张量x和y的加法结果:", z)

# 张量乘法
z = x * y
print("张量x和y的乘法结果:", z)

# 张量切片
print("张量x和y的切片", x[:, 1])  # 获取第 2 列

print("\n## 3.自动求导:")
## 3.自动求导
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 2
# Tips: The calculation of the gradient is performed in the backward method.
# What's more,what it actually do is to calulate the derivatives of the function(here is y=x^2).
# If there are more than one variable, the gradient will be calculated for all of them by calculating the
# partial derivatives of the function.
# Tips: Reason we use the sum function:Assume that the Y = [y1,y2,y3],and we can know above that y = x^2
# So z = y1+y2+y3=x1^2+x2^2+x3^2,then here we do : dz/dx= (dz/dy * dy/dx) = 1*2x
# Finally,the gradient of X is 2x,[2.,4.,6.].
z = y.sum()
z.backward()  # 计算梯度:
print("x 的梯度:", x.grad)  # 输出 x 的梯度

## 4.张量数据类型的转换
print("\n## 4.张量数据类型的转换:")
x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
x = x.to(torch.float64)
print("x 的数据类型:", x.dtype)
x = x.to(torch.int32)
print("x 的数据类型:", x.dtype)
x = x.to(torch.int64)
print("x 的数据类型:", x.dtype)
x = x.to(torch.float32)

## 5.张量的形状转换
x = torch.tensor([[1, 2, 3], [4, 5, 6]])
print("x 的形状:", x.size())
# 将其数据展平
x_flatten = x.view(-1)
print("展平后 x 的:", x_flatten)
# 张量转置
x_transposed = x.t()
print("展平且转置后的 x:", x_transposed)

print("\n## 6.张量的形状的复制:")
## 6.张量的形状的复制
# If you want to create a tensor like a specific shape, you can use the torch.randn_like() function.
d = torch.rand(5,5)
y = torch.randn_like(d)
print("The shape of the y :", y.shape)

print("\n## 7.神经网络的创建:")
## 7.神经网络的创建
class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        print("fc1's shape of weights:", self.fc1.weight.size())
        ## What i can find here is that the weights are initialized randomly by starting the program in sevals time
        print("fc1's weights:", self.fc1.weight)
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x):
        x = Fun.relu(self.fc1(x))
        print("After activating by relu x =", x)
        x = Fun.softmax(self.fc2(x), dim=-1)  # dim=-1 means the last dimension
        return max(x) # 取最大概率


net = Net()
x_data = torch.tensor(list(range(10)), dtype=torch.float32)
print("\n简单的神经网络的创建:", net)
print("x_data经过隐藏层的输出为:", net(x_data))
