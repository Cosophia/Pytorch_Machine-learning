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

print("\n## 2.张量的计算:")
## 2.张量的计算
# 张量加法
x = torch.rand(2, 3)
y = torch.rand(2, 3)
z = x + y
print("张量x和y的加法结果:", z)

# 张量乘法
z = x * y
print("张量x和y的乘法结果:", z)

# 张量减法
z = x - y
print("张量x和y的减法结果:", z)

# 张量除法
z = x / y
print("张量x和y的除法结果:", z)

# 全局均值
print("张量x的全局均值:", x.mean())
print("dim=0,按列计算x的均值:", torch.tensor([[1,2,3],[1,2,3]],dtype=torch.float32).mean(dim=0)) # 可指定维度
print("dim=1,按行计算x的均值:", torch.tensor([[1,2,3],[1,2,3]],dtype=torch.float32).mean(dim=1)) # 可指定维度
print("dim=-1,按维数对最后一维计算x的均值:", torch.tensor([[1,2,3],[1,2,3]],dtype=torch.float32).mean(dim=-1)) # 可指定维度

# 计算总和
print("全局总和:", x.sum())
print("按列求和:", x.sum(dim=0))
print("按行求和:", x.sum(dim=1))

# 计算平方
print("平方：", x.pow(2))


# 点乘(Dot Product) & 矩阵乘法(matrix multiplication)
# 创建向量
u = torch.tensor([[1, 2, 3], [4, 5, 6]])
v = torch.tensor([[4, 5, 6], [7, 8, 9],[10,11,12]])
w = torch.tensor([1, 2, 3])
z = torch.tensor([4, 5, 6])
print("矩阵乘法(matmul):", torch.matmul(u, v))
print("点乘(matmul):", torch.matmul(w, z))
print("点乘(*后求和_:", (w * z).sum())
print("矩阵乘法(matmul):",u @ z)



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
# 张量重塑
print("The way to reshape the tensor,what we use in Pytorch is 'view',result:",x.view(3,2))
# 张量转置
x_transposed = x.t()
print("展平且转置后的 x:", x_transposed)

print("\n## 6.张量的形状的复制:")
## 6.张量的形状的复制
# If you want to create a tensor like a specific shape, you can use the torch.randn_like() function.
d = torch.rand(5, 5)
y = torch.randn_like(d)
print("The shape of the y :", y.shape)


## 7.张量的索引
print("\n## 7.张量的索引:")
x = torch.rand(5,5)
print(x)
print("索引单个元素:", x[1,2])  # 索引单个元素
print("索引单行:", x[1]) # 索引单行
print("索引单列:", x[:, 1]) # 索引单列
print("切片多行：", x[1:3]) # 切片多行
print("切片多列：", x[:, 1:3]) # 切片多列
print("切片多行多列：", x[1:3, 1:3]) # 切片多行多列
print("布尔索引:", x[x > 0.5]) # 布尔索引
print("高级索引:", x[[1, 2, 3], [1, 2, 3]]) # 高级索引 (第1,2,3行的第1,2,3列的数据)

## 8.利用索引改变具体的数据
print("\n## 8.利用索引改变具体的数据:")
# 特定的元素修改
x[1, 2] = 10
print("修改后的 x:", x)
# 使用切片修改
x[:, 1] = 0
print("修改后的 x:", x)
# 使用布尔索引修改
mask = x > 0.5
x[mask] = -1
print("修改后的 x:", x)
# 使用广播修改
x[:, 1:] = 10
print("广播修改后:", x)
# 使用视图创建视图
y = x[1:, 1:]
print("视图:", y)
y[0, 0] = 10
print("修改视图后:", x)  # 修改视图会影响原张量
# 使用索引创建副本
y = x[1:, 1:].clone() # 利用clone()创建副本
print("副本:", y)
y[0, 0] = 10
print("修改副本后:", x) # 修改副本不会影响原张量


print("\n## 9.神经网络的创建:")
## 9.神经网络的创建
class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        print("fc1's shape of weights:", self.fc1.weight.size())
        ## What i can find here is that the weights are initialized randomly by starting the program in sevals time
        print("fc1's weights:", self.fc1.weight)
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x):
        """前向传播"""
        x = Fun.relu(self.fc1(x))
        print("After activating by relu x =", x)
        x = Fun.softmax(self.fc2(x), dim=-1)  # dim=-1 means the last dimension
        return max(x)  # 取最大概率


if __name__ == "__main__":
    net = Net()
    x_data = torch.tensor(list(range(10)), dtype=torch.float32)
    print("\n简单的神经网络的创建:", net)
    print("x_data经过隐藏层的输出为:", net(x_data))
