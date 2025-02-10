from matplotlib import pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import torch.nn.functional as Fun
import torch


# 线性模型的定义
class Linear_Regression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Linear_Regression, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim, bias=True)
    def forward(self, x):
        return self.fc1(x)

# 损失函数
def loss(y_prediction, y_train):
    return Fun.mse_loss(y_prediction.view(-1), y_train)

# 优化器
def optimizer(Model):
    return torch.optim.Adam(Model.parameters(), lr=0.01)


def fit(Model, X_data, Y_data, epoch):
    for epoch in range(epoch+1):
        Model.train()
        y_pre = Model(X_data)
        optimizer(Model).zero_grad()  # 梯度清零
        loss(y_pre.view(-1), Y_data).backward()  # 反向传播
        optimizer(Model).step()  # 梯度更新
        print("epoch:", epoch, "loss:", loss(y_pre.view(-1), Y_data).item())


if __name__ == '__main__':
    ## 数据读取
    data = load_diabetes()
    X = data.data
    Y = data.target
    feature_names = data.feature_names
    ## 数据标准化
    Scaler = StandardScaler()
    X = Scaler.fit_transform(X)

    ## 数据分割
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    Y_train = torch.tensor(Y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    Y_test = torch.tensor(Y_test, dtype=torch.float32)
    model = Linear_Regression(X_train.shape[1], 1)
    fit(model, X_train, Y_train, epoch=4000)

    # 评估模型
    with torch.no_grad():
        y_test_pred = model(X_test)
        print("mse:",loss(y_test_pred.view(-1), Y_test))
        print("r2:", r2_score(Y_test, y_test_pred.view(-1)))


    # 可视化结果
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(Y_test)), Y_test, c='black', label='True Data')
    plt.plot(range(len(Y_test)), y_test_pred, c='red', label='Predict')
    plt.legend()
    plt.show()

    # 可视化结果
    for i in range(len(feature_names)):
        plt.scatter(X_test[:, i].numpy(), Y_test.numpy(), color='black', label='True_value')
        plt.scatter(X_test[:, i].numpy(), y_test_pred.numpy(), color='blue', label='Prediction')
        plt.xlabel(f'{feature_names[i]}')
        plt.ylabel('Target')
        plt.title(f'Feature:{feature_names[i]}')
        plt.legend()
        plt.show()
