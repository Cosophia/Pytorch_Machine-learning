from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
import torch.nn as nn
import torch.nn.functional as Fun
import torch
import pandas as pd
import seaborn as sns

def loss(y_prediction, y_train):
    return Fun.binary_cross_entropy(y_prediction.view(-1), y_train) # 二交叉熵损失函数

# 优化器
def optimizer(Model):
    return torch.optim.RAdam(Model.parameters(), lr=0.01,betas=(0.9, 0.999), eps=1e-8)


def fit(Model, X_data, Y_data, epoch):
    for epoch in range(epoch+1):
        optimizer(model).zero_grad() # 梯度清零
        y_pre=Model(X_data) # 前向传播
        loss(y_pre.view(-1),Y_data).backward() #反向传播
        optimizer(Model).step() # 梯度更新
        print("epoch:", epoch, "loss:", loss(y_pre.view(-1), Y_data).item())

# 模型建立
class Logistic_Regression(nn.Module):
    def __init__(self, input_dim):
        super(Logistic_Regression, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1,bias=True)

    def forward(self, x):
        x = Fun.sigmoid(self.fc1(x))
        return x


if __name__ == '__main__':
    # 数据预处理
    data = pd.read_csv("./spam.csv")
    vectorizer = CountVectorizer() # 创建向量化器
    X = vectorizer.fit_transform(data['Message']) # 将数据集转化为矩阵
    Y = data['Category']
    Y=Y.map({"ham":0,"spam":1}) # 为数据集添加标签
    # X_shape = X.toarray().shape
    # print(X_shape)
    X = torch.tensor(X.toarray(),dtype=torch.float32)
    Y = torch.tensor(Y.values,dtype=torch.float32)


    # 数据集划分
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

    # 数据归一化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 数据转换成tensor
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    Y_train = torch.tensor(Y_train, dtype=torch.float32)
    Y_test = torch.tensor(Y_test, dtype=torch.float32)

    # 模型建立
    model = Logistic_Regression(X_train.shape[1])

    # 模型训练
    fit(model, X_train, Y_train, 10000)

    # 模型评估
    with torch.no_grad():
        y_test_pred = model(X_test)
        y_test_pred_cls = (y_test_pred > 0.5).float()

    accuracy=accuracy_score(Y_test.numpy(), y_test_pred_cls) # 准确率
    conf_matrix = confusion_matrix(Y_test.numpy(), y_test_pred_cls) # 混淆矩阵
    class_report = classification_report(Y_test.numpy(), y_test_pred_cls) # 分类报告
    print("accuracy:",accuracy*100,"%")
    print("conf_matrix:",conf_matrix)
    print("class_report:",class_report)

    # 混淆矩阵可视化
    sns.heatmap(conf_matrix, annot=True, fmt="d",cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

