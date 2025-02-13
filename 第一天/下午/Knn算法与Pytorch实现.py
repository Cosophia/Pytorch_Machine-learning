from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
import torch


# KNN算法与Pytorch实现
def knn_pytorch(X_train, Y_train, X_test, k):
    # 计算距离
    distance=torch.cdist(X_test,X_train)
    # 取前k个最接近的索引
    _,index=torch.topk(distance,k,largest=False, dim=1)
    # 根据索引取标签
    close_labels=Y_train[index]
    # 统计标签
    prediction=torch.mode(close_labels,dim=1)[0]
    return prediction


if __name__ == '__main__':
    # 数据预处理
    data=load_iris()
    X=data.data
    Y=data.target
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    Y_train = torch.tensor(Y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    Y_test = torch.tensor(Y_test, dtype=torch.long)

    # 模型训练
    predictions = knn_pytorch(X_train, Y_train, X_test,  6)
    knn=KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, Y_train)
    predictions_2=knn.predict(X_test)
    print("meoth Accuracy:", accuracy_score(Y_test, predictions))
    print("api Accuracy:", accuracy_score(Y_test, predictions_2))

