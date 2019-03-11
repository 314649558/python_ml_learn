from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

import matplotlib.pyplot as plt


cancer=load_breast_cancer()
X_train,X_test,y_train,y_test=train_test_split(cancer.data,cancer.target,random_state=0)

mlp=MLPClassifier(random_state=42)
mlp.fit(X_train,y_train)
print("Accuracy on training set:{:.3f}".format(mlp.score(X_train,y_train)))
print("Accuracy on test set:{:.3f}".format(mlp.score(X_test,y_test)))

#计算训练集中每个特征的平均值
mean_on_train=X_train.mean(axis=0)

# 计算训练集中每个特征的标准差
std_on_train=X_train.std(axis=0)

# 减去平均值，然后乘以标准差的倒数
X_train_scaled=(X_train-mean_on_train)/std_on_train

X_test_scaled=(X_test-mean_on_train)/std_on_train

mlp=MLPClassifier(max_iter=1000,random_state=0,alpha=1)

mlp.fit(X_train_scaled,y_train)


print("Accuracy on training set:{:.3f}".format(mlp.score(X_train_scaled,y_train)))
print("Accuracy on test set:{:.3f}".format(mlp.score(X_test_scaled,y_test)))



plt.figure(figsize=(20,5))
plt.imshow(mlp.coefs_[0],interpolation='none',cmap='viridis')
plt.yticks(range(30),cancer.feature_names)
plt.xlabel("Columns in weight matrix")
plt.ylabel("Input feature")
plt.colorbar()
plt.show()