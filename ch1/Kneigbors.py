from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np

iris_dataset=load_iris()
X_train,X_test,y_train,y_test=train_test_split(iris_dataset['data'],iris_dataset['target'],random_state=0)

knn=KNeighborsClassifier(n_neighbors=1)
r=knn.fit(X_train,y_train)   #基于训练集构建模型
print(r)


print("\n--------------------------------------")

#做预测
#构造新的数据
X_new=np.array([[5,2.9,1,0.2]])
print("X_new.shape:{}".format(X_new.shape))

prediction=knn.predict(X_new)  # 做预测
print("Prediction:{}".format(prediction))
print("Predicted target name:{}".format(iris_dataset['target_names'][prediction]))


#评估模型
#用来检测我们的做出的预测是否正确
#一般会用到测试集，而这些测试集是之前没有做过模型训练的

y_pred=knn.predict(X_test)
print("Test set predictions:\n{}".format(y_pred))

print("Test set score:{:.2f}".format(np.mean(y_pred==y_test)))


print("Test set score:{:.2f}".format(knn.score(X_test,y_test)))
