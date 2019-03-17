from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
cancer=load_breast_cancer()
X_train,X_test,y_train,y_test=train_test_split(cancer.data,cancer.target,random_state=0)

svm=SVC(C=100)

svm.fit(X_train,y_train)
print("Test set accurayc: {:.2f}".format(svm.score(X_test,y_test)))


# 使用0-1缩放进行预处理
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
# 在缩放后的训练数据上学习SVM
svm.fit(X_train_scaled,y_train)
print("Test set accurayc: {:.2f}".format(svm.score(X_test_scaled,y_test)))


# 利用零均值和单位方差的缩放方法进行预处理
standardScaler = StandardScaler()
standardScaler.fit(X_train)
X_train_scaled2=scaler.transform(X_train)
X_test_scaled2=scaler.transform(X_test)

svm.fit(X_train_scaled2,y_train)
print("Test set accurayc: {:.2f}".format(svm.score(X_test_scaled2,y_test)))

