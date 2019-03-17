from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
cancer=load_breast_cancer()
X_train,X_test,y_train,y_test=train_test_split(cancer.data,cancer.target,random_state=1)

#print(y_train)
#print(X_test)
#print(y_train)
#print(y_test)

print(X_train.shape)
print(X_test.shape)

scaler = MinMaxScaler()
print(scaler.fit(X_train))


# 对数据进行缩放

#变换数据
X_train_scaled = scaler.transform(X_train)
# 在缩放前后分别打印数据集属性
print("transformed shape:\n{}".format(X_train_scaled.shape))
print("per-feature minimum before scaling:\n{}".format(X_train.min(axis=0)))
print("per-feature maximum before scaling:\n{}".format(X_train.max(axis=0)))
print("per-feature minimum after scaling:\n{}".format(X_train_scaled.min(axis=0)))
print("per-feature maximum after scaling:\n{}".format(X_train_scaled.max(axis=0)))

#对测试数据进行缩放
X_test_scaled=scaler.transform(X_test)
#在缩放之后打印测试数据属性
print("per-feature minimum after scaling:\n{}".format(X_test_scaled.min(axis=0)))
print("per-feature maximum after scaling:\n{}".format(X_test_scaled.max(axis=0)))