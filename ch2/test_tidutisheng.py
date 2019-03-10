from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

cancer=load_breast_cancer()

X_train,X_test,y_train,y_test=train_test_split(cancer.data,cancer.target,random_state=0)


print(X_train)
print("--------------------------------------")
print("--------------------------------------")
print("--------------------------------------")
print(y_train)


gbrt = GradientBoostingClassifier(random_state=0)
gbrt.fit(X_train,y_train)
print("--------------------------------------")
print("-----------result-------------------")
print("--------------------------------------")
print("Accuary on traing set :{:0.3f}".format(gbrt.score(X_train,y_train)))
print("Accuary on test set :{:0.3f}".format(gbrt.score(X_test,y_test)))

#上面的结果由于训练集的结果达到100% 很有可能存在过拟合 ，因此我们可以通过降低深度来预剪枝
gbrt2=GradientBoostingClassifier(random_state=0,max_depth=1)
gbrt2.fit(X_train,y_train)
print("max_depth=1-----------------------------")
print("Accuary on traing set :{:0.3f}".format(gbrt2.score(X_train,y_train)))
print("Accuary on test set :{:0.3f}".format(gbrt2.score(X_test,y_test)))

#也可通过降低学习率来矫正过拟合
gbrt3=GradientBoostingClassifier(n_estimators=1000,random_state=0,learning_rate=0.01)
gbrt3.fit(X_train,y_train)
print("learning_rate=1-----------------------------")
print("Accuary on traing set :{:0.3f}".format(gbrt3.score(X_train,y_train)))
print("Accuary on test set :{:0.3f}".format(gbrt3.score(X_test,y_test)))


