from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from mglearn.datasets import *
X,y=make_wave(n_samples=60)

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42)


#print(y_test)


print("-----------------------------------------------------------------")

lr = LinearRegression().fit(X_train,y_train)

#斜率
print("lr.coef_:{}".format(lr.coef_))
# 截距
print("lr.intercept_:{}".format(lr.intercept_))


print("Training set score: {:.2f}".format(lr.score(X_train,y_train)))
print("Test set score: {:.2f}".format(lr.score(X_test,y_test)))


