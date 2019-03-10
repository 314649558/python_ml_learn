from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from mglearn.datasets import *
X,y=make_wave(n_samples=60)

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42)




ridge=Ridge().fit(X_train,y_train)
print("Training set score:{:0.2f}".format(ridge.score(X_train,y_train)))
print("Test set score:{:0.2f}".format(ridge.score(X_test,y_test)))

#加大alpha参数会使系数更加接近0，从而降低训练集性能，但可能会提高泛化性能
ridge2=Ridge(alpha=10).fit(X_train,y_train)
print("Training set score:{:0.2f}".format(ridge2.score(X_train,y_train)))
print("Test set score:{:0.2f}".format(ridge2.score(X_test,y_test)))



