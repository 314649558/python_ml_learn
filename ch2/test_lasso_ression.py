from sklearn.linear_model import Lasso
import numpy as np
from sklearn.model_selection import train_test_split
from mglearn.datasets import *
X,y=make_wave(n_samples=60)

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42)


lasso=Lasso(alpha=0.1,max_iter=100000).fit(X_train,y_train)
print("Training set score:{:.2f}".format(lasso.score(X_train,y_train)))
print("Test set score:{:.2f}".format(lasso.score(X_test,y_test)))
print("Number of features used:{}".format(np.sum(lasso.coef_ != 0)))


