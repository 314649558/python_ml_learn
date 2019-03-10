from sklearn.svm import LinearSVC
from sklearn.datasets import make_blobs
from mglearn.tools import discrete_scatter
from mglearn.plots import plot_2d_separator
import matplotlib.pyplot as plt
import numpy as np
from mglearn import cm2
from mpl_toolkits.mplot3d import Axes3D,axes3d

X,y=make_blobs(centers=4,random_state=8)
y=y%2
linear_svm=LinearSVC().fit(X,y)
plot_2d_separator(linear_svm,X)
discrete_scatter(X[:,0],X[:,1],y)
plt.xlabel("Feature 0")
plt.xlabel("Feature 1")
#plt.show()

#添加第二个特征的平方，作为一个新特征
X_new = np.hstack([X,X[:,1:]**2])

figure=plt.figure()
#3D可视化
ax=Axes3D(figure,elev=-152,azim=-26)

# 首先画出所有 y==0的点，然后画出所有y==1的点
mask = y ==0

ax.scatter(X_new[mask,0],X_new[mask,1],X_new[mask,2],c='b',cmap=cm2,s=60)
ax.scatter(X_new[~mask,0],X_new[~mask,1],X_new[~mask,2],c='r',marker='^',cmap=cm2,s=60)

ax.set_xlabel("feature 0")
ax.set_ylabel("feature 1")
ax.set_zlabel("feature ** 2")

plt.show()
