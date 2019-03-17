from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mglearn import cm2

# 构造数据
X,_ = make_blobs(n_samples=50,centers=5,random_state=4,cluster_std=2)
# 将其分为训练集和测试集
X_train,X_test=train_test_split(X,random_state=1,test_size=0.1)

# 绘制训练集和测试集
fig,axes = plt.subplots(1,1,figsize=(13,4))

print(X_train[:,1])

axes[0].scatter(X_train[:, 0], X_train[:, 1], c=cm2(0), label="Training set", s=60)
axes[0].scatter(X_test[:,0],X_test[:,1],marker='^',c=cm2(1),label="Test set",s=60)
axes[0].legend(loc='upper left')
axes[0].set_title("Original Data")


for ax in axes:
    ax.set_xlabel("Feature 0")
    ax.set_ylabel("Feature 1")

#plt.show()