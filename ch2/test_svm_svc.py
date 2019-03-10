from sklearn.svm import SVC
from mglearn.tools import make_handcrafted_dataset
from mglearn.plots import plot_2d_separator
from mglearn import discrete_scatter
import matplotlib.pyplot as plt


X,y=make_handcrafted_dataset()


#gamma参数用于控制高斯核的宽度
#C是正则化参数，限制 每个点的重要性
svm=SVC(kernel='rbf',C=10,gamma=0.1).fit(X,y)

plot_2d_separator(svm,X,eps=0.5)
discrete_scatter(X[:,0],X[:,1],y)
#画出支持向量
sv=svm.support_vectors_
# 支持向量的类别标签有dual_coef_的正负号给出
sv_labels=svm.dual_coef_.ravel() > 0
discrete_scatter(sv[:,0],sv[:,1],sv_labels,s=15,markeredgewidth=3)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")





#####################################
# 尝试改变C和gamma参数
#####################################

fig,axes=plt.subplots(3,3,figsize=(15,10))


from mglearn.plots import plot_svm

for ax,C in zip(axes,[-1,0,3]):
    for a,gamma in zip(ax,range(-1,2)):
        print("C:{}\t\t gamma:{}".format(C,gamma))
        plot_svm(log_C=C,log_gamma=gamma,ax=a)


axes[0,0].legend(["class 0","class 1","sv class 0","sv class 1"],ncol=4,loc=(.9,1.2))



from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
cancer=load_breast_cancer()
X_train,X_test,y_train,y_test=train_test_split(cancer.data,cancer.target,random_state=0)

#计算训练集中每个特征最小值
min_on_training=X_train.min(axis=0)
# 计算训练集中每个特征的范围 (最大值-最小值)
range_on_training=(X_train - min_on_training).max(axis=0)

#减去最小值，然后除以范围
#这样每个特征都是min=0和max=1
X_train_scaled = (X_train - min_on_training) / range_on_training
print("Minimun for each feature \n{}".format(X_train_scaled.min(axis=0)))
print("Maximun for each feature \n{}".format(X_train_scaled.max(axis=0)))

#利用训练集的最小值和范围对测试机做相同的变换
X_test_scaled = (X_test - min_on_training) / range_on_training



svc=SVC(C=1000)
svc.fit(X_train_scaled,y_train)
print("Accuracy on training set: {:.3f}".format(svc.score(X_train_scaled,y_train)))
print("Accuracy on test set: {:.3f}".format(svc.score(X_test_scaled,y_test)))

#plt.show()
