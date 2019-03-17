import matplotlib.pyplot as plt
from mglearn import cm3
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from mglearn import discrete_scatter

cancer=load_breast_cancer()

scaler = StandardScaler()
scaler.fit(cancer.data)
X_scaled=scaler.transform(cancer.data)

#保留数据的前2个主成分
pca=PCA(n_components=2)
#对乳腺癌数据拟合PCA模型
pca.fit(X_scaled)

# 将数据变换到前两个主成分的方向上
X_pca=pca.transform(X_scaled)
print("Original shape:{}".format(str(X_scaled.shape)))
print("Reduced shape:{}".format(str(X_pca.shape)))




# 对第一个和第二个主成分作图，按类别着色

plt.figure(figsize=(8,8))
discrete_scatter(X_pca[:,0],X_pca[:,1],cancer.target)
plt.legend(cancer.target_names,loc="best")
plt.gca().set_aspect("equal")
plt.xlabel("First principal component")
plt.xlabel("Second principal component")



print(pca.components_.shape)
print(pca.components_)

# 对系数可视化
plt.matshow(pca.components_,cmap="viridis")
plt.yticks([0,1],["First component","Second component"])
plt.colorbar()
plt.xticks(range(len(cancer.feature_names)),cancer.feature_names,rotation=60,ha="left")

plt.xlabel("Feature")
plt.xlabel("Principal components")
plt.show()