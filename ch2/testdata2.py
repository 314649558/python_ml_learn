from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_boston
import numpy as np
import mglearn.datasets,mglearn.plots

cancer = load_breast_cancer()


print("cancer.keys(): \n{}".format(cancer.keys()))


print("Shape of cancer data:{}".format(cancer.data.shape))


print("Sample counts per class: \n{}".format({n: v for n , v in zip(cancer.target_names,np.bincount(cancer.target))}))

print("Feature names:\n{}".format(cancer.feature_names))

print("\n------------------------DESCR---------------------------------\n")

#print(cancer.DESCR)


print("\n波士顿房价数据集\n")
boston=load_boston()
print("Data shape:{}".format(boston.data.shape))


X,y=mglearn.datasets.load_extended_boston()

print("X.shape:{}".format((X.shape)))

#mglearn.plots.plot_knn_classification

a=np.random.RandomState(42)

print(a)