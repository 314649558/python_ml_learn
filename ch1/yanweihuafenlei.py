# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 09:50:00 2019

@author: Administrator
"""

from sklearn.datasets import load_iris

iris_dataset=load_iris()

print("Keys of iris_dataset:\n{}".format(iris_dataset.keys()))


print(iris_dataset['DESCR'][:193]+"\n...")


# 需要预测花的品种
print("Target names:\n{}".format(iris_dataset['target_names']))


# feature_names 键对应的只是一个字符串列表，对每一个特征进行了说明
print("Feature names:\n{}".format(iris_dataset['feature_names']))

# data 里面是需要预测的数据  它是NumPy类型的数组
print("Type of data:{}".format(type(iris_dataset['data'])))


print("Shape of data:{}".format(iris_dataset['data'].shape))


print("data:\n{}".format(iris_dataset['data'][:5]))


print("target:\n{}".format(iris_dataset['target']))