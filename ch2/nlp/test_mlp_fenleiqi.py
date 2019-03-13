from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_circles
import numpy as np
from sklearn.model_selection import train_test_split


X,y = make_circles(noise=0.25,factor=0.5,random_state=1)

# 为了便于说明，我们将两个类别重命名为blue 和 red
y_named = np.array(["blue","red"])[y]


# 我们可以对任意个数组调用train_test_split
# 所有数组的划分方式都是一致的
X_train,X_test,y_train_named,y_test_named,y_train,y_test = train_test_split(X,y_named,y,random_state=0)

# 构建梯度提升模型
gbrt = GradientBoostingClassifier(random_state=0)
gbrt.fit(X_train,y_train_named)

print("X_test.shape:{}".format(X_test.shape))
print("Decision function shape:{}".format(gbrt.decision_function(X_test)))
# 显示 decision function的前几个元素
print("Decision function element:{}".format(gbrt.decision_function(X_test)[:6]))

print("---------------------------------------------")
print("---------------------------------------------")

#通过仅查看决策函数的正负号来在现预测值
print("Thresholded  decision function :\n{}".format(gbrt.decision_function(X_test)>0))

print("Thresholded  decision function :\n{}".format(gbrt.predict(X_test)))


# predict_proba 输出的是每个类别的概率 通常比 decision_function更容易理解

print("Shape of probabilities: {}".format(gbrt.predict_proba(X_test).shape))

print("Shape of probabilities: {}".format(gbrt.predict_proba(X_test[:6])))