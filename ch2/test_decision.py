from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
cancer=load_breast_cancer()

X_train,y_train,X_test,y_test=train_test_split(cancer.data,cancer.target,stratify=cancer.target,random_state=42)

# 如果不限制max_depth【决策时深度】训练集数据会100%的分类，出现过拟合，从而导致在测试数据上模型适用
tree=DecisionTreeClassifier(max_depth=4,random_state=0)

tree.fit(X_train,y_train)

print("Accuracy on training set:{:.3f}".format(tree.score(X_train,y_train)))
print("Accuracy on test set:{:.3f}".format(tree.score(X_test,y_test)))

