from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from mglearn.plots import plot_2d_separator
from mglearn.tools import discrete_scatter
import matplotlib.pyplot as plt
X,y=make_moons(n_samples=100,noise=.25,random_state=3)

X_train,X_test,y_train,y_test=train_test_split(X,y,stratify=y,random_state=42)

#hidden_layer_sizes=[10,10] 表示使用两个隐藏层，每个隐藏层包含10个隐单元
mlp=MLPClassifier(solver="lbfgs",activation="tanh",random_state=0,hidden_layer_sizes=[10,10]).fit(X_train,y_train)

plot_2d_separator(mlp,X_train,fill=True,alpha=.3)

discrete_scatter(X_train[:,0],X_train[:,1],y_train)

plt.xlabel("Feature 0")
plt.ylabel("Feature 1")

plt.show()

