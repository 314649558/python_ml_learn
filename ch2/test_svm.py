from sklearn.datasets import make_blobs
from mglearn.tools import discrete_scatter
import matplotlib.pyplot as plt
X,y=make_blobs(centers=4,random_state=8)
discrete_scatter(X[:,0],X[:,1],y)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.show()

