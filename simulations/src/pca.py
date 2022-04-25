'''
File: permutations.py
Created Date: Monday February 21th 2022 - 18:42am
Authors: Alexandre Hippert-Ferrer
Contact: alexandre.hippert-ferrer@centralesupelec.fr
-----
Last Modified: Wed Dec 01 2021
Modified By: Alexandre Hippert-Ferrer
-----
Copyright (c) 2022 CentraleSupélec
'''

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import SpectralEmbedding
from sklearn.datasets import make_moons, make_blobs

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import numpy.linalg as la
import numpy as np
import time


def pca_data(X, nb_components, reverse=False):
    """ A function that centers data and applies PCA on an image.
        Inputs:
            * X: numpy array of the image.
            * nb_components: number of components to keep.
    """
    # center pixels
    n, p = X.shape
    mean = np.mean(X, axis=0)
    X = X - mean
    # check pixels are centered
    assert (np.abs(np.mean(X, axis=0)) < 1e-8).all()

    # apply PCA
    SCM = (1/len(X))*X.T@X
    d, Q = la.eig(SCM)

    if reverse:
        reverse_idx = np.arange(len(d)-1, -1, step=-1)
        Q = Q[:, reverse_idx]

    Q = Q[:, :nb_components]
    X = X@Q

    return X

# Iris dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# Blobs dataset
np.random.seed(0)
# Generate sample data
n_samples = 100
n_components = 3
X, y_true = make_blobs(
    n_samples=n_samples, n_features=5, centers=n_components, cluster_std=0.60, random_state=0
)

df = pd.DataFrame(X, columns=['f1', 'f2', 'f3', 'f4', 'f5'])
target = pd.DataFrame(y_true, columns=['target'])
df = pd.concat([target, df], axis = 1)

# load dataset into Pandas DataFrame
#df = pd.read_csv(url, names=['sepal length','sepal width','petal length','petal width','target'])

#features = ['sepal length', 'sepal width', 'petal length', 'petal width']

features = ['f1', 'f2', 'f3', 'f4', 'f5']

# Separating out the features
x = df.loc[:, features].values

# Separating out the target
y = df.loc[:,['target']].values

# Standardizing the features
x = StandardScaler().fit_transform(x)

# PCA 1
start = time.time()
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
print(time.time()-start)
principalDf = pd.DataFrame(data = principalComponents,
                           columns = ['principal component 1',
                                      'principal component 2']
)
finalDf = pd.concat([principalDf, df[['target']]], axis = 1)

# Home made PCA
start = time.time()
principalComponents_new = pca_data(x, 2)
print(time.time()-start)

principalDf_new = pd.DataFrame(data = principalComponents_new,
                           columns = ['principal component 1',
                                      'principal component 2']
)
finalDf_new = pd.concat([principalDf_new, df[['target']]], axis = 1)

# Laplacian Eigenmaps
embedding = SpectralEmbedding(n_components=2)
X_transformed = embedding.fit_transform(x)
principalDf_laplacian = pd.DataFrame(data = X_transformed,
                                     columns = ['principal component 1',
                                                'principal component 2']
)
finalDf_laplacian = pd.concat([principalDf_laplacian, df[['target']]], axis = 1)


fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(111)#, projection='3d')
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
#ax.set_zlabel('Principal Component 3', fontsize = 15)
ax.set_title('3 component PCA', fontsize = 20)
targets = [0, 1, 2]
#targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
colors = ['r', 'g', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['target'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1'],
               finalDf.loc[indicesToKeep, 'principal component 2'],
               #finalDf.loc[indicesToKeep, 'principal component 3'],
               c = color, s = 50)
ax.legend(targets)
ax.grid()

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(111)#, projection='3d')
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
#ax.set_zlabel('Principal Component 3', fontsize = 15)
ax.set_title('3 component PCA', fontsize = 20)
targets = [0, 1, 2]
#targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
colors = ['r', 'g', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf_new['target'] == target
    ax.scatter(finalDf_new.loc[indicesToKeep, 'principal component 1'],
               finalDf_new.loc[indicesToKeep, 'principal component 2'],
               #finalDf_new.loc[indicesToKeep, 'principal component 3'],
               c = color, s = 50)
ax.legend(targets)
ax.grid()

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(111)#, projection='3d')
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
#ax.set_zlabel('Principal Component 3', fontsize = 15)
ax.set_title('3 component PCA', fontsize = 20)
targets = [0, 1, 2]
#targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
colors = ['r', 'g', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf_laplacian['target'] == target
    ax.scatter(finalDf_laplacian.loc[indicesToKeep, 'principal component 1'],
               finalDf_laplacian.loc[indicesToKeep, 'principal component 2'],
               #finalDf_laplacian.loc[indicesToKeep, 'principal component 3'],
               c = color, s = 50)
ax.legend(targets)
ax.grid()



plt.show()
