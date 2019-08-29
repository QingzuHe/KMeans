# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 20:02:25 2019

@author: hqz
"""


import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

n_samples = 1500
random_state = 170
X, y = make_blobs(n_samples=n_samples, random_state=random_state)


# Incorrect number of clusters
y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(X)

np.savetxt("X.txt", X, fmt='%.10f', delimiter=' ')

plt.figure(0)
plt.figure(figsize=(15, 20))

plt.subplot(421)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.title("Sklearn KMeans")

y_pred_1 = np.loadtxt("X_labels.txt")

plt.subplot(422)
plt.scatter(X[:, 0], X[:, 1], c=y_pred_1)
plt.title("C++11 KMeans")

# Anisotropicly distributed data
transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
X_aniso = np.dot(X, transformation)
y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(X_aniso)

np.savetxt("X_aniso.txt", X_aniso, fmt='%.10f', delimiter=' ')

plt.subplot(423)
plt.scatter(X_aniso[:, 0], X_aniso[:, 1], c=y_pred)
plt.title("Sklearn KMeans")

y_pred_1 = np.loadtxt("X_aniso_labels.txt")

plt.subplot(424)
plt.scatter(X_aniso[:, 0], X_aniso[:, 1], c=y_pred_1)
plt.title("C++11 KMeans")

# Different variance
X_varied, y_varied = make_blobs(n_samples=n_samples,
                                cluster_std=[1.0, 2.5, 0.5],
                                random_state=random_state)
y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(X_varied)

np.savetxt("X_varied.txt", X_varied, fmt='%.10f', delimiter=' ')

plt.subplot(425)
plt.scatter(X_varied[:, 0], X_varied[:, 1], c=y_pred)
plt.title("Sklearn KMeans")

y_pred_1 = np.loadtxt("X_varied_labels.txt")

plt.subplot(426)
plt.scatter(X_varied[:, 0], X_varied[:, 1], c=y_pred_1)
plt.title("C++11 KMeans")


# Unevenly sized blobs
X_filtered = np.vstack((X[y == 0][:500], X[y == 1][:100], X[y == 2][:10]))
y_pred = KMeans(n_clusters=3,
                random_state=random_state).fit_predict(X_filtered)

np.savetxt("X_filtered.txt", X_filtered, fmt='%.10f', delimiter=' ')

plt.subplot(427)
plt.scatter(X_filtered[:, 0], X_filtered[:, 1], c=y_pred)
plt.title("Sklearn KMeans")

y_pred_1 = np.loadtxt("X_filtered_labels.txt")

plt.subplot(428)
plt.scatter(X_filtered[:, 0], X_filtered[:, 1], c=y_pred_1)
plt.title("C++11 KMeans")

plt.savefig("ResultsOfComparison.jpg")
plt.show()
