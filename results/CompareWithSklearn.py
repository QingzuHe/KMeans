
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

plt.subplot(321)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.title("Sklearn KMeans")

y_pred_1 = np.loadtxt("X_labels.txt")

plt.subplot(322)
plt.scatter(X[:, 0], X[:, 1], c=y_pred_1)
plt.title("C++11 KMeans")

# Anisotropicly distributed data
transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
X_aniso = np.dot(X, transformation)
y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(X_aniso)

np.savetxt("X_aniso.txt", X_aniso, fmt='%.10f', delimiter=' ')

plt.subplot(323)
plt.scatter(X_aniso[:, 0], X_aniso[:, 1], c=y_pred)
plt.title("Sklearn KMeans")

y_pred_1 = np.loadtxt("X_aniso_labels.txt")

plt.subplot(324)
plt.scatter(X_aniso[:, 0], X_aniso[:, 1], c=y_pred_1)
plt.title("C++11 KMeans")

# Different variance
X_varied, y_varied = make_blobs(n_samples=n_samples,
                                cluster_std=[1.0, 2.5, 0.5],
                                random_state=random_state)
y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(X_varied)

np.savetxt("X_varied.txt", X_varied, fmt='%.10f', delimiter=' ')

plt.subplot(325)
plt.scatter(X_varied[:, 0], X_varied[:, 1], c=y_pred)
plt.title("Sklearn KMeans")

y_pred_1 = np.loadtxt("X_varied_labels.txt")

plt.subplot(326)
plt.scatter(X_varied[:, 0], X_varied[:, 1], c=y_pred_1)
plt.title("C++11 KMeans")
plt.savefig("Results of the comparison", dpi=300)
plt.show()
