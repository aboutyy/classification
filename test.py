# -*- coding: UTF-8 -*-
__author__ = 'Administrator'

import progressbar
import time
import laspy
print(__doc__)

import numpy as np

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D

##############################################################################
# Generate sample data
# centers = [[1, 1], [-1, -1], [1, -1]]
# centers = [[1, 1, 0], [-1, -1, -1], [1, -1, -2]]
# X, labels_true = make_blobs(n_samples=1750, n_features=3, centers=centers, cluster_std=0.4,
#                             random_state=0)
# X = StandardScaler().fit_transform(X)
infile = laspy.file.File('2.las', mode="rw")
X = np.vstack([infile.x, infile.y, infile.z]).transpose()
##############################################################################
# Compute DBSCAN
db = DBSCAN(eps=0.3, min_samples=10).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
infile.gps_time = labels
# Number of clusters in labels, ignoring noise if present.
# n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

# print('Estimated number of clusters: %d' % n_clusters_)
# print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
# print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
# print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
# print("Adjusted Rand Index: %0.3f"
#       % metrics.adjusted_rand_score(labels_true, labels))
# print("Adjusted Mutual Information: %0.3f"
#       % metrics.adjusted_mutual_info_score(labels_true, labels))
# print("Silhouette Coefficient: %0.3f"
#       % metrics.silhouette_score(X, labels))

##############################################################################
# Plot result
# import matplotlib.pyplot as plt

# Black removed and is used for noise instead.
# unique_labels = set(labels)
# colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
# for k, col in zip(unique_labels, colors):
#     if k == -1:
#         # Black used for noise.
#         col = 'k'
#
#     class_member_mask = (labels == k)
#
#     xy = X[class_member_mask & core_samples_mask]
#     plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
#              markeredgecolor='k', markersize=14)
#
#     xy = X[class_member_mask & ~core_samples_mask]
#     plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
#              markeredgecolor='k', markersize=6)
# plt.title('Estimated number of clusters: %d' % n_clusters_)
# plt.show()
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# for k, col in zip(unique_labels, colors):
#     if k == -1:
#         # Black used for noise.
#         col = 'k'
#
#     class_member_mask = (labels == k)
#
#     xy = X[class_member_mask & core_samples_mask]
#     ax.scatter(xy[:, 0], xy[:, 1], xy[:, 2], c=col, marker='o')
#
#     xy = X[class_member_mask & ~core_samples_mask]
#     ax.scatter(xy[:, 0], xy[:, 1], xy[:, 2], c=col, marker='^')
#
# # ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=colors)
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
# plt.show()
infile.close()



#
# widgets = ['Something: ', progressbar.Percentage(), ' ', progressbar.Bar(),
#            ' ', progressbar.ETA(), ' ', progressbar.FileTransferSpeed()]
# pbar = progressbar.ProgressBar(widgets=widgets, maxval=10000000).start()
# for i in range(1000000):
#   pbar.update(10*i+1)
# pbar.finish()
