import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# 0 - Import related libraries

import urllib
import zipfile
import os
import scipy.io
import math

import seaborn as sns
from scipy.spatial.distance import directed_hausdorff
from sklearn.cluster import DBSCAN

from kmedoid import kMedoids # kMedoids code is adapted from https://github.com/letiantian/kmedoids

sns.set()
plt.rcParams['figure.figsize'] = (12, 12)

# Utility Functions

color_lst = plt.rcParams['axes.prop_cycle'].by_key()['color']
color_lst.extend(['firebrick', 'olive', 'indigo', 'khaki', 'teal', 'saddlebrown',
                  'skyblue', 'coral', 'darkorange', 'lime', 'darkorchid', 'dimgray'])


def plot_cluster(traj_lst, cluster_lst):
    '''
    Plots given trajectories with a color that is specific for every trajectory's own cluster index.
    Outlier trajectories which are specified with -1 in `cluster_lst` are plotted dashed with black color
    '''
    cluster_count = np.max(cluster_lst) + 1

    for traj, cluster in zip(traj_lst, cluster_lst):

        if cluster == -1:
            # Means it it a noisy trajectory, paint it black
            plt.plot(traj[:, 0], traj[:, 1], c='k', linestyle='dashed')

        else:
            plt.plot(traj[:, 0], traj[:, 1], c=color_lst[cluster % len(color_lst)])
    plt.show()



root = ".\outdata"
dirs = os.listdir(root)
traj_lst = []
for dir in dirs:
    pathleft = os.path.join(root,dir,"left.csv")
    pathright = os.path.join(root,dir,"right.csv")
    print(pathleft)
    leftdf = pd.read_csv(pathleft);rightdf = pd.read_csv(pathright)
    leftdf_x = leftdf['x'];leftdf_y = leftdf['y']
    rightdf_x = rightdf['x'];rightdf_y = rightdf['y']
    traj = [[leftdf_x[i],leftdf_y[i],rightdf_x[i],rightdf_y[i]] for i in range(min(len(leftdf_x),len(rightdf_x)))]
    traj = np.array(traj)
    traj_lst.append(traj)
for traj in traj_lst:
    #print(traj)
    plt.plot(traj[:, 0], traj[:, 1])
plt.show()
# print(root) 

degree_threshold = 5
######轨迹压缩
# for traj_index, traj in enumerate(traj_lst):

#     hold_index_lst = []
#     previous_azimuth = 1000

#     for point_index, point in enumerate(traj[:-1]):
#         next_point = traj[point_index + 1]
#         diff_vector = next_point - point
#         azimuth = (math.degrees(math.atan2(*diff_vector)) + 360) % 360

#         if abs(azimuth - previous_azimuth) > degree_threshold:
#             hold_index_lst.append(point_index)
#             previous_azimuth = azimuth
#     hold_index_lst.append(traj.shape[0] - 1)  # Last point of trajectory is always added

#     traj_lst[traj_index] = traj[hold_index_lst, :]

def hausdorff( u, v):
    d = max(directed_hausdorff(u, v)[0], directed_hausdorff(v, u)[0])
    return d

traj_count = len(traj_lst)
D = np.zeros((traj_count, traj_count))

# This may take a while
for i in range(traj_count):
    for j in range(i + 1, traj_count):
        distance = hausdorff(traj_lst[i], traj_lst[j])
        D[i, j] = distance
        D[j, i] = distance
    print(i+1,"of",traj_count)

# 4 - Different clustering methods

# # 4.1 - kmedoids

# k = 10  # The number of clusters
# medoid_center_lst, cluster2index_lst = kMedoids(D, k)

# cluster_lst = np.empty((traj_count,), dtype=int)

# for cluster in cluster2index_lst:
#     cluster_lst[cluster2index_lst[cluster]] = cluster

# plot_cluster(traj_lst, cluster_lst)


# 4.2 - dbscan

mdl = DBSCAN(eps=0.65  , min_samples=1)
cluster_lst = mdl.fit_predict(D)

plot_cluster(traj_lst, cluster_lst)