# Comparing Trajectory Clustering Methods

## Update (Feb 2022)

If you have a problem downloading the public dataset described in the demo file, please try [this link](https://seljuk.me/upload/CVRR_dataset_trajectory_clustering.zip).

## Update (Feb 2022)

I recently published [a blog post](https://seljuk.me/notes-on-trajectory-clustering.html) regarding trajectory clustering. It suplements the repo in a more theoretical level, you may check it out if the general approach is not clear.

## Update (Feb 2019)

Added a [notebook](demo/demo.ipynb) demonstrating every step of the project. Please look at that first, it is more shorter and understandable than other parts of the project. It also shows these steps on a public dataset.

Public Dataset:

![Public Dataset](demo/demo1.png)

Clustered Trajectories:

![Clustered Trajectories](demo/demo2.png)

----

## Introduction

This was my pattern recognition course term project. The goal is to compare 4 clustering algorithms (k-medoids, gaussian mixture model, dbscan and hdbscan) on civil flight data. More detail can be found in report.pdf file.

![A snapshot of data](data.png)

Resulting clusters look like this:

![Resulting clusters with one method](result.png)

Trajectory segmentation is applied to reduce the number of sample points and hausdorff distance is used to compare the similarity between trajectories.

![Trajectory Segmentation](segmentation.png)
