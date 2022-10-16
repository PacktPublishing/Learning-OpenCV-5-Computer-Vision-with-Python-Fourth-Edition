#!/usr/bin/env python

import cv2
import matplotlib.pyplot as plt

from sklearn.cluster import Birch


model_path = '../3d_models/jburkardt/street_lamp.ply'

vertices, normals = cv2.loadPointCloud(model_path)
vertices = vertices.squeeze(axis=1)

clusterer = Birch(threshold=0.3, n_clusters=6)
clusterer.fit(vertices)
labels = clusterer.predict(vertices)

xs = vertices[:,0]
ys = vertices[:,2]
zs = vertices[:,1]

min_x, max_x = xs.min(), xs.max()
min_y, max_y = ys.min(), ys.max()
min_z, max_z = zs.min(), zs.max()

fig = plt.figure(figsize=(6, 6))

ax = fig.add_subplot(111, projection='3d')

ax.set_xlim3d(min_x, max_x)
ax.set_ylim3d(min_y, max_y)
ax.set_zlim3d(min_z, max_z)

ax.set_box_aspect([max_x - min_x,
                   max_y - min_y,
                   max_z - min_z])

ax.scatter(xs, ys, zs, c=labels)

plt.show()
