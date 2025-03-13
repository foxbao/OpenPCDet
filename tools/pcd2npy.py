import numpy as np
import open3d as o3d

pcd = o3d.io.read_point_cloud("point_cloud.pcd")
points = np.asarray(pcd.points)
zeros_column = np.zeros((points.shape[0], 1))

points = np.concatenate((points, zeros_column), axis=1)
points[:, 3] = 0 
np.save('my_data.npy', points) 