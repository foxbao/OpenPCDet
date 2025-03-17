import numpy as np
import open3d as o3d



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--source_file', type=str, default='point_cloud.pcd', help='specify the config of dataset')

    args = parser.parse_args()
    source_file=args.source_file
    pcd = o3d.io.read_point_cloud(source_file)
    points = np.asarray(pcd.points)
    zeros_column = np.zeros((points.shape[0], 2))

    points = np.concatenate((points, zeros_column), axis=1)
    points[:, 3] = 0 
    points[:, 4] = 0 
    np.save('my_data.npy', points) 