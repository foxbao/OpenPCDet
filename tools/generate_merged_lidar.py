import numpy as np
import open3d as o3d
from pcdet.datasets.kl.kl_dataset_utils import transform_points
# from pcdet.datasets.kl.kl_dataset import read_bin

def save_point_cloud_to_pcd(points, output_path):
    """
    将点云数据保存为 .pcd 文件
    :param points: 点云数据，形状为 (N, 3) 或 (N, 4)
    :param output_path: 输出 .pcd 文件的路径
    """
    # 创建 Open3D 点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])  # 只取 x, y, z 坐标

    # 如果有强度信息（第 4 列），可以将其添加到点云中
    if points.shape[1] >= 4:
        intensities = points[:, 3]  # 强度信息
        pcd.colors = o3d.utility.Vector3dVector(np.tile(intensities[:, np.newaxis], (1, 3)))  # 将强度映射到 RGB

    # 保存为 .pcd 文件
    o3d.io.write_point_cloud(output_path, pcd)
    print(f"点云已保存到 {output_path}")

def save_point_cloud_to_bin(points, output_path):
    """
    将点云数据保存为 .bin 文件
    :param points: 点云数据，形状为 (N, 3) 或 (N, 4)
    :param output_path: 输出 .bin 文件的路径
    """
    # 将点云数据保存为二进制文件
    points.astype(np.float32).tofile(output_path)
    print(f"点云已保存到 {output_path}")


def read_bin(bin_file):
    dtype = np.dtype([
        ('x', np.float32),  # 4 bytes
        ('y', np.float32),  # 4 bytes
        ('z', np.float32),  # 4 bytes
        ('intensity',np.float32),
        ('ring',np.float32),
        ('timestamp_2us',np.float32),
    ])
    
    # 读取 .bin 文件
    data = np.fromfile(bin_file, dtype=dtype)
    # points = data.reshape(-1, 6)
    # 提取 x, y, z 坐标
    points = np.vstack((data['x'], data['y'], data['z'],data['intensity'],data['ring'],data['timestamp_2us'])).transpose()
    return points

if __name__ == "__main__":

    point_clouds = []
    lidars = []
    lidars.append(
        {
            "path": "../data/kl/v1.0-trainval/sample/s7/igv_1114_rain_01-century02/lidar/helios_front_left/1731568817501.bin",
            "extrinsic": np.array(
                [7.336647033691406,
                1.4024821519851685,
                -0.07428044080734253,
                0.0009888865918614663,
                0.004264600754721424,
                0.3800393157007753,
                0.9249599741639623]
            ),
        }
    )

    lidars.append(
        {
            "path": "../data/kl/v1.0-trainval/sample/s7/igv_1114_rain_01-century02/lidar/helios_rear_right/1731568817501.bin",
            "extrinsic": np.array(
                [-7.255270481109619,
                -1.6236361265182495,
                0.0032361075282096863,
                -0.0005317689861419239,
                -0.0035887806881134726,
                0.9248185834950713,
                -0.3803911480267223]
            ),
        }
    )

    for lidar in lidars:
        points = read_bin(lidar["path"])
        times = np.zeros((points.shape[0], 1))
        # points = np.concatenate((points, times), axis=1)
        points = transform_points(points, lidar["extrinsic"])
        point_clouds.append(points)

    if point_clouds:  # 如果列表不为空
        merged_point_cloud = np.concatenate(point_clouds, axis=0)
        np.save("merged_point_cloud.npy", merged_point_cloud)
        # 保存为 .pcd 文件
        save_point_cloud_to_pcd(merged_point_cloud, "merged_point_cloud.pcd")

        # 保存为 .bin 文件
        save_point_cloud_to_bin(merged_point_cloud, "merged_point_cloud.bin")

    
