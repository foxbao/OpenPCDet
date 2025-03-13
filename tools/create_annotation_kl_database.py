# from pcdet.datasets.kl.kl_dataset import KLDataset
# from pcdet.datasets.kl.kl_dataset import create_kl_infos
# from pcdet.utils import common_utils
# import open3d as o3d
# import numpy as np
# import json
# import os
# import shutil
# from typing import List, Tuple
# import pickle

# # from pcdet.datasets.kl.kl_dataset import create_kl_info

# # def bin_to_pcd(bin_file, pcd_file):
# #     # 定义数据类型
# #     # The lidar .bin data of corage contains 24 bytes for one point
# #     dtype = np.dtype([
# #         ('x', np.float32),  # 4 bytes
# #         ('y', np.float32),  # 4 bytes
# #         ('z', np.float32),  # 4 bytes

# #         ('intensity',np.float32),
# #         ('ring',np.float32),
# #         ('timestamp_2us',np.float32),
        
# #     ])
    
# #     # 读取 .bin 文件
# #     data = np.fromfile(bin_file, dtype=dtype)
# #     # points = data.reshape(-1, 6)
# #     # 提取 x, y, z 坐标
# #     points = np.vstack((data['x'], data['y'], data['z'])).transpose()
    
# #     # 创建 Open3D 点云对象
# #     pcd = o3d.geometry.PointCloud()
# #     pcd.points = o3d.utility.Vector3dVector(points)
    
# #     # 保存为 .pcd 文件
# #     o3d.io.write_point_cloud(pcd_file, pcd)
# #     print(f"Saved {pcd_file}")



# # def save_pcd_as_bin(pcd_file, bin_file):
# #     """将 pcd 文件保存为 pcd.bin 文件"""
# #     # 使用 open3d 读取 pcd 文件
# #     pcd = o3d.io.read_point_cloud(pcd_file)
# #     points = np.asarray(pcd.points,dtype=np.float32)  # 点云的坐标 (N, 3)
    
# #     # 如果点云数据是 4 维（例如包含强度信息），也可以这样处理
# #     # points = np.asarray(pcd.points)  # (N, 3)
# #     # intensity = np.asarray(pcd.colors)  # (N, 1) 可能需要根据文件类型来处理
# #     # 去掉包含 NaN 的点
# #     valid_points = points[~np.isnan(points).any(axis=1)]  # 删除包含 NaN 的点
# #     # 将点云数据保存为 .bin 文件
# #     # 将过滤后的点云数据保存为 .bin 文件
# #     valid_points.tofile(bin_file)

# # def generate_infos(dataset_root_dir, dataset_save_dir, split):
# #     """生成 infos 文件"""
# #     import glob
# #     data_dir=dataset_root_dir/split
# #     save_dir=dataset_save_dir/split
# #     infos = []
# #     bin_files = list(data_dir.rglob('*.bin'))
# #     # pcd_files = glob.glob(os.path.join(root_dir, split, '*.bin'))
# #     for idx,bin_file in enumerate(bin_files):
        
# #         base = os.path.basename(bin_file)
# #         pcd_file = os.path.join(save_dir, base.replace('.bin', '.pcd'))
# #         bin_to_pcd(bin_file, pcd_file)

# # def quaternion_to_yaw(rotation)->float:
# #     """
# #     将四元数转换为偏航角 (yaw)。
# #     :param rotation: 四元数，形状为 4 的 numpy 数组。
# #     :return: 偏航角
# #     """
# #     qx, qy, qz, qw = rotation[0], rotation[1], rotation[2], rotation[3]
# #     yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))
# #     return yaw

# # def convert_to_gt_boxes_7dof(xyz, lwh, rotation):
# #     # 确保输入是 numpy 数组
# #     xyz = np.asarray(xyz)
# #     lwh = np.asarray(lwh)
# #     rotation = np.asarray(rotation)
    
# #     # 将四元数转换为偏航角
# #     yaw = quaternion_to_yaw(rotation)
    
# #     # 将 xyz, lwh, yaw 拼接成 gt_boxes
# #     gt_boxes = np.concatenate([xyz, lwh, [yaw]])
    
# #     return gt_boxes

# # def convert_json_to_gt(annotations:List[dict]):
# #     gt_boxes=[]
# #     gt_names=[]
# #     gt_subtype=[]
# #     gt_boxes_token=[]
# #     for data in annotations:
# #         gt_boxes.append(convert_to_gt_boxes_7dof(data['xyz'],data['lwh'],data['rotation']))
# #         gt_names.append(data['label'])
# #         gt_subtype.append(data['subtype'])
# #         gt_boxes_token.append(data['track_id'])
# #     gt_boxes = np.vstack(gt_boxes)
# #     gt_names = np.array(gt_names)
# #     gt_subtype= np.array(gt_subtype)
# #     gt_boxes_token = np.array(gt_boxes_token)
# #     return gt_boxes,gt_names,gt_subtype,gt_boxes_token

# # def split_label(label_dir,save_dir):
# #     json_files = list(label_dir.glob('*.json'))

# #     total_files = len(json_files)
# #     train_size = int(total_files * 0.8)
# #     val_size = int(total_files * 0.1)
# #     split_files = {
# #         'train': json_files[:train_size],
# #         'val': json_files[train_size:train_size + val_size],
# #         'test': json_files[train_size + val_size:]
# #     }

# #     # 创建目录并复制文件
# #     for split, files in split_files.items():
# #         # 创建目录
# #         split_dir = save_dir / split
# #         split_dir.mkdir(exist_ok=True)

# #         # 复制文件
# #         for file in files:
# #             shutil.copy(file, split_dir / file.name)
# #         print(f"{split} 目录已创建，复制了 {len(files)} 个文件")


# # def merge_json_files(input_dir, output_file,sensor_folders):
# #     merged_data = []
    
# #     # 遍历目录下的所有 JSON 文件
# #     for json_file in input_dir.glob('*.json'):
# #         # 读取文件名（时间戳）
# #         timestamp = json_file.stem  # 去掉扩展名，获取文件名（时间戳）
        
# #         # 读取 JSON 文件内容
# #         with open(json_file, 'r', encoding='utf-8') as f:
# #             data = json.load(f)
        
# #         lidars=find_pointclouds(timestamp, sensor_folders)

# #         gt_boxes,gt_names,gt_subtypes,gt_boxes_token=convert_json_to_gt(data)
# #         # 为每个样本添加 timestamp、token 和 pointcloud_path
# #         sample = {
# #             'timestamp': timestamp,
# #             'token': generate_token(),
# #             'annotations': data,
# #             'gt_boxes':gt_boxes,
# #             'gt_names':gt_names,
# #             'gt_subtypes':gt_subtypes,
# #             'gt_boxes_token':gt_boxes_token
# #         }
# #         sample.update(lidars)
        
# #         # 添加到合并后的数据中
# #         merged_data.append(sample)
# #     return merged_data
# #     # # 将合并后的数据保存到输出文件
# #     # with open(output_file, 'w', encoding='utf-8') as f:
# #     #     json.dump(merged_data, f, indent=4, ensure_ascii=False)

# # # 查找最近的点云文件
# # def find_nearest_pointcloud(timestamp, pointcloud_files):
# #     """
# #     根据时间戳查找最近的点云文件。
# #     :param timestamp: 标注文件的时间戳（整数或字符串）
# #     :param pointcloud_files: 点云文件列表（Path 对象）
# #     :return: 最近的点云文件路径（字符串）
# #     """
# #     timestamp = int(timestamp)  # 确保时间戳是整数
# #     # 提取点云文件的时间戳
# #     pointcloud_timestamps = [int(f.stem) for f in pointcloud_files]
# #     # 计算时间戳差值
# #     diffs = np.abs(np.array(pointcloud_timestamps) - timestamp)
# #     # 找到最小差值的索引
# #     nearest_index = np.argmin(diffs)
# #     return str(pointcloud_files[nearest_index])


# # # 查找点云文件
# # def find_pointclouds(timestamp:str, lidar_folders:dict)->dict:
# #     """
# #     根据时间戳在六个文件夹中查找对应的点云文件。
# #     :param timestamp: 标注文件的时间戳（字符串）
# #     :param lidar_folders: 六个激光雷达的文件夹名称列表
# #     :param corage_data_path: 点云文件根目录
# #     :return: 一个字典，键为雷达名称，值为点云文件路径（如果文件不存在，则用 null 表示）
# #     """
# #     lidars = {}
# #     for lidar_name in lidar_folders:
# #         pointcloud_files = list(lidar_folders[lidar_name].glob('*.bin'))  # 假设点云文件是 .pcd 格式
# #         pointcloud_file=find_nearest_pointcloud(timestamp,pointcloud_files)
# #         lidars[lidar_name] = str(pointcloud_file)
# #     return lidars

# # # 生成 token 的工具函数
# # def generate_token():
# #     import uuid
# #     return str(uuid.uuid4())

# # def create_kl_info(version, data_path, save_path, max_sweeps=10, with_cam=False):
# #     data_path = data_path / version
# #     save_path = save_path / version
# #     corage_path=data_path/'corage'
# #     corage_labal_path=corage_path/'s7'/'igv_1114_rain_01-century02'
# #     corage_data_lidar_dir=corage_path/'training_s7'/'igv_1114_rain_01-century02'/'lidar'
# #     corage_data_camera_dir=corage_path/'training_s7'/'igv_1114_rain_01-century02'/'camera'
# #     corage_data_localization_dir=corage_path/'training_s7'/'igv_1114_rain_01-century02'/'localization'

# #     sensor_folders={
# #         'helios_front_left':corage_data_lidar_dir/'helios_front_left',
# #         'helios_rear_right':corage_data_lidar_dir/'helios_rear_right',
# #         'bp_front_left':corage_data_lidar_dir/'bp_front_left',
# #         'bp_front_right':corage_data_lidar_dir/'bp_front_right',
# #         'bp_rear_left':corage_data_lidar_dir/'bp_rear_left',
# #         'bp_rear_right':corage_data_lidar_dir/'bp_rear_right'}
    
# #     split_label(corage_labal_path,save_path)

# #     # 定义输出文件路径
# #     output_files = {
# #         'train': data_path / 'train.pkl',
# #         'val': data_path / 'val.pkl',
# #         'test': data_path / 'test.pkl'
# #     }


# #     # 合并 train、val、test 目录下的文件
# #     for split, output_file in output_files.items():
# #         input_dir = data_path / split
# #         if input_dir.exists():  # 检查目录是否存在
# #             merged_data=merge_json_files(input_dir, output_file,sensor_folders)
# #             with open(output_file, 'wb') as f:
# #                 pickle.dump(merged_data, f)
# #                 print(f"{split} 目录下的文件已合并到 {output_file}")
# #         else:
# #             print(f"目录不存在: {input_dir}")

# if __name__ == '__main__':
#     # 示例调用
#     # bin_file = "1731568817501.bin"  # 输入的 .bin 文件
#     # pcd_file = "point_cloud.pcd"  # 输出的 .pcd 文件
#     # bin_to_pcd(bin_file, pcd_file)


#     import yaml
#     import argparse
#     from pathlib import Path
#     from easydict import EasyDict

#     parser = argparse.ArgumentParser(description='arg parser')
#     parser.add_argument('--cfg_file', type=str, default=None, help='specify the config of dataset')
#     parser.add_argument('--func', type=str, default='create_kl_infos', help='')
#     parser.add_argument('--version', type=str, default='v1.0-trainval', help='')
#     parser.add_argument('--with_cam', action='store_true', default=False, help='use camera or not')
#     args = parser.parse_args()
    
#     if args.func == 'create_kl_infos':
#     #     # dataset_cfg = EasyDict(yaml.safe_load(open(args.cfg_file)))
#         ROOT_DIR = (Path(__file__).resolve().parent / '../').resolve()
#         data_path=ROOT_DIR / 'data' / 'kl'
#         # save_path=ROOT_DIR / 'data' / 'kl'
#         create_kl_infos(args.version, data_path, data_path, max_sweeps=10, with_cam=args.with_cam)

#         aaaaa=1
        