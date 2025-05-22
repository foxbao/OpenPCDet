import os
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import bisect
from transforms3d.quaternions import quat2mat
import struct

def load_bin(file_path):
    return np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)

def save_bin(points, file_path):
    points.astype(np.float32).tofile(file_path)

def save_pcd_with_intensity(points, file_path):
    # points shape: (N, >=4), 前4列是x,y,z,intensity
    num_points = points.shape[0]
    header = f"""# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS x y z intensity
SIZE 4 4 4 4
TYPE F F F F
COUNT 1 1 1 1
WIDTH {num_points}
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS {num_points}
DATA ascii
"""
    with open(file_path, 'w') as f:
        f.write(header)
        for pt in points:
            f.write(f"{pt[0]} {pt[1]} {pt[2]} {pt[3]}\n")

def load_pcd_binary_with_ring(file_path):
    with open(file_path, 'rb') as f:
        header = b""
        while True:
            line = f.readline()
            header += line
            if line.strip() == b"DATA binary":
                break
        header_str = header.decode("utf-8")
        lines = header_str.splitlines()
        # 解析FIELDS, SIZE, TYPE, COUNT, WIDTH, HEIGHT
        field_line = [line for line in lines if line.startswith("FIELDS")][0]
        fields = field_line.strip().split()[1:]
        size_line = [line for line in lines if line.startswith("SIZE")][0]
        sizes = list(map(int, size_line.strip().split()[1:]))
        type_line = [line for line in lines if line.startswith("TYPE")][0]
        types = type_line.strip().split()[1:]
        count_line = [line for line in lines if line.startswith("COUNT")][0]
        counts = list(map(int, count_line.strip().split()[1:]))
        width_line = [line for line in lines if line.startswith("WIDTH")][0]
        width = int(width_line.strip().split()[1])
        height_line = [line for line in lines if line.startswith("HEIGHT")][0]
        height = int(height_line.strip().split()[1])
        num_points = width * height

        # PCD type+size 转 struct fmt 映射
        fmt_map = {
            ('F', 4): 'f',
            ('F', 8): 'd',
            ('U', 1): 'B',
            ('U', 2): 'H',
            ('U', 4): 'I',
            ('I', 1): 'b',
            ('I', 2): 'h',
            ('I', 4): 'i',
        }

        fmt = ""
        for t, s, c in zip(types, sizes, counts):
            fmt += fmt_map[(t, s)] * c
        point_struct = struct.Struct(fmt)
        point_size = point_struct.size

        raw = f.read(num_points * point_size)
        points = []
        for i in range(num_points):
            point = point_struct.unpack_from(raw, i * point_size)
            points.append(point)
        points = np.array(points, dtype=np.float32)
        return points  # Nx#fields

def is_pcd_binary(file_path):
    # 读头部确定是否是binary格式pcd
    with open(file_path, 'rb') as f:
        for _ in range(20):  # 读前20行以内找DATA行
            line = f.readline()
            if line.startswith(b'DATA binary'):
                return True
            if line.startswith(b'DATA ascii'):
                return False
    return False  # 默认ASCII（或非binary）

def get_transformation_matrix(pose, quat_mode='wxyz'):
    translation = np.array(pose[:3])
    if quat_mode == 'wxyz':
        quat = np.array([pose[3], pose[4], pose[5], pose[6]])  # w, x, y, z
    elif quat_mode == 'xyzw':
        quat = np.array([pose[6], pose[3], pose[4], pose[5]])  # 转为 w, x, y, z
    else:
        raise ValueError(f"Unsupported quaternion mode: {quat_mode}")

    rotation = quat2mat(quat)
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = translation
    return T

def find_closest_timestamp(sorted_timestamps, target_ts):
    pos = bisect.bisect_left(sorted_timestamps, target_ts)
    candidates = []
    if pos > 0:
        candidates.append(sorted_timestamps[pos - 1])
    if pos < len(sorted_timestamps):
        candidates.append(sorted_timestamps[pos])
    closest = min(candidates, key=lambda x: abs(x - target_ts)) if candidates else None
    return closest

def merge_pointclouds_to_vehicle(sensor_root_dir, extrinsics_json_path, output_dir, max_time_diff=0.05, quat_mode='wxyz'):
    folder_to_extrinsics_key = {
        # "bp_front_left": "Tx_baselink_lidar_bp_front_left",
        # "bp_rear_right": "Tx_baselink_lidar_bp_rear_right",
        "helios_front_left": "Tx_baselink_lidar_helios_front_left",
        "helios_rear_right": "Tx_baselink_lidar_helios_rear_right"
    }

    with open(extrinsics_json_path, 'r') as f:
        extrinsics_dict = json.load(f)

    sensor_root = Path(sensor_root_dir)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    sensor_timestamps = {}
    for folder_name in folder_to_extrinsics_key.keys():
        folder_path = sensor_root / folder_name
        # 判断文件后缀，bin or pcd
        files = sorted(folder_path.glob("*"))
        ts_list = []
        for f in files:
            if f.suffix == '.bin' or f.suffix == '.pcd':
                try:
                    ts_list.append(float(f.stem))
                except:
                    continue
        sensor_timestamps[folder_name] = sorted(ts_list)

    reference_sensor = "helios_front_left"
    ref_timestamps = sensor_timestamps[reference_sensor]

    for ref_ts in tqdm(ref_timestamps, desc="Processing frames"):
        merged_points = []

        for folder_name, extrinsics_key in folder_to_extrinsics_key.items():
            ts_list = sensor_timestamps[folder_name]
            closest_ts = find_closest_timestamp(ts_list, ref_ts)
            if closest_ts is None or abs(closest_ts - ref_ts) > max_time_diff:
                continue

            # 找文件
            folder_path = sensor_root / folder_name
            file_bin = folder_path / f"{closest_ts:.9f}.bin"
            file_pcd = folder_path / f"{closest_ts:.9f}.pcd"
            points = None

            if file_bin.exists():
                points = load_bin(file_bin)
            elif file_pcd.exists():
                if is_pcd_binary(file_pcd):
                    points = load_pcd_binary_with_ring(file_pcd)
                else:
                    # ASCII pcd读取，简单实现只取 x y z intensity 四个字段
                    points = []
                    with open(file_pcd, 'r') as f:
                        data_started = False
                        for line in f:
                            if data_started:
                                parts = line.strip().split()
                                if len(parts) >= 4:
                                    x, y, z, intensity = map(float, parts[:4])
                                    points.append([x, y, z, intensity])
                            if line.strip() == 'DATA ascii':
                                data_started = True
                    points = np.array(points, dtype=np.float32)
            else:
                # 没找到文件就跳过
                continue

            # 变换点云到车体坐标系
            T = get_transformation_matrix(extrinsics_dict[extrinsics_key], quat_mode=quat_mode)
            points_xyz = points[:, :3]
            points_intensity = points[:, 3:4] if points.shape[1] > 3 else np.zeros((points.shape[0],1), dtype=np.float32)
            points_hom = np.hstack((points_xyz, np.ones((points.shape[0], 1))))
            transformed_xyz = (T @ points_hom.T).T[:, :3]
            merged = np.hstack((transformed_xyz, points_intensity))
            merged_points.append(merged)

        if merged_points:
            merged_all = np.concatenate(merged_points, axis=0)
            bin_filename = f"{ref_ts:.9f}.bin"
            pcd_filename = f"{ref_ts:.9f}.pcd"
            save_bin(merged_all, output_root / bin_filename)
            save_pcd_with_intensity(merged_all, output_root / pcd_filename)

if __name__ == "__main__":
    sensor_root_dir = "data/202505201008_record"
    extrinsics_json_path = "data/002_params/params/extrinsics.json"
    # sensor_root_dir = "data/out_chenxu"
    # extrinsics_json_path = "data/out_chenxu/extrinsic.json"
    output_dir = sensor_root_dir + "_merged"

    # 手动指定 quat_mode
    merge_pointclouds_to_vehicle(sensor_root_dir, extrinsics_json_path, output_dir, quat_mode='xyzw')
