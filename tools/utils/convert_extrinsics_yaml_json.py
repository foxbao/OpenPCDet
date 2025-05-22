import os
import yaml
import json
import numpy as np
from transforms3d.quaternions import quat2mat, mat2quat

def load_transform_matrix(yaml_path):
    """从 YAML 文件中加载变换并转为 4x4 齐次矩阵"""
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    trans = data['transform']['translation']
    rot = data['transform']['rotation']
    t = np.array([trans['x'], trans['y'], trans['z']])
    q = np.array([rot['w'], rot['x'], rot['y'], rot['z']])  # 注意：wxyz 顺序
    R = quat2mat(q)  # 3x3 旋转矩阵

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def transform_to_list(T):
    """从 4x4 齐次矩阵提取 translation 和四元数"""
    t = T[:3, 3]
    R = T[:3, :3]
    q = mat2quat(R)  # 返回 [w, x, y, z]
    return [*t, *q]

def main():
    # 路径设置
    base_dir = "./data/002_params/params"
    imu_yaml = os.path.join(base_dir, "imu_vehicle_extrinsics.yaml")

    lidar_yaml_files = {
        "helios_front_left": "helios_front_left_extrinsics.yaml",
        "helios_rear_right": "helios_rear_right_extrinsics.yaml",
        "bp_front_left": "bp_front_left_extrinsics.yaml",
        "bp_rear_right": "bp_rear_right_extrinsics.yaml",
    }

    # imu -> vehicle（从 YAML 文件读出来的是 imu -> vehicle）
    T_vehicle_imu = load_transform_matrix(imu_yaml)

    extrinsics = {}

    for name, lidar_file in lidar_yaml_files.items():
        lidar_path = os.path.join(base_dir, lidar_file)
        T_lidar_imu = load_transform_matrix(lidar_path)  # 从 imu 到 lidar，实际上是 T_lidar_imu
        T_imu_lidar = np.linalg.inv(T_lidar_imu)         # 我们需要 imu <- lidar
        T_vehicle_lidar = T_vehicle_imu @ T_imu_lidar    # vehicle <- imu <- lidar
        extrinsics[f"Tx_baselink_lidar_{name}"] = transform_to_list(T_vehicle_lidar)

    output_path = os.path.join(base_dir, "extrinsics.json")
    with open(output_path, "w") as f:
        json.dump(extrinsics, f, indent=4)

    print(f"✅ extrinsics.json 已生成于 {output_path}")

if __name__ == "__main__":
    main()
