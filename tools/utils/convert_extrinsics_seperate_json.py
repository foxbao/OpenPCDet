import os
import json
import numpy as np
from scipy.spatial.transform import Rotation as R

def load_extrinsics(input_dir, quat_mode='xyzw'):
    assert quat_mode in ['xyzw', 'wxyz'], "quat_mode 必须为 'xyzw' 或 'wxyz'"

    extrinsics = {}

    for filename in os.listdir(input_dir):
        if filename.endswith('.json'):
            filepath = os.path.join(input_dir, filename)
            with open(filepath, 'r') as f:
                data = json.load(f)

            transform_flat = data['Tranform']
            matrix = np.array(transform_flat).reshape(4, 4)

            # 提取平移向量
            translation = matrix[:3, 3].tolist()

            # 提取旋转矩阵并转为四元数
            quat_xyzw = R.from_matrix(matrix[:3, :3]).as_quat()  # [x, y, z, w]

            if quat_mode == 'wxyz':
                quat = [quat_xyzw[3]] + quat_xyzw[:3].tolist()  # [w, x, y, z]
            else:
                quat = quat_xyzw.tolist()  # [x, y, z, w]

            # 拼接 [tx, ty, tz, ..., ...]
            transform_vec = translation + quat

            # 构造键名
            sensor_name = filename.replace('.json', '')
            key = f"Tx_baselink_{sensor_name}"

            extrinsics[key] = transform_vec

    return extrinsics

def load_extrinsics_raw(input_dir):
    """直接加载原始的 4x4 变换矩阵，合并为一个字典"""
    extrinsics_raw = {}

    for filename in os.listdir(input_dir):
        if filename.endswith('.json'):
            filepath = os.path.join(input_dir, filename)
            with open(filepath, 'r') as f:
                data = json.load(f)

            transform_flat = data['Tranform']
            sensor_name = filename.replace('.json', '')
            key = f"Tx_baselink_{sensor_name}"

            extrinsics_raw[key] = transform_flat

    return extrinsics_raw

def main():
    input_dir = 'data/002_params/vehicle_base_cpp'
    output_dir = 'data/002_params/vehicle_base_cpp/merged'
    output_json_path = os.path.join(output_dir, 'extrinsics.json')
    output_raw_json_path = os.path.join(output_dir, 'extrinsics_raw.json')

    # ✅ 确保输出文件夹存在
    os.makedirs(output_dir, exist_ok=True)

    # 四元数版本
    extrinsics = load_extrinsics(input_dir, quat_mode='wxyz')
    with open(output_json_path, 'w') as f:
        json.dump(extrinsics, f, indent=4)
    print(f"✅ 成功生成 {output_json_path}（带四元数）")

    # 原始矩阵版本
    extrinsics_raw = load_extrinsics_raw(input_dir)
    with open(output_raw_json_path, 'w') as f:
        json.dump(extrinsics_raw, f, indent=4)
    print(f"✅ 成功生成 {output_raw_json_path}（原始矩阵）")

if __name__ == '__main__':
    main()
