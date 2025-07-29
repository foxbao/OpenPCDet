import os
import json
import shutil
import bisect
from pathlib import Path

def extract_timestamp(filename):
    """从文件名中提取浮点时间戳，例如 '1714354789.123.json'"""
    return float(Path(filename).stem)

def collect_files_with_timestamps(root_dir, extensions=('.pcd', '.jpeg')):
    """
    遍历 root_dir 下所有文件，收集指定扩展名的文件及其 timestamp（假设用 stem 名作为时间戳）
    返回： {relative_path: [(filepath, timestamp), ...]}
    """
    result = {}
    for sensor_name in os.listdir(root_dir):
        sensor_path = os.path.join(root_dir, sensor_name)
        if not os.path.isdir(sensor_path):
            continue
        file_list = []
        for fname in os.listdir(sensor_path):
            if not fname.endswith(extensions):
                continue
            try:
                ts = extract_timestamp(fname)
                file_list.append((os.path.join(sensor_path, fname), ts))
            except:
                continue
        file_list.sort(key=lambda x: x[1])  # 按 timestamp 升序
        result[sensor_name] = file_list
    return result

def find_nearest_by_timestamp_bisect(target_ts, sorted_file_list_with_ts):
    timestamps = [ts for _, ts in sorted_file_list_with_ts]
    idx = bisect.bisect_left(timestamps, target_ts)

    candidates = []
    if idx < len(timestamps):
        candidates.append((sorted_file_list_with_ts[idx], abs(timestamps[idx] - target_ts)))
    if idx > 0:
        candidates.append((sorted_file_list_with_ts[idx - 1], abs(timestamps[idx - 1] - target_ts)))

    if not candidates:
        return None
    nearest_file, min_diff = min(candidates, key=lambda x: x[1])
    return nearest_file[0]

# 设置路径
label_src_root = "/home/baojiali/Downloads/disk1/data/lightwheel_data/label"
sample_src_root = "/home/baojiali/Downloads/disk1/data/lightwheel_data/sample"
dst_root = "/home/baojiali/Downloads/disk1/data/lightwheel_data_clean_WheelCrane"
# target_subtypes = {"Crane", "ContainerForklift"}
target_subtypes = {"WheelCrane"}

# 遍历 label 目录
for date_folder in os.listdir(label_src_root):
    date_label_path = os.path.join(label_src_root, date_folder)
    date_sample_path = os.path.join(sample_src_root, date_folder)
    if not os.path.isdir(date_label_path):
        continue

    for sub_folder in os.listdir(date_label_path):
        label_folder_path = os.path.join(date_label_path, sub_folder)
        sample_folder_path = os.path.join(date_sample_path, sub_folder)
        if not os.path.isdir(label_folder_path) or not os.path.isdir(sample_folder_path):
            continue

        # 收集 lidar 和 camera 样本列表（以文件名为时间戳）
        lidar_root = os.path.join(sample_folder_path, 'lidar')
        camera_root = os.path.join(sample_folder_path, 'camera')
        lidar_files = collect_files_with_timestamps(lidar_root, extensions=('.pcd',))
        # camera_files = collect_files_with_timestamps(camera_root, extensions=('.jpeg',))
        camera_files = collect_files_with_timestamps(camera_root, extensions=('.jpeg', '.jpg'))
        
        # ✅ 添加这个标志变量，判断是否有 label 被复制
        any_label_copied = False

        for fname in os.listdir(label_folder_path):
            if not fname.endswith('.json'):
                continue
            label_path = os.path.join(label_folder_path, fname)

            try:
                with open(label_path, 'r') as f:
                    data = json.load(f)
            except Exception as e:
                print(f"Failed to read {label_path}: {e}")
                continue

            if not any(obj.get("subtype") in target_subtypes for obj in data):
                continue

            # ✅ 复制 label 文件
            rel_label_path = os.path.relpath(label_path, label_src_root)
            dst_label_path = os.path.join(dst_root, "label", rel_label_path)
            os.makedirs(os.path.dirname(dst_label_path), exist_ok=True)
            shutil.copy2(label_path, dst_label_path)
            print(f"Copied label: {label_path} -> {dst_label_path}")
            
            # ✅ 同时复制 sample_folder_path 下的 .json 和 .txt 文件
            for item in os.listdir(sample_folder_path):
                if item.endswith('.json') or item.endswith('.txt'):
                    item_src = os.path.join(sample_folder_path, item)
                    item_dst = os.path.join(dst_root, "sample", date_folder, sub_folder, item)
                    os.makedirs(os.path.dirname(item_dst), exist_ok=True)
                    shutil.copy2(item_src, item_dst)
                    print(f"Copied extra file: {item_src} -> {item_dst}")

            # 提取时间戳
            try:
                ts = extract_timestamp(fname)
            except:
                continue

            # ✅ 复制 lidar 最近的 pcd 文件
            for sensor, files in lidar_files.items():
                match = find_nearest_by_timestamp_bisect(ts, files)
                if match:
                    dst_path = os.path.join(dst_root, "sample", date_folder, sub_folder, "lidar", sensor)
                    os.makedirs(dst_path, exist_ok=True)
                    shutil.copy2(match, dst_path)
                    print(f"Copied lidar: {match} -> {dst_path}")

            # ✅ 复制 camera 最近的 jpeg 文件
            for sensor, files in camera_files.items():
                match = find_nearest_by_timestamp_bisect(ts, files)
                if match:
                    dst_path = os.path.join(dst_root, "sample", date_folder, sub_folder, "camera", sensor)
                    os.makedirs(dst_path, exist_ok=True)
                    shutil.copy2(match, dst_path)
                    print(f"Copied camera: {match} -> {dst_path}")
                    
        # ✅ 如果有 label 被复制，则复制 extrinsics.json（子目录处理完之后）
        if any_label_copied:
            extrinsics_src = os.path.join(date_sample_path, "extrinsics.json")
            extrinsics_dst = os.path.join(dst_root, "sample", date_folder, "extrinsics.json")
            if os.path.exists(extrinsics_src):
                os.makedirs(os.path.dirname(extrinsics_dst), exist_ok=True)
                shutil.copy2(extrinsics_src, extrinsics_dst)
                print(f"Copied extrinsics (date-level): {extrinsics_src} -> {extrinsics_dst}")
