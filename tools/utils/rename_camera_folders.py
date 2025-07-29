import os

def rename_camera_folders(root_dir):
    for subdir in os.listdir(root_dir):
        sub_path = os.path.join(root_dir, subdir)
        if not os.path.isdir(sub_path):
            continue

        camera_dir = os.path.join(sub_path, "camera")
        if not os.path.isdir(camera_dir):
            continue

        for folder_name in os.listdir(camera_dir):
            full_path = os.path.join(camera_dir, folder_name)
            if os.path.isdir(full_path) and folder_name.endswith("_raw_compressed"):
                new_name = folder_name.replace("_raw_compressed", "")
                new_path = os.path.join(camera_dir, new_name)
                print(f"Renaming: {full_path} -> {new_path}")
                os.rename(full_path, new_path)

# 示例调用
if __name__ == "__main__":
    # root_folder = "data/20250611/20250611_out"  # 替换为你的实际路径
    # root_folder="/home/baojiali/Downloads/disk1/data/lightwheel_data/sample/20250611"
    root_folder="/home/baojiali/Downloads/disk1/data/lightwheel_data/sample/20250621"
    # root_folder = "data/20250611/20250611_selected"  # 替换为你的实际路径
    
    rename_camera_folders(root_folder)
