import os
import numpy as np

def process_pointcloud_folder(input_dir, output_dir, z_offset=-1.6):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    bin_files = [f for f in os.listdir(input_dir) if f.endswith('.bin')]

    for bin_file in bin_files:
        input_path = os.path.join(input_dir, bin_file)
        output_path = os.path.join(output_dir, bin_file)

        # 加载点云：默认每个点是[x, y, z, intensity]
        points = np.fromfile(input_path, dtype=np.float32).reshape(-1, 4)
        points[:, 2] += z_offset  # z 减去 1.6

        # 保存处理后的点云
        points.tofile(output_path)

    print(f"✅ 所有点云已处理完成，保存于：{output_dir}")

if __name__ == "__main__":
    input_folder = "data/output_pcd_20250521_no_cut"
    # input_folder = "data/out_chenxu_merged"
    output_folder = input_folder+"_z_offset"
    process_pointcloud_folder(input_folder, output_folder,z_offset=-1.54)
