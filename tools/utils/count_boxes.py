import os
import json
import math
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# ✅ 计算均值和标准差（已存在）
def compute_mean_std(data_list):
    if not data_list:
        return 0.0, 0.0
    n = len(data_list)
    mean = sum(data_list) / n
    var = sum((x - mean) ** 2 for x in data_list) / n
    std = math.sqrt(var)
    return mean, std

# ✅ 新增：用于合并所有尺寸数据
def merge_label_dims(label_dims_list):
    merged = defaultdict(list)
    for label_dims in label_dims_list:
        for k, dims_list in label_dims.items():
            merged[k].extend(dims_list)
    return merged

# ✅ 新增：统计尺寸（lwh）的均值和标准差
def compute_lwh_stats(dim_list):
    if not dim_list:
        return (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)
    l_list, w_list, h_list = zip(*dim_list)
    l_mean, l_std = compute_mean_std(l_list)
    w_mean, w_std = compute_mean_std(w_list)
    h_mean, h_std = compute_mean_std(h_list)
    return (l_mean, l_std), (w_mean, w_std), (h_mean, h_std)

# ✅ 修改：增加提取 lwh 的逻辑
def count_boxes_in_folder(folder_path):
    total_boxes = 0
    label_counter = defaultdict(int)
    subtype_counter = defaultdict(int)
    label_pts = defaultdict(list)
    subtype_pts = defaultdict(list)
    label_dims = defaultdict(list)      # ✅ 新增：存放 label 尺寸
    subtype_dims = defaultdict(list)    # ✅ 新增：存放 subtype 尺寸

    for subdir, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".json"):
                json_path = os.path.join(subdir, file)
                try:
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            total_boxes += len(data)
                            for item in data:
                                label = item.get("label", "Unknown")
                                subtype = item.get("subtype", "Unknown")
                                num_pts = item.get("num_lidar_pts", None)

                                label_counter[label] += 1
                                subtype_counter[subtype] += 1

                                if num_pts is not None:
                                    label_pts[label].append(num_pts)
                                    subtype_pts[subtype].append(num_pts)

                                # ✅ 提取 lwh 尺寸字段
                                lwh = item.get("lwh", None)
                                if lwh and len(lwh) == 3:
                                    label_dims[label].append(lwh)
                                    subtype_dims[subtype].append(lwh)

                except Exception as e:
                    print(f"读取出错: {json_path}, 错误: {e}")
    
    return total_boxes, label_counter, subtype_counter, label_pts, subtype_pts, label_dims, subtype_dims

def merge_counters(counter_list):
    merged = defaultdict(int)
    for c in counter_list:
        for k, v in c.items():
            merged[k] += v
    return merged

def merge_label_pts(label_pts_list):
    merged = defaultdict(list)
    for label_pts in label_pts_list:
        for k, vlist in label_pts.items():
            merged[k].extend(vlist)
    return merged

# ✅ 主函数修改：合并并统计尺寸信息
def count_boxes_multiple_folders(root_folders):
    grand_total = 0
    all_label_counters = []
    all_subtype_counters = []
    all_label_pts = []
    all_subtype_pts = []
    all_label_dims = []     # ✅ 新增
    all_subtype_dims = []   # ✅ 新增

    for folder in root_folders:
        print(f"\n📁 正在统计文件夹: {folder}")
        total, label_counter, subtype_counter, label_pts, subtype_pts, label_dims, subtype_dims = count_boxes_in_folder(folder)
        grand_total += total
        all_label_counters.append(label_counter)
        all_subtype_counters.append(subtype_counter)
        all_label_pts.append(label_pts)
        all_subtype_pts.append(subtype_pts)
        all_label_dims.append(label_dims)      # ✅ 新增
        all_subtype_dims.append(subtype_dims)  # ✅ 新增

        print(f"  📦 标注框总数: {total}")
        print("  📊 按 label 分类:")
        for label, count in label_counter.items():
            print(f"    {label}: {count}")
        print("  📊 按 subtype 分类:")
        for subtype, count in subtype_counter.items():
            print(f"    {subtype}: {count}")

    print("\n============================")
    print("📈 所有文件夹合计统计结果")
    print("============================")
    print(f"📦 总标注框数: {grand_total}")

    total_labels = merge_counters(all_label_counters)
    total_subtypes = merge_counters(all_subtype_counters)
    merged_label_pts = merge_label_pts(all_label_pts)
    merged_subtype_pts = merge_label_pts(all_subtype_pts)
    merged_label_dims = merge_label_dims(all_label_dims)        # ✅
    merged_subtype_dims = merge_label_dims(all_subtype_dims)    # ✅

    print("📊 合并后 label:")
    for label, count in total_labels.items():
        print(f"  {label}: {count}")
    print("📊 合并后 subtype:")
    for subtype, count in total_subtypes.items():
        print(f"  {subtype}: {count}")

    print("\n📉 点数统计（每类 label）:")
    for label, pts_list in merged_label_pts.items():
        mean, std = compute_mean_std(pts_list)
        print(f"  {label}: 平均点数 = {mean:.2f}, 标准差 = {std:.2f}")

    print("\n📉 点数统计（每类 subtype）:")
    for subtype, pts_list in merged_subtype_pts.items():
        mean, std = compute_mean_std(pts_list)
        print(f"  {subtype}: 平均点数 = {mean:.2f}, 标准差 = {std:.2f}")

    # ✅ 尺寸统计输出
    print("\n📏 尺寸统计（每类 label）:")
    for label, dims in merged_label_dims.items():
        (l_mean, l_std), (w_mean, w_std), (h_mean, h_std) = compute_lwh_stats(dims)
        print(f"  {label}:")
        print(f"    长 l: 平均 = {l_mean:.2f}, 标准差 = {l_std:.2f}")
        print(f"    宽 w: 平均 = {w_mean:.2f}, 标准差 = {w_std:.2f}")
        print(f"    高 h: 平均 = {h_mean:.2f}, 标准差 = {h_std:.2f}")

    print("\n📏 尺寸统计（每类 subtype）:")
    for subtype, dims in merged_subtype_dims.items():
        (l_mean, l_std), (w_mean, w_std), (h_mean, h_std) = compute_lwh_stats(dims)
        print(f"  {subtype}:")
        print(f"    长 l: 平均 = {l_mean:.2f}, 标准差 = {l_std:.2f}")
        print(f"    宽 w: 平均 = {w_mean:.2f}, 标准差 = {w_std:.2f}")
        print(f"    高 h: 平均 = {h_mean:.2f}, 标准差 = {h_std:.2f}")

    print("\n📊 正在生成图表...")
    # plot_histograms(merged_label_pts, title_prefix="Label: ")
    # plot_mean_std_scatter(merged_label_pts, title="📉 各 label 的点数均值 vs 标准差")
    # plot_histograms(merged_subtype_pts, title_prefix="Subtype: ")
    # plot_mean_std_scatter(merged_subtype_pts, title="📉 各 subtype 的点数均值 vs 标准差")


def plot_histograms(pts_dict, title_prefix=""):
    for label, pts_list in pts_dict.items():
        if not pts_list:
            continue
        plt.figure(figsize=(8, 4))
        sns.histplot(pts_list, bins=50, kde=True)
        plt.title(f"{title_prefix}{label} - 点数分布 (num_lidar_pts)")
        plt.xlabel("num_lidar_pts")
        plt.ylabel("数量")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

def plot_mean_std_scatter(pts_dict, title):
    labels = []
    means = []
    stds = []
    for label, pts_list in pts_dict.items():
        if not pts_list:
            continue
        mean, std = compute_mean_std(pts_list)
        labels.append(label)
        means.append(mean)
        stds.append(std)

    plt.figure(figsize=(10, 6))
    plt.scatter(means, stds)

    for i, label in enumerate(labels):
        plt.annotate(label, (means[i], stds[i]), fontsize=8, alpha=0.7)

    plt.xlabel("平均点数")
    plt.ylabel("标准差")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    root_dirs = [
        # 'data/kl/v1.0-trainval/label/s1',
        # 'data/kl/v1.0-trainval/label/s2',
        # 'data/kl/v1.0-trainval/label/s3',
        # 'data/kl/v1.0-trainval/label/s5',
        # 'data/kl/v1.0-trainval/label/s6',
        # 'data/kl/v1.0-trainval/label/s7',
        # 'data/kl/v1.0-trainval/label/s9',
        # 'data/kl/v1.0-trainval/label/s11',
        
        'data/kl/v1.0-trainval/label/20250507',
        'data/kl/v1.0-trainval/label/20250521',
        'data/kl/v1.0-trainval/label/20250522',
        'data/kl/v1.0-trainval/label/20250515_20250516',
        'data/kl/v1.0-trainval/label/20250522_8',
        'data/kl/v1.0-trainval/label/20250523',
        'data/kl/v1.0-trainval/label/20250527',
        'data/kl/v1.0-trainval/label/20250604',
        'data/kl/v1.0-trainval/label/20250605',
        'data/kl/v1.0-trainval/label/20250606',
        'data/kl/v1.0-trainval/label/20250607',
        'data/kl/v1.0-trainval/label/20250609',
        'data/kl/v1.0-trainval/label/20250611',
        'data/kl/v1.0-trainval/label/20250612',
        'data/kl/v1.0-trainval/label/20250613',
        'data/kl/v1.0-trainval/label/20250618',
        'data/kl/v1.0-trainval/label/20250619',
        'data/kl/v1.0-trainval/label/20250620',
        'data/kl/v1.0-trainval/label/20250621',
        'data/kl/v1.0-trainval/label/20250624',
        'data/kl/v1.0-trainval/label/20250625',
        
        'data/kl/v1.0-trainval/label/20250627',
        'data/kl/v1.0-trainval/label/20250630',
        'data/kl/v1.0-trainval/label/20250702',
        'data/kl/v1.0-trainval/label/20250703',
        'data/kl/v1.0-trainval/label/20250705',
        
        'data/kl/v1.0-trainval/label/20250709',
        'data/kl/v1.0-trainval/label/20250709_chache',
        'data/kl/v1.0-trainval/label/20250710',
    ]
    count_boxes_multiple_folders(root_dirs)
