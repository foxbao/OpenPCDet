import os
import json

# ==== 配置项 ====
z_offset = 1.5  # 要加的偏移值
# sample_dir = "/home/baojiali/Downloads/public_code/OpenPCDet/data/kl/v1.0-trainval/sample"
# label_dir = "/home/baojiali/Downloads/public_code/OpenPCDet/data/kl/v1.0-trainval/label"

sample_dir = "/home/baojiali/Downloads/public_code/OpenPCDet/data/lightwheel/v1.0-trainval/sample"
label_dir = "/home/baojiali/Downloads/public_code/OpenPCDet/data/lightwheel/v1.0-trainval/label"

def process_extrinsics_json(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)

    modified = False
    for key, value in data.items():
        if isinstance(value, list) and len(value) >= 3:
            old_val = value[2]
            value[2] += z_offset
            modified = True
            print(f"[extrinsics] {file_path}: {key} z {old_val:.3f} -> {value[2]:.3f}")

    if modified:
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)

def process_label_json(file_path):
    with open(file_path, "r") as f:
        try:
            data = json.load(f)
        except Exception as e:
            print(f"[label] Failed to parse {file_path}: {e}")
            return

    modified = False
    for obj in data:
        if "xyz" in obj and isinstance(obj["xyz"], list) and len(obj["xyz"]) == 3:
            old_z = obj["xyz"][2]
            obj["xyz"][2] += z_offset
            modified = True
            print(f"[label] {file_path}: z {old_z:.3f} -> {obj['xyz'][2]:.3f}")

    if modified:
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)

def main():
    # 处理 sample/extrinsics.json
    for s_folder in os.listdir(sample_dir):
        s_path = os.path.join(sample_dir, s_folder)
        if not os.path.isdir(s_path):
            continue

        for scene in os.listdir(s_path):
            scene_path = os.path.join(s_path, scene)
            extrinsics_path = os.path.join(scene_path, "extrinsics.json")

            if os.path.exists(extrinsics_path):
                process_extrinsics_json(extrinsics_path)

    # 处理 label/*.json
    for s_folder in os.listdir(label_dir):
        s_path = os.path.join(label_dir, s_folder)
        if not os.path.isdir(s_path):
            continue

        for scene in os.listdir(s_path):
            scene_path = os.path.join(s_path, scene)
            if not os.path.isdir(scene_path):
                continue

            for fname in os.listdir(scene_path):
                if fname.endswith(".json") and fname != "extrinsics.json":
                    fpath = os.path.join(scene_path, fname)
                    process_label_json(fpath)

if __name__ == "__main__":
    main()
