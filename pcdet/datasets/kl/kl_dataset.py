import copy
import json
import numpy as np
import os
import pickle
from pathlib import Path
from tqdm import tqdm

from ..dataset import DatasetTemplate
# print(__name__)
# print(__package__)
class KLDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        super().__init__(dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger)



def create_kl_infos(version, data_path, save_path, max_sweeps=10, with_cam=False):
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils import splits
    from . import kl_utils
    data_path = data_path / version
    save_path = save_path / version

    assert version in ['v1.0-trainval', 'v1.0-test', 'v1.0-mini']
    if version == 'v1.0-trainval':
        train_scenes = splits.train
        val_scenes = splits.val
    elif version == 'v1.0-test':
        train_scenes = splits.test
        val_scenes = []
    elif version == 'v1.0-mini':
        train_scenes = splits.mini_train
        val_scenes = splits.mini_val
    else:
        raise NotImplementedError

    nusc = NuScenes(version=version, dataroot=data_path, verbose=True)
    available_scenes = kl_utils.get_available_scenes(nusc)
    available_scene_names = [s['name'] for s in available_scenes]
    train_scenes = list(filter(lambda x: x in available_scene_names, train_scenes))
    val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
    train_scenes = set([available_scenes[available_scene_names.index(s)]['token'] for s in train_scenes])
    val_scenes = set([available_scenes[available_scene_names.index(s)]['token'] for s in val_scenes])

    print('%s: train scene(%d), val scene(%d)' % (version, len(train_scenes), len(val_scenes)))

    train_nusc_infos, val_nusc_infos = kl_utils.fill_trainval_infos(
        data_path=data_path, nusc=nusc, train_scenes=train_scenes, val_scenes=val_scenes,
        test='test' in version, max_sweeps=max_sweeps, with_cam=with_cam
    )

    if version == 'v1.0-test':
        print('test sample: %d' % len(train_nusc_infos))
        with open(save_path / f'nuscenes_infos_{max_sweeps}sweeps_test.pkl', 'wb') as f:
            pickle.dump(train_nusc_infos, f)
    else:
        print('train sample: %d, val sample: %d' % (len(train_nusc_infos), len(val_nusc_infos)))
        with open(save_path / f'nuscenes_infos_{max_sweeps}sweeps_train.pkl', 'wb') as f:
            pickle.dump(train_nusc_infos, f)
        with open(save_path / f'nuscenes_infos_{max_sweeps}sweeps_val.pkl', 'wb') as f:
            pickle.dump(val_nusc_infos, f)



def save_pcd_as_bin(pcd_file, bin_file):
    """将 pcd 文件保存为 pcd.bin 文件"""
    # 使用 open3d 读取 pcd 文件
    import open3d as o3d
    pcd = o3d.io.read_point_cloud(pcd_file)
    points = np.asarray(pcd.points,dtype=np.float32)  # 点云的坐标 (N, 3)
    
    # 如果点云数据是 4 维（例如包含强度信息），也可以这样处理
    # points = np.asarray(pcd.points)  # (N, 3)
    # intensity = np.asarray(pcd.colors)  # (N, 1) 可能需要根据文件类型来处理
    # 去掉包含 NaN 的点
    valid_points = points[~np.isnan(points).any(axis=1)]  # 删除包含 NaN 的点
    # 将点云数据保存为 .bin 文件
    # 将过滤后的点云数据保存为 .bin 文件
    valid_points.tofile(bin_file)

def generate_infos(root_dir, save_path, split):
    """生成 infos 文件"""
    import glob

    infos = []
    pcd_files = glob.glob(os.path.join(root_dir, split, '*.pcd'))
    for idx,pcd_file in enumerate(pcd_files):
        
        base = os.path.basename(pcd_file)
        pcd_bin_file = os.path.join(root_dir, split, base.replace('.pcd', '.pcd.bin'))
        save_pcd_as_bin(pcd_file, pcd_bin_file)
        json_file = os.path.join(root_dir, split, base.replace('.pcd', '.json'))
        if os.path.exists(json_file):
            infos.append({
                'token':idx,
                'pcd_file': os.path.relpath(pcd_bin_file, root_dir),
                'ann_file': os.path.relpath(json_file, root_dir),
                'lidar2ego_rotation': [0, 0, 0,0],
                'lidar2ego_translation': [0, 0, 0],
                'ego2global_rotation': [0, 0, 0,0],
                'ego2global_translation': [0, 0, 0]
            })
    data=dict()
    data['infos']=infos
    data['metadata']={'version':"v1.0-mini"}
    with open(os.path.join(save_path, f'{split}_infos.json'), 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == '__main__':
    root_dir = 'data/bao3d/'  # 根据实际路径调整
    save_path = 'data/bao3d/'
    for split in ['train', 'val', 'test']:
        generate_infos(root_dir, save_path, split)
    print("Infos files generated successfully.")