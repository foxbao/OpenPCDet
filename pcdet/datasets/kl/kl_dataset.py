import copy
import numpy as np
# import os
import pickle
from pathlib import Path
from tqdm import tqdm
import open3d as o3d
from typing import List, Tuple
# import shutil
from ..dataset import DatasetTemplate
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from scipy.spatial.transform import Rotation as R
from .kl_dataset_utils import fill_trainval_infos
from .kl import KL
from pcdet.utils import common_utils
import random

def check_nan_inf(arr):
    """
    检查数组中是否有 NaN 或 Inf，并打印其位置。
    
    :param arr: numpy 数组
    :return: True（如果有 NaN 或 Inf），否则 False
    """
    
    if arr.dtype.kind in {'U', 'S', 'O'}:
        # 判断是否是字符串类型
        if np.issubdtype(arr.dtype, np.str_) or np.issubdtype(arr.dtype, np.object_):
            # print("Array contains string data, skipping NaN/Inf check.")
            return False
        
    has_nan = np.isnan(arr)
    has_inf = np.isinf(arr)

    if np.any(has_nan):
        print("Found NaN at indices:", np.argwhere(has_nan))

    if np.any(has_inf):
        print("Found Inf at indices:", np.argwhere(has_inf))

    return np.any(has_nan) or np.any(has_inf)

def read_pcd_with_intensity(pcd_path):
    # 读取文件头
    with open(pcd_path, 'rb') as f:
        header = []
        while True:
            line = f.readline().decode('utf-8').strip()
            header.append(line)
            if line.startswith('DATA'):
                break

    # 解析字段、类型、大小
    fields, size, type_ = None, None, None
    for line in header:
        if line.startswith('FIELDS'):
            fields = line.split()[1:]
        elif line.startswith('SIZE'):
            size = list(map(int, line.split()[1:]))
        elif line.startswith('TYPE'):
            type_ = line.split()[1:]

    if fields is None or size is None or type_ is None:
        raise ValueError("Invalid PCD header: missing FIELDS/SIZE/TYPE")

    if not len(fields) == len(size) == len(type_):
        raise ValueError("FIELDS/SIZE/TYPE length mismatch")

    # 构建 dtype：根据 TYPE 和 SIZE 推断
    def get_numpy_dtype(t, s):
        if t == 'F':
            if s == 4:
                return np.float32
            elif s == 8:
                return np.float64
        elif t == 'U':
            if s == 1:
                return np.uint8
            elif s == 2:
                return np.uint16
            elif s == 4:
                return np.uint32
        elif t == 'I':
            if s == 1:
                return np.int8
            elif s == 2:
                return np.int16
            elif s == 4:
                return np.int32
        raise ValueError(f"Unsupported TYPE/SIZE combination: TYPE={t}, SIZE={s}")

    dtype = np.dtype([(f, get_numpy_dtype(t, s)) for f, t, s in zip(fields, type_, size)])

    # 计算数据起始位置
    data_offset = len('\n'.join(header)) + 1
    data = np.fromfile(pcd_path, dtype=dtype, offset=data_offset)

    # 检查字段存在
    required = {'x', 'y', 'z', 'intensity', 'ring'}
    if not required.issubset(data.dtype.names):
        raise ValueError(f"Missing required fields. Expected at least: {required}")

    # 构造输出数据（自动判断是否含有 timestamp_2us）
    base_fields = ['x', 'y', 'z', 'intensity', 'ring']
    base_fields = ['x', 'y', 'z', 'intensity']
    arrs = [data[f].astype(np.float32) for f in base_fields]

    # if 'timestamp_2us' in data.dtype.names:
    #     arrs.append(data['timestamp_2us'].astype(np.float32))

    all_data = np.vstack(arrs).T

    # 过滤含 NaN 的点
    valid_mask = ~np.isnan(all_data).any(axis=1)
    return all_data[valid_mask]


def read_pc(pc_file, verbose=False):
    """
    读取点云（支持.bin/.pcd），自动过滤NaN/Inf
    
    Args:
        pc_file: 文件路径（Path对象或字符串）
        verbose: 是否打印调试信息
        
    Returns:
        np.ndarray: (N, 4)的合法点云数据 [x, y, z, intensity]
    """
    pc_file = Path(pc_file)
    if not pc_file.exists():
        raise FileNotFoundError(f"Point cloud file not found: {pc_file}")

    try:
        if pc_file.suffix == '.bin':
            dtype = np.dtype([
                ('x', np.float32), ('y', np.float32), ('z', np.float32),
                ('intensity', np.float32), ('ring', np.float32),  # 根据实际格式调整
                ('timestamp_2us', np.float32)
            ])
            data = np.fromfile(pc_file, dtype=dtype)
            points = np.vstack((data['x'], data['y'], data['z'], data['intensity'])).T
            
        elif pc_file.suffix == '.pcd':
            points = read_pcd_with_intensity(pc_file)
            
        else:
            raise ValueError(f"Unsupported file format: {pc_file.suffix}")

        # 二次检查（防止上游未处理的情况）
        valid_mask = np.isfinite(points).all(axis=1)
        if np.any(~valid_mask):
            points = points[valid_mask]
            if verbose:
                print(f"Secondary filtering: Removed {np.sum(~valid_mask)} invalid points")

        # 空数据检查
        if len(points) == 0:
            raise ValueError(f"Empty point cloud after filtering: {pc_file}")

        points = points[np.max(np.abs(points[:, :3]), axis=1) < 1e3]  # 保留合理值,防止数值溢出
        return points

    except Exception as e:
        raise RuntimeError(f"Error reading {pc_file}: {str(e)}")



def kl_eval(eval_det_annos, eval_gt_annos):
    pass

class KLDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        root_path = (root_path if root_path is not None else Path(dataset_cfg.DATA_PATH)) / dataset_cfg.VERSION
        super().__init__(dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger)
        self.infos = []
        self.camera_config = self.dataset_cfg.get('CAMERA_CONFIG', None)
        if self.camera_config is not None:
            self.use_camera = self.camera_config.get('USE_CAMERA', True)
            self.camera_image_config = self.camera_config.IMAGE
        else:
            self.use_camera = False
        self.include_kl_data(self.mode)
        
        self.filter_gt_by_points = self.dataset_cfg.get('POINT_FILTER', {}).get('ENABLED', False)
        self.class_min_points_dict = self.dataset_cfg.get('POINT_FILTER', {}).get('FILTER_MIN_POINTS_BY_CLASS', {})
        
        self.intensity_filter_cfg = self.dataset_cfg.get('INTENSITY_FILTER', {})
        self.use_intensity_filter = self.intensity_filter_cfg.get('ENABLED', False)
        self.intensity_threshold = self.intensity_filter_cfg.get('THRESHOLD', 0.0)

    def include_kl_data(self, mode):
        self.logger.info('Loading KL dataset')
        kl_infos = []

        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.root_path / info_path
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                kl_infos.extend(infos)

        self.infos.extend(kl_infos)
        self.logger.info('Total samples for KL dataset: %d' % (len(kl_infos)))


    def get_merged_lidar(self,index,use_extrinsic=True)->np.ndarray:
        
        point_clouds = []
        info = self.infos[index]
        # lidar_names=['helios_front_left','helios_rear_right']
        # lidar_extrinsic_names=['Tx_baselink_lidar_helios_front_left','Tx_baselink_lidar_helios_rear_right']

        extrinsic_names={}
        extrinsic_names['helios_front_left']='Tx_baselink_lidar_helios_front_left'
        extrinsic_names['helios_rear_right']='Tx_baselink_lidar_helios_rear_right'
        extrinsic_names['bp_front_left']='Tx_baselink_lidar_bp_front_left' # 向下补盲
        # extrinsic_names['bp_front_right']='Tx_baselink_lidar_bp_front_right' #向上补盲
        # extrinsic_names['bp_rear_left']='Tx_baselink_lidar_bp_rear_left' #向上补盲
        extrinsic_names['bp_rear_right']='Tx_baselink_lidar_bp_rear_right' #向下补盲
        
        # lidar_configurations=[]
        # lidar_configurations.append({"lidar_name": "helios_front_left","lidar_extrinsic_name": "Tx_baselink_lidar_helios_front_left"})
        # lidar_configurations.append({"lidar_name": "helios_rear_right","lidar_extrinsic_name": "Tx_baselink_lidar_helios_rear_right"})
        
        for lidar_name, lidar_extrinsic_name in extrinsic_names.items():
            lidar_path = self.root_path / info['lidars'][lidar_name]
            points=read_pc(lidar_path)
            
            # ⭐ 如果开启强度过滤
            if self.use_intensity_filter:
                intensity = points[:, 3]
                mask = intensity >= self.intensity_threshold
                points = points[mask]
            # times = np.zeros((points.shape[0], 1))
            # points = np.concatenate((points, times), axis=1)
            if use_extrinsic:
                lidar_extrinsic=info['sensor_extrinsics'][lidar_extrinsic_name]
                points=self.transform_points(points, lidar_extrinsic)
            point_clouds.append(points)
                
        if point_clouds:  # 如果列表不为空
            merged_point_cloud = np.concatenate(point_clouds, axis=0)

        # import open3d as o3d
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(merged_point_cloud[:, :3])
        # o3d.io.write_point_cloud("output_with_intensity.pcd", pcd, write_ascii=True)
        return merged_point_cloud
    
    def get_lidar(self, index,lidar_name='helios_front_left'):
        info = self.infos[index]
        lidar_path = self.root_path / info['lidars'][lidar_name]

        points=read_pc(lidar_path)
        times = np.zeros((points.shape[0], 1))
        points = np.concatenate((points, times), axis=1)
        return points

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.infos) * self.total_epochs

        return len(self.infos)
    

    
    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.infos)

        info = copy.deepcopy(self.infos[index])

        points=self.get_merged_lidar(index,True)
        # check_nan_inf(points)
        input_dict = {
            'points': points,
            'frame_id': Path(info['lidars']['helios_front_left']).stem,
            'metadata': {'token': info['token']}
        }

        if 'annos' in info:
            annos = info['annos']
            gt_names = annos['name']
            gt_boxes_lidar = annos['gt_boxes_lidar']
            gt_num_lidar_pts=annos['num_lidar_pts']
            
            # ⭐ 点数过滤逻辑开始 ⭐
            if getattr(self, 'filter_gt_by_points', False):
                keep_mask = np.ones(len(gt_names), dtype=bool)
                for i in range(len(gt_names)):
                    cls = gt_names[i]
                    min_pts = self.class_min_points_dict.get(cls, 0)
                    if gt_num_lidar_pts[i] < min_pts:
                        keep_mask[i] = False

                gt_names = gt_names[keep_mask]
                gt_boxes_lidar = gt_boxes_lidar[keep_mask]
                gt_num_lidar_pts = gt_num_lidar_pts[keep_mask]
            # ⭐ 点数过滤逻辑结束 ⭐

            input_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_boxes_lidar
                # 'gt_num_lidar_pts':gt_num_lidar_pts
            })

        if self.use_camera:
            input_dict = self.load_camera_info(input_dict, info)

        data_dict = self.prepare_data(data_dict=input_dict)

        if self.dataset_cfg.get('SET_NAN_VELOCITY_TO_ZEROS', False) and 'gt_boxes' in info:
            gt_boxes = data_dict['gt_boxes']
            gt_boxes[np.isnan(gt_boxes)] = 0
            data_dict['gt_boxes'] = gt_boxes

        # if not self.dataset_cfg.PRED_VELOCITY and 'gt_boxes' in data_dict:
        #     data_dict['gt_boxes'] = data_dict['gt_boxes'][:, [0, 1, 2, 3, 4, 5, 6, -1]]
        data_dict['timestamp']=info['timestamp']
        helios_front_left_path=info['lidars']['helios_front_left']
        parts = helios_front_left_path.split('/')
        sample_index = parts.index('sample')
        folder = '/'.join(parts[sample_index+1:sample_index+3])
        data_dict['folder']=folder

        return data_dict
    
    def evaluation(self, det_annos, class_names, **kwargs):
        if 'annos' not in self.infos[0].keys():
            return 'No ground-truth boxes for evaluation', {}

        def kitti_eval(eval_det_annos, eval_gt_annos, map_name_to_kitti):
            from ..kitti.kitti_object_eval_python import eval as kitti_eval
            from .kl_object_eval_python import eval as kl_eval
            from ..kitti import kitti_utils

            kitti_utils.transform_annotations_to_kitti_format(eval_det_annos, map_name_to_kitti=map_name_to_kitti)
            kitti_utils.transform_annotations_to_kitti_format(
                eval_gt_annos, map_name_to_kitti=map_name_to_kitti,
                info_with_fakelidar=self.dataset_cfg.get('INFO_WITH_FAKELIDAR', False)
            )
            kitti_class_names = [map_name_to_kitti[x] for x in class_names]
            kitti_class_names_unique = list(set(kitti_class_names))
            ap_result_str, ap_dict = kl_eval.get_kl_eval_result(
                gt_annos=eval_gt_annos, dt_annos=eval_det_annos, current_classes=kitti_class_names_unique
            )
            return ap_result_str, ap_dict

        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.infos]
        # with open('eval_det_annos.pkl', 'wb') as f:  # 'wb' 表示以二进制写入模式打开文件
        #     pickle.dump(eval_det_annos, f)
        
        # with open('eval_gt_annos.pkl', 'wb') as f:  # 'wb' 表示以二进制写入模式打开文件
        #     pickle.dump(eval_gt_annos, f)

        map_name_to_kitti = {
            # 行人相关
            'Pedestrian': 'Pedestrian',
            'Cone': 'Cone',  # 锥桶尺寸接近行人
            
            # 轿车类
            'Car': 'Car',
            'IGV-Full': 'IGV-Full',     # 智能引导车
            'IGV-Empty': 'IGV-Empty',    # 空载引导车
            'OtherVehicle': 'OtherVehicle', # 其他车辆
            
            # 卡车类（独立类别）
            'Truck': 'Truck',                  # 卡车
            'Trailer-Empty': 'Trailer-Empty',          # 空拖车
            'Trailer-Full': 'Trailer-Full',           # 满载拖车
            'Lorry': 'Lorry',                  # 货车
            'ContainerForklift': 'ContainerForklift',      # 集装箱叉车
            
            # 工程车辆
            'Crane': 'Crane',                    # 起重机
            'Forklift': 'Forklift',                 # 普通叉车
            'ConstructionVehicle': 'ConstructionVehicle',       # 工程车
            'WheelCrane':'WheelCrane'
        }

        if kwargs['eval_metric'] == 'kitti':
            ap_result_str, ap_dict = kitti_eval(eval_det_annos, eval_gt_annos, map_name_to_kitti)
        else:
            raise NotImplementedError

        return ap_result_str, ap_dict
    
    def transform_points(self,point_cloud, extrinsic):
        # 提取平移向量
        translation = np.array(extrinsic[:3])  # [Tx, Ty, Tz]

        # 提取四元数
        quaternion = np.array(extrinsic[3:])  # [qx, qy, qz, qw]
        rotation_matrix = R.from_quat(quaternion).as_matrix()
        positions = point_cloud[:, :3]
        rotated_positions = np.dot(positions, rotation_matrix.T)
        transformed_positions = rotated_positions + translation
        point_cloud[:, :3] = transformed_positions
        return point_cloud


    def create_groundtruth_database(self,used_classes=None):
        import torch

        database_save_path = self.root_path / f'gt_database'
        db_info_save_path = self.root_path / f'kl_dbinfos.pkl'

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

                
        for idx in tqdm(range(len(self.infos))):
            # print('gt_database sample: %d/%d' % (idx + 1, len(self.infos)))
            sample_idx = idx
            info = self.infos[idx]
            points=self.get_merged_lidar(idx,True)
            annos = info['annos']
            names = annos['name']
            difficulty = annos['difficulty']
            gt_boxes = annos['gt_boxes_lidar']
            # gt_boxes = info['gt_boxes']
            # gt_names = info['gt_names']
            # tranform the points from lidar coordinate to vehicle coordinate
            box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                torch.from_numpy(points[:, 0:3]).unsqueeze(dim=0).float().cuda(),
                torch.from_numpy(gt_boxes[:, 0:7]).unsqueeze(dim=0).float().cuda()
            ).long().squeeze(dim=0).cpu().numpy()

            for i in range(gt_boxes.shape[0]):
                filename = '%s_%s_%d.bin' % (sample_idx, names[i], i)
                filepath = database_save_path / filename
                gt_points = points[box_idxs_of_pts == i]

                gt_points[:, :3] -= gt_boxes[i, :3]
                with open(filepath, 'w') as f:
                    gt_points.tofile(f)

                if (used_classes is None) or names[i] in used_classes:
                    db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                    db_info = {'name': names[i], 'path': db_path, 'image_idx': sample_idx, 'gt_idx': i,
                                'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0],'difficulty':0}
                    if names[i] in all_db_infos:
                        all_db_infos[names[i]].append(db_info)
                    else:
                        all_db_infos[names[i]] = [db_info]
        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)


def split_samples(samples):
    total_files = len(samples)
    train_size = int(total_files * 0.9)
    val_size = int(total_files * 0.09)
    split_samples = {
        'train': samples[:train_size],
        'val': samples[train_size:train_size + val_size],
        'test': samples[train_size + val_size:]
    }
    train_samples=split_samples['train']
    val_scenes=split_samples['val']
    test_scenes=split_samples['test']
    return train_samples,val_scenes,test_scenes

def analyze_kl_infos(version, data_path, save_path,with_cam=False):
    import json
    from . import kl_dataset_utils
    from .kl_dataset_utils import convert_json_to_annotations
    import tqdm
    import numpy as np
    from collections import Counter
    
    counter = Counter()
    kl = KL(version=version, dataroot=data_path, verbose=True)
    samples=kl.get_all_sample()
    print('total labelled samples:',len(samples))
    for sample in tqdm.tqdm(kl.samples, desc='create_info', dynamic_ncols=True):
        with open(sample['label'], 'r', encoding='utf-8') as f:
            data = json.load(f)
            annotations=convert_json_to_annotations(data)
            name=annotations['name']
            counter.update(name.tolist()) 

    # progress_bar.close()

    # 打印统计结果
    print('类别统计结果:')
    total_count = 0
    for cls_name, count in counter.items():
        print(f'{cls_name}: {count}')
        total_count += count

    print(f'\n所有类别的总数：{total_count}')
    

def create_kl_infos(version, data_path, save_path,with_cam=False):
    from . import kl_dataset_utils
    kl = KL(version=version, dataroot=data_path, verbose=True)
    samples=kl.get_all_sample()
    random.shuffle(samples)

    train_samples,val_samples,test_samples=split_samples(samples)
    train_samples = {d['token'] for d in train_samples}
    val_samples = {d['token'] for d in val_samples}
    test_samples = {d['token'] for d in test_samples}
    
    train_kl_infos,val_kl_infos,test_kl_infos=kl_dataset_utils.fill_trainval_infos(kl,train_samples,val_samples,test_samples)

    print('train sample: %d, val sample: %d, test sample: %d' % (len(train_kl_infos), len(val_kl_infos),len(test_kl_infos)))
    with open(save_path /version/ f'kl_infos_train.pkl', 'wb') as f:
        pickle.dump(train_kl_infos, f)
    with open(save_path /version/ f'kl_infos_val.pkl', 'wb') as f:
        pickle.dump(val_kl_infos, f)
    with open(save_path /version/ f'kl_infos_test.pkl', 'wb') as f:
        pickle.dump(test_kl_infos, f)


if __name__ == '__main__':

    import yaml
    import argparse
    from pathlib import Path
    from easydict import EasyDict

    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config of dataset')
    parser.add_argument('--func', type=str, default='create_kl_infos', help='')
    parser.add_argument('--version', type=str, default='v1.0-trainval', help='')
    parser.add_argument('--with_cam', action='store_true', default=False, help='use camera or not')
    args = parser.parse_args()
    
    
    if args.func == 'analyze_kl_infos':
        dataset_cfg = EasyDict(yaml.safe_load(open(args.cfg_file)))
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        dataset_cfg.VERSION = args.version
        analyze_kl_infos(
            version=dataset_cfg.VERSION,
            data_path=ROOT_DIR / 'data' / 'kl',
            save_path=ROOT_DIR / 'data' / 'kl',
            with_cam=args.with_cam
        )

    if args.func == 'create_kl_infos':
        dataset_cfg = EasyDict(yaml.safe_load(open(args.cfg_file)))
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        dataset_cfg.VERSION = args.version
        data_path = Path(dataset_cfg.DATA_PATH)  # 转换为 Path
        
        last_two_parts = data_path.parts[-2:]   # 取最后两部分，如 ('data', 'kl')
        folder=last_two_parts[0]
        name=last_two_parts[1]
        # 生成pkl文件
        create_kl_infos(
            version=dataset_cfg.VERSION,
            data_path=ROOT_DIR / folder / name,
            save_path=ROOT_DIR /folder / name,
            with_cam=args.with_cam
        )

    kl_dataset = KLDataset(
        dataset_cfg=dataset_cfg, class_names=None,
        root_path=ROOT_DIR / folder / name,
        logger=common_utils.create_logger(), training=True
    )

    kl_dataset.create_groundtruth_database()