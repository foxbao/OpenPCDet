import argparse
import glob
from pathlib import Path


try:
    import open3d
    from visual_utils import open3d_vis_utils as V
    OPEN3D_FLAG = True
except:
    import mayavi.mlab as mlab
    from visual_utils import visualize_utils as V
    OPEN3D_FLAG = False

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None,dataset_mode='kitti'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

        self.dataset_mode=dataset_mode
        if dataset_mode=='kitti':
            self.feature_num=4
        elif dataset_mode=='nuscenes':
            self.feature_num=5
        elif dataset_mode=='kl':
            self.feature_num=6
        else:
            self.feature_num=4

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):

        file_name=self.sample_file_list[index]
        ext = Path(file_name).suffix
        if ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, self.feature_num)
        elif ext == '.npy':
            points = np.load(self.sample_file_list[index])
        elif ext == '.pcd':
            aaaa=1
        else:
            raise NotImplementedError
        print("Points shape:", points.shape)
        print("Min x:", points[:, 0].min())
        print("Max x:", points[:, 0].max())
        if self.dataset_mode=='kl':
            points = points[:, :4]

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        print("Points shape:", data_dict['points'].shape)
        print("Min x:", data_dict['points'][:, 0].min())
        print("Max x:", data_dict['points'][:, 0].max())
        return data_dict


def parse_config():
    # 创建一个参数解析器
    parser = argparse.ArgumentParser(description='arg parser')
    # 添加一个参数，指定配置文件
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    # 添加一个参数，指定点云数据文件或目录
    parser.add_argument('--data_path', type=str, default='000008.bin',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default='pv_rcnn_8369.pth', help='specify the pretrained model')
    parser.add_argument('--mode',type=str, default='kitti', help='specify the dataset')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), logger=logger,dataset_mode=args.mode
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):

            logger.info(f'Visualized sample index: \t{idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)

            # 过滤 pred_scores 大于 0.5 的预测
            mask = pred_dicts[0]['pred_scores'] > 0.5
            filtered_boxes = pred_dicts[0]['pred_boxes'][mask]
            filtered_scores = pred_dicts[0]['pred_scores'][mask]
            filtered_labels = pred_dicts[0]['pred_labels'][mask]


            V.draw_scenes(
                points=data_dict['points'][:, 1:], ref_boxes=filtered_boxes,
                ref_scores=filtered_scores, ref_labels=filtered_labels
            )

            if not OPEN3D_FLAG:
                mlab.show(stop=True)

    logger.info('Demo done.')


if __name__ == '__main__':
    main()
