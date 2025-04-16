import argparse
import glob
from pathlib import Path
import os
import open3d as o3d
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
from visual_utils.visualize_tools import offscreen_visualization_array,visualization_array_pyvista,class_names
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from pcdet.datasets import build_dataloader
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from tqdm import tqdm
from visual_utils.open3d_vis_utils import box_colormap
class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None,dataset_mode='kl'):
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
            self.feature_num=4
        else:
            self.feature_num=4

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):

        file_name=self.sample_file_list[index]
        ext = Path(file_name).suffix
        if ext == '.bin':
            points = np.fromfile(file_name, dtype=np.float32).reshape(-1, self.feature_num)
        elif ext == '.npy':
            points = np.load(file_name)
        elif ext == '.pcd':
            points = np.loadtxt(file_name, skiprows=11)
            # xyz = data[:, :3]
            # intensity = data[:, 3]
        else:
            raise NotImplementedError
        # print("Points shape:", points.shape)
        # print("Min x:", points[:, 0].min())
        # print("Max x:", points[:, 0].max())
        if self.dataset_mode=='kl':
            points = points[:, :self.feature_num]

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        # print("Points shape:", data_dict['points'].shape)
        # print("Min x:", data_dict['points'][:, 0].min())
        # print("Max x:", data_dict['points'][:, 0].max())
        return data_dict



def SaveBoxPred(boxes: list, file_name: str):
    """
    将9维检测框数据完整写入txt文件
    格式: x1 y1 x2 y2 dim5 dim6 dim7 class_id score

    参数:
        boxes: 形状为[N, 9]的列表，每个bbox包含9个数值
        file_name: 输出文件路径
    """

    if isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().numpy()
    try:
        # 自动创建目录
        os.makedirs(os.path.dirname(file_name) or ".", exist_ok=True)
        
        with open(file_name, 'w') as f:
            for box in boxes:
                if len(box) != 9:
                    print(f"跳过无效数据: 需要9维, 实际得到{len(box)}维")
                    continue
                f.write(" ".join(map(str, box)) + "\n")
                
        # print(f"检测框已保存至: {file_name}")
    except Exception as e:
        print(f"保存失败: {str(e)}")


def parse_config():

    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')

    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--local_rank', type=int, default=None, help='local rank for distributed training')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')
    parser.add_argument('--max_waiting_mins', type=int, default=30, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--eval_tag', type=str, default='default', help='eval tag for this experiment')
    parser.add_argument('--eval_all', action='store_true', default=False, help='whether to evaluate all checkpoints')
    parser.add_argument('--ckpt_dir', type=str, default=None, help='specify a ckpt directory to be evaluated if needed')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='save the demo result to image')
    parser.add_argument('--infer_time', action='store_true', default=False, help='calculate inference latency')
    parser.add_argument('--data_path', type=str, default=None,help='specify the point cloud data file or directory')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    np.random.seed(1024)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg

def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')

    if args.infer_time:
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    if args.launcher == 'none':
        dist_test = False
        total_gpus = 1
    else:
        if args.local_rank is None:
            args.local_rank = int(os.environ.get('LOCAL_RANK', '0'))

        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_test = True

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        args.batch_size = args.batch_size // total_gpus

    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_output_dir = output_dir / 'eval'

    # 单文件输入模式
    if args.data_path:
        demo_dataset = DemoDataset(
            dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
            root_path=Path(args.data_path),  logger=logger
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
                print('detected ',pred_dicts[0]['pred_boxes'].size()[0],' objects')
                
                if args.save_to_file:
                    V.save_scenes(
                    points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
                    ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
                )
                else:
                    V.draw_scenes(
                        points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
                        ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
                    )

                if not OPEN3D_FLAG:
                    mlab.show(stop=True)
    # kl数据集的eval所有文件模式
    else:
        test_set, test_loader, sampler = build_dataloader(
            dataset_cfg=cfg.DATA_CONFIG,
            class_names=cfg.CLASS_NAMES,
            batch_size=1,
            dist=dist_test, workers=args.workers, logger=logger, training=False
        )
        

        model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
        model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
        model.cuda()
        model.eval()
        with torch.no_grad():
            for idx, batch_dict in enumerate(tqdm(test_loader)):
                load_data_to_gpu(batch_dict)
                pred_dicts, _ = model.forward(batch_dict)
                mask = pred_dicts[0]['pred_scores'] > 0.01
                filtered_boxes = pred_dicts[0]['pred_boxes'][mask]
                filtered_scores = pred_dicts[0]['pred_scores'][mask]
                filtered_labels = pred_dicts[0]['pred_labels'][mask]
                # filtered_boxes = pred_dicts[0]['pred_boxes']
                # filtered_scores = pred_dicts[0]['pred_scores']
                # filtered_labels = pred_dicts[0]['pred_labels']
    
                boxes_label = torch.cat((filtered_boxes, filtered_labels.unsqueeze(1)), dim=1)
                boxes_label_score= torch.cat((boxes_label, filtered_scores.unsqueeze(1)), dim=1)

                
                if args.save_to_file:
                    timestamp= batch_dict['timestamp'][0]
                    dataset_name=cfg.EXP_GROUP_PATH
                    model_name=cfg.TAG
                    folder=f"../result/{dataset_name}/{model_name}/{batch_dict['folder'][0]}"
                    pred_result_name=folder+"/"+timestamp+".txt"
                    SaveBoxPred(boxes_label_score,pred_result_name)
                    output_image=folder+"/"+timestamp+".png"
                    gt_boxes=batch_dict['gt_boxes'][0]

                    # offscreen_visualization_array(
                    #     batch_dict['points'][:, 1:],
                    #     ref_boxes=boxes_label,
                    #     gt_boxes=gt_boxes,
                    #     box_colormap=box_colormap,
                    #     output_image=output_image
                    # )
                    visualization_array_pyvista(
                        batch_dict['points'][:, 1:],
                        ref_boxes=boxes_label_score,
                        gt_boxes=gt_boxes,
                        output_image=output_image,
                        class_names=class_names,
                        box_colormap=box_colormap,
                        off_screen=True
                        
                    )
                else:
                    V.draw_scenes(
                        points=batch_dict['points'][:, 1:], gt_boxes=batch_dict['gt_boxes'][0],ref_boxes=filtered_boxes,
                        ref_scores=filtered_scores, ref_labels=filtered_labels
                    )
                if not OPEN3D_FLAG:
                    mlab.show(stop=True)

    logger.info('Demo done.')


if __name__ == '__main__':
    main()
