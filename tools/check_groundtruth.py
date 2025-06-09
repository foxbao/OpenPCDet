import argparse
import datetime
import os
import re
from pathlib import Path
import numpy as np

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import common_utils
from pcdet.datasets import build_dataloader
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file

from tqdm import tqdm
from visual_utils.visualize_tools import offscreen_visualization_array,visualization_array_pyvista
from visual_utils.open3d_vis_utils import box_colormap


from collections import defaultdict

def update_class_stats(stats_dict, boxes):
    for box in boxes:
        l, w, h = box[3:6]
        z = box[2]
        bottom = z - h / 2
        label = int(box[-1])
        
        stats_dict[label]['l_sum'] += l
        stats_dict[label]['w_sum'] += w
        stats_dict[label]['h_sum'] += h
        stats_dict[label]['bottom_sum'] += bottom
        stats_dict[label]['count'] += 1
        


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
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')
    parser.add_argument('--infer_time', action='store_true', default=False, help='calculate inference latency')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    np.random.seed(1024)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg

def analyze_gt(train_loader):
    # 初始化一个 defaultdict 保存每类的统计信息
    class_stats = defaultdict(lambda: {'l_sum': 0, 'w_sum': 0, 'h_sum': 0, 'bottom_sum': 0, 'count': 0})
    print("\nAverage LWH and anchor bottom height per class:")
    for idx, batch_dict in enumerate(tqdm(train_loader)):
        gt_boxes = batch_dict['gt_boxes'][0]
        update_class_stats(class_stats, gt_boxes)
        
    print("\nAverage LWH and anchor bottom height per class:")
    for cls_id in sorted(class_stats.keys()):
        stats = class_stats[cls_id]
        if stats['count'] == 0:
            continue
        avg_l = stats['l_sum'] / stats['count']
        avg_w = stats['w_sum'] / stats['count']
        avg_h = stats['h_sum'] / stats['count']
        avg_bottom = stats['bottom_sum'] / stats['count']
        class_name = cfg.CLASS_NAMES[cls_id-1] if cls_id < len(cfg.CLASS_NAMES) else str(cls_id)
        print(f"Class {class_name} (ID: {cls_id-1}):")
        print(f"  Avg Length: {avg_l:.2f}, Width: {avg_w:.2f}, Height: {avg_h:.2f}")
        print(f"  Avg Anchor Bottom Height: {avg_bottom:.2f}")

def main():
    # os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    args, cfg = parse_config()

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

    if not args.eval_all:
        num_list = re.findall(r'\d+', args.ckpt) if args.ckpt is not None else []
        epoch_id = num_list[-1] if num_list.__len__() > 0 else 'no_number'
        eval_output_dir = eval_output_dir / ('epoch_%s' % epoch_id) / cfg.DATA_CONFIG.DATA_SPLIT['test']
    else:
        eval_output_dir = eval_output_dir / 'eval_all_default'

    if args.eval_tag is not None:
        eval_output_dir = eval_output_dir / args.eval_tag

    eval_output_dir.mkdir(parents=True, exist_ok=True)
    log_file = eval_output_dir / ('log_eval_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    if dist_test:
        logger.info('total_batch_size: %d' % (total_gpus * args.batch_size))
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)

    train_set, train_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=dist_test, workers=args.workers, logger=logger, training=True
    )
    
    analyze_gt(train_loader=train_loader)
    
    print("saving to ","../result/groundtruth_train")
    
    
    for idx, batch_dict in enumerate(tqdm(train_loader)):
        points = batch_dict['points']
        points = points[:, 1:]
        gt_boxes=batch_dict['gt_boxes'][0]
        folder="../result/groundtruth_train/"+batch_dict['folder'][0]
        timestamp=batch_dict['timestamp'][0]
        output_image=folder+"/"+timestamp+".png"
        
        if args.save_to_file:
            offscreen_visualization_array(
                points,
                gt_boxes=gt_boxes,
                output_image=output_image,
                box_colormap=box_colormap
            )
        else:
            visualization_array_pyvista(
                points,
                gt_boxes=gt_boxes,
                output_image=output_image,
                box_colormap=box_colormap
            )

    val_set, val_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=dist_test, workers=args.workers, logger=logger, training=False
    )

    print("saving to ","../result/groundtruth_val")
    # 注意，要运行这个代码，需要修改一下kl_dataset里面的返回值，在__getitem__函数中，将timestamp和folder都返回
    for idx, batch_dict in enumerate(tqdm(val_loader)):
        points = batch_dict['points']
        points = points[:, 1:]
        gt_boxes=batch_dict['gt_boxes'][0]
        folder="../result/groundtruth_val/"+batch_dict['folder'][0]
        timestamp=batch_dict['timestamp'][0]
        output_image=folder+"/"+timestamp+".png"
        if args.save_to_file:            
            offscreen_visualization_array(
                points,
                gt_boxes=gt_boxes,
                output_image=output_image,
                box_colormap=box_colormap
            )
        else:
            visualization_array_pyvista(
                points,
                gt_boxes=gt_boxes,
                output_image=output_image,
                box_colormap=box_colormap
            )



if __name__ == '__main__':
    main()
