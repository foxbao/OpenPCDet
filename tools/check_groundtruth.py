
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
import open3d as o3d
from tqdm import tqdm
box_colormap = [
    (1, 1, 1),
    (1, 0, 0),        # 红色, Pedestrian，因为数据集里面会对每个label+1
    (0, 1, 0),        # 绿色，Car 
    (0, 0, 1),        # 蓝色，IGV-Full 
    (1, 1, 0),        # 黄色，Truck
    (0, 1, 1),        # 青色，Trailer-Empty
    (1, 0, 1),        # 紫色，Trailer-Full
    (0.5, 0.5, 0.5),  # 灰色，IGV-Empty
    (1, 0.5, 0),      # 橙色，Crane
    (0.5, 0, 0.5),    # 深紫色，OtherVehicle
    (0, 0.5, 0.5),    # 深青色，Cone
    (0.2, 0.8, 0.2),  # 浅绿，ContainerForklift
    (0.8, 0.2, 0.2),  # 浅红，Forklift
    (0.2, 0.2, 0.8),  # 浅蓝，Lorry
    (0.7, 0.7, 0.2),  # 橄榄绿，ConstructionVehicle
    (0.6, 0.3, 0.7),  # 淡紫色
    (0.9, 0.6, 0.1),  # 金色
    (0.4, 0.7, 0.4),  # 薄荷绿
    (0.3, 0.5, 0.8),  # 天蓝
    (0.8, 0.4, 0.6),  # 粉红
    (0.1, 0.9, 0.5)   # 荧光绿
]

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




def create_3d_box(center, size, rotation_matrix=np.eye(3), color=[1, 0, 0]):
    """
    创建3D边界框（8个顶点+12条边）
    参数:
        center: [x,y,z] 框中心坐标
        size: [长,宽,高]
        rotation_matrix: 3x3旋转矩阵
        color: RGB颜色值
    """
    # 计算半长宽高
    l, w, h = size[0]/2, size[1]/2, size[2]/2
    
    # 定义8个顶点的相对坐标
    vertices = np.array([
        [-l, -w, -h], [l, -w, -h], [l, w, -h], [-l, w, -h],
        [-l, -w, h], [l, -w, h], [l, w, h], [-l, w, h]
    ])
    
    # 应用旋转和平移
    vertices = vertices @ rotation_matrix.T + center
    
    # 定义12条边的连接关系
    lines = [
        [0,1], [1,2], [2,3], [3,0],  # 底面
        [4,5], [5,6], [6,7], [7,4],  # 顶面
        [0,4], [1,5], [2,6], [3,7]   # 侧面
    ]
    
    # 创建线框
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(vertices)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color for _ in range(len(lines))])
    
    return line_set

def create_3d_bbox(bbox_3d, color=(1, 0, 0)):
    """创建3D检测框线框"""
    center = bbox_3d[:3]
    length, width, height = bbox_3d[3:6]
    rotation = bbox_3d[6]
    
    # 计算8个角点
    corners = np.array([
        [length/2, width/2, height/2],
        [length/2, width/2, -height/2],
        [length/2, -width/2, height/2],
        [length/2, -width/2, -height/2],
        [-length/2, width/2, height/2],
        [-length/2, width/2, -height/2],
        [-length/2, -width/2, height/2],
        [-length/2, -width/2, -height/2]
    ])
    
    # 应用旋转和平移
    # rotation=0
    rot_mat = np.array([
        [np.cos(rotation), -np.sin(rotation), 0],
        [np.sin(rotation), np.cos(rotation), 0],
        [0, 0, np.cos(rotation)]
    ])

    # rot_mat=np.ones(3,3)
    corners = np.dot(corners, rot_mat.T) + center
    
    # 定义12条边
    lines = [
        [0, 1], [0, 2], [1, 3], [2, 3],
        [4, 5], [4, 6], [5, 7], [6, 7],
        [0, 4], [1, 5], [2, 6], [3, 7]]
    
    # 创建线框
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color for _ in range(len(lines))])
    return line_set

def offscreen_visualization_array(points_array, bboxes=None,folder="",output_image="point_cloud.png",zoom=0.25):

    assert points_array.shape[1] >= 4, "输入数组需要至少4列（XYZ+强度）"
    xyz = points_array[:, :3]
    # 读取点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(np.ones((points_array.shape[0], 3)))  # 所有点设为白色

    bbox_geometries = []
    for i, bbox in enumerate(bboxes):
        # color = cm.plasma(i / len(bboxes))[:3]  # 框的颜色渐变
        # color = (0, 1, 0)  # 固定为绿色 RGB
        label=int(bbox[7])
        color = box_colormap[label]  # 固定为绿色 RGB

        bbox_geometries.append(create_3d_bbox(bbox, color=color))
    
    # 添加坐标系
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2)

    # 创建离屏可视化器
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)  # 不可见窗口
    
    # 添加几何体
    vis.add_geometry(pcd)
    vis.add_geometry(coordinate_frame)
    for bbox in bbox_geometries:
        vis.add_geometry(bbox)

    
    # 渲染设置
    opt = vis.get_render_option()
    opt.background_color = np.array([0, 0, 0])  # 黑色背景
    opt.point_size = 2.0
    opt.light_on = True   # 启用光照（增强白色点可见性）
    
    # 调整视图
    ctr = vis.get_view_control()
    ctr.set_front([0, 0, -1])
    ctr.set_up([0, -1, 0])
    ctr.set_lookat(pcd.get_center())  # ⭐️ 核心：焦点对准点云中心
    ctr.set_zoom(zoom)
    
    # 强制渲染
    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    
    # 保存图像
    # print(output_image)
    output_path=folder+"/"+output_image+".png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)  # 自动创建文件夹
    vis.capture_screen_image(output_path, do_render=True)
    vis.destroy_window()


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

    for idx, batch_dict in enumerate(tqdm(train_loader)):
        points = batch_dict['points']
        points = points[:, 1:]
        gt_boxes=batch_dict['gt_boxes'][0]
        folder="groundtruth_train/"+batch_dict['folder'][0]
        timestamp=batch_dict['timestamp'][0]
        offscreen_visualization_array(
            points,
            bboxes=gt_boxes,
            folder=folder,
            output_image=timestamp
        )

    val_set, val_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=dist_test, workers=args.workers, logger=logger, training=False
    )

    
    # 注意，要运行这个代码，需要修改一下kl_dataset里面的返回值，在__getitem__函数中，将timestamp和folder都返回
    for idx, batch_dict in enumerate(tqdm(val_loader)):
        points = batch_dict['points']
        points = points[:, 1:]
        gt_boxes=batch_dict['gt_boxes'][0]
        folder="groundtruth_val/"+batch_dict['folder'][0]
        timestamp=batch_dict['timestamp'][0]
        offscreen_visualization_array(
            points,
            bboxes=gt_boxes,
            folder=folder,
            output_image=timestamp
        )

    # for idx, batch_dict in enumerate(test_loader):
    #     points=batch_dict['points']
    #     points=points[:,1:]
    #     # timestamp=batch_dict['timestamp'][0]
    #     offscreen_visualization_array(points,folder=batch_dict['folder'][0],output_image=batch_dict['timestamp'][0])


if __name__ == '__main__':
    main()
