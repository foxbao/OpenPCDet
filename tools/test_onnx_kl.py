import _init_path
import argparse
import datetime
import glob
import os
import re
import time
from pathlib import Path
from tensorboardX import SummaryWriter
from eval_utils import eval_utils

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
from pcdet.datasets import build_dataloader
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file


def parse_config():

    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument(
        "--cfg_file", type=str, default=None, help="specify the config for training"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        required=False,
        help="batch size for training",
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="number of workers for dataloader"
    )
    parser.add_argument(
        "--extra_tag", type=str, default="default", help="extra tag for this experiment"
    )
    parser.add_argument(
        "--ckpt", type=str, default=None, help="checkpoint to start from"
    )
    parser.add_argument(
        "--pretrained_model", type=str, default=None, help="pretrained_model"
    )
    parser.add_argument(
        "--launcher", choices=["none", "pytorch", "slurm"], default="none"
    )
    parser.add_argument(
        "--tcp_port", type=int, default=18888, help="tcp port for distrbuted training"
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=None,
        help="local rank for distributed training",
    )
    parser.add_argument(
        "--set",
        dest="set_cfgs",
        default=None,
        nargs=argparse.REMAINDER,
        help="set extra config keys if needed",
    )

    parser.add_argument(
        "--max_waiting_mins", type=int, default=30, help="max waiting minutes"
    )
    parser.add_argument("--start_epoch", type=int, default=0, help="")
    parser.add_argument(
        "--eval_tag", type=str, default="default", help="eval tag for this experiment"
    )
    parser.add_argument(
        "--eval_all",
        action="store_true",
        default=False,
        help="whether to evaluate all checkpoints",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="specify a ckpt directory to be evaluated if needed",
    )
    parser.add_argument("--save_to_file", action="store_true", default=False, help="")
    parser.add_argument(
        "--infer_time",
        action="store_true",
        default=False,
        help="calculate inference latency",
    )

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = "/".join(
        args.cfg_file.split("/")[1:-1]
    )  # remove 'cfgs' and 'xxxx.yaml'

    np.random.seed(1024)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


# 替换前向传播，仅保留网络主体部分
class ExportModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, voxel_features, voxel_coords, voxel_num_points):
        # 仅保留网络主体部分（参考 OpenPCDet 的 forward 函数）
        batch_dict = {
            "voxels": voxel_features,
            "voxel_coords": voxel_coords,
            "voxel_num_points": voxel_num_points,
            "batch_size": 4,
        }

        pred_dicts, ret_dict = self.model(batch_dict)
        return pred_dicts["batch_box_preds"], pred_dicts["batch_cls_preds"]


def main():
    args, cfg = parse_config()

    if args.infer_time:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    if args.launcher == "none":
        dist_test = False
        total_gpus = 1
    else:
        if args.local_rank is None:
            args.local_rank = int(os.environ.get("LOCAL_RANK", "0"))

        total_gpus, cfg.LOCAL_RANK = getattr(
            common_utils, "init_dist_%s" % args.launcher
        )(args.tcp_port, args.local_rank, backend="nccl")
        dist_test = True

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert (
            args.batch_size % total_gpus == 0
        ), "Batch size should match the number of gpus"
        args.batch_size = args.batch_size // total_gpus

    output_dir = cfg.ROOT_DIR / "output" / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_output_dir = output_dir / "eval"

    if not args.eval_all:
        num_list = re.findall(r"\d+", args.ckpt) if args.ckpt is not None else []
        epoch_id = num_list[-1] if num_list.__len__() > 0 else "no_number"
        eval_output_dir = (
            eval_output_dir
            / ("epoch_%s" % epoch_id)
            / cfg.DATA_CONFIG.DATA_SPLIT["test"]
        )
    else:
        eval_output_dir = eval_output_dir / "eval_all_default"

    if args.eval_tag is not None:
        eval_output_dir = eval_output_dir / args.eval_tag

    eval_output_dir.mkdir(parents=True, exist_ok=True)
    log_file = eval_output_dir / (
        "log_eval_%s.txt" % datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file
    logger.info("**********************Start logging**********************")
    gpu_list = (
        os.environ["CUDA_VISIBLE_DEVICES"]
        if "CUDA_VISIBLE_DEVICES" in os.environ.keys()
        else "ALL"
    )
    logger.info("CUDA_VISIBLE_DEVICES=%s" % gpu_list)

    if dist_test:
        logger.info("total_batch_size: %d" % (total_gpus * args.batch_size))
    for key, val in vars(args).items():
        logger.info("{:16} {}".format(key, val))
    log_config_to_file(cfg, logger=logger)

    ckpt_dir = args.ckpt_dir if args.ckpt_dir is not None else output_dir / "ckpt"

    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_test,
        workers=args.workers,
        logger=logger,
        training=False,
    )

    model = build_network(
        model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set
    )
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=dist_test)
    model.cuda()
    model.eval()
    export_model = ExportModel(model)

    num_voxels = 55702
    max_points_per_voxel = 32
    num_point_features = 4
    batch_size=1

    dummy_voxel_features = torch.randn(
        num_voxels,
        max_points_per_voxel,
        num_point_features,
        dtype=torch.float32,
        device="cuda",
    )
    dummy_voxel_coords = torch.randint(
        0, 4, (num_voxels, 4), dtype=torch.int32, device="cuda"
    )  # 包含 batch_idx
    dummy_voxel_num_points = torch.randint(
        1, max_points_per_voxel, (num_voxels,), dtype=torch.int32, device="cuda"
    )

    export_model(dummy_voxel_features, dummy_voxel_coords, dummy_voxel_num_points)


if __name__ == "__main__":
    # export_model = ExportModel(model)
    main()
