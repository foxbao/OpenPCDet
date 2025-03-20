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
from pcdet.datasets.kl.kl_dataset import KLDataset
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


def parse_config():
    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument(
        "--cfg_file",
        type=str,
        default="cfgs/kitti_models/second.yaml",
        help="specify the config for demo",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="pv_rcnn_8369.pth",
        help="specify the pretrained model",
    )

    parser.add_argument(
        "--version",
        type=str,
        default="v1.0-mini",
        help="specify the pretrained model",
    )

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info("-----------------Quick Demo of OpenPCDet-------------------------")
    dataset_cfg=cfg.DATA_CONFIG
    ROOT_DIR = (Path(__file__).resolve().parent / '../').resolve()
    
    kl_dataset = KLDataset(
        dataset_cfg=dataset_cfg, class_names=cfg.CLASS_NAMES,
        root_path=ROOT_DIR / 'data' / 'kl',
        logger=common_utils.create_logger(), training=True
    )

    logger.info(f"Total number of samples: \t{len(kl_dataset)}")

    model = build_network(
        model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=kl_dataset
    )
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    with torch.no_grad():
        for idx, data_dict in enumerate(kl_dataset):
            logger.info(f"Visualized sample index: \t{idx + 1}")
            data_dict = kl_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)

            V.draw_scenes(
                points=data_dict["points"][:, 1:],
                ref_boxes=pred_dicts[0]["pred_boxes"],
                ref_scores=pred_dicts[0]["pred_scores"],
                ref_labels=pred_dicts[0]["pred_labels"],
            )

            if not OPEN3D_FLAG:
                mlab.show(stop=True)

    logger.info("Demo done.")


if __name__ == "__main__":
    main()
