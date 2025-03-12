from pcdet.datasets.kl.kl_dataset import KLDataset
from pcdet.utils import common_utils
from pcdet.datasets.kl.kl_dataset import create_kl_infos

if __name__ == '__main__':
    import yaml
    import argparse
    from pathlib import Path
    from easydict import EasyDict

    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/dataset_configs/kl_dataset.yaml', help='specify the config of dataset')
    parser.add_argument('--func', type=str, default='create_kl_infos', help='')
    parser.add_argument('--version', type=str, default='v1.0-trainval', help='')
    parser.add_argument('--with_cam', action='store_true', default=False, help='use camera or not')
    args = parser.parse_args()


    if args.func == 'create_kl_infos':
    #     # dataset_cfg = EasyDict(yaml.safe_load(open(args.cfg_file)))
        ROOT_DIR = (Path(__file__).resolve().parent / '../').resolve()
        data_path=ROOT_DIR / 'data' / 'kl'
        create_kl_infos(args.version, data_path, data_path,with_cam=args.with_cam)

    dataset_cfg = EasyDict(yaml.safe_load(open(args.cfg_file)))
    ROOT_DIR = (Path(__file__).resolve().parent / '../').resolve()
    dataset_cfg.VERSION = args.version
    data_path=ROOT_DIR / 'data' / 'kl'
    kl_dataset = KLDataset(
        dataset_cfg=dataset_cfg, class_names=None,
        root_path=ROOT_DIR / 'data' / 'kl',
        logger=common_utils.create_logger(), training=True
    )

    kl_dataset.create_groundtruth_database()