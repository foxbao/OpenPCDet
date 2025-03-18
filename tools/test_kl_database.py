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
        dataset_cfg = EasyDict(yaml.safe_load(open(args.cfg_file)))
        dataset_cfg.VERSION=args.version
        ROOT_DIR = (Path(__file__).resolve().parent / '../').resolve()
        data_path=ROOT_DIR / 'data' / 'kl'
        create_kl_infos(dataset_cfg.VERSION, data_path, data_path,with_cam=args.with_cam)

        kl_dataset = KLDataset(
            dataset_cfg=dataset_cfg, class_names=None,
            root_path=ROOT_DIR / 'data' / 'kl',
            logger=common_utils.create_logger(), training=True
        )

        kl_dataset.create_groundtruth_database()
    elif args.func == 'visualize_infos':
        dataset_cfg = EasyDict(yaml.safe_load(open(args.cfg_file)))
        dataset_cfg.VERSION=args.version
        ROOT_DIR = (Path(__file__).resolve().parent / '../').resolve()
        kl_dataset = KLDataset(
            dataset_cfg=dataset_cfg, class_names=None,
            root_path=ROOT_DIR / 'data' / 'kl',
            logger=common_utils.create_logger(), training=True
        )

        from torch.utils.data import DataLoader
        # 创建 DataLoader
        kl_dataloader = DataLoader(
            dataset=kl_dataset,  # 你的数据集
            batch_size=1,        # 每次取一个样本
            shuffle=False,       # 是否打乱数据
            num_workers=0        # 数据加载的线程数（0 表示在主进程中加载）
        )

        # 获取第一笔数据
        first_batch = next(iter(kl_dataloader))

        # 打印数据
        print(first_batch)