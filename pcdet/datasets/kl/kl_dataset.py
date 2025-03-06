from ..dataset import DatasetTemplate
# print(__name__)
# print(__package__)
class KLDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        super().__init__(dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger)



def create_kl_infos(dataset_cfg, class_names, data_path, save_path, workers=4):
    pass

# if __name__ == '__main__':
#     import yaml
#     import argparse
#     from pathlib import Path
#     from easydict import EasyDict

#     parser = argparse.ArgumentParser(description='arg parser')
#     parser.add_argument('--cfg_file', type=str, default=None, help='specify the config of dataset')
#     parser.add_argument('--func', type=str, default='create_kl_infos', help='')
#     parser.add_argument('--version', type=str, default='v1.0-trainval', help='')
#     parser.add_argument('--with_cam', action='store_true', default=False, help='use camera or not')
#     args = parser.parse_args()
    
#     if args.func == 'create_kl_infos':
#         dataset_cfg = EasyDict(yaml.safe_load(open(args.cfg_file)))
#         ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
#         dataset_cfg.VERSION = args.version
#         create_kl_infos(
#             version=dataset_cfg.VERSION,
#             data_path=ROOT_DIR / 'data' / 'nuscenes',
#             save_path=ROOT_DIR / 'data' / 'nuscenes',
#             max_sweeps=dataset_cfg.MAX_SWEEPS,
#             with_cam=args.with_cam
#         )
#         kl_dataset=KLDataset()
#         kl_dataset = KLDataset(
#             dataset_cfg=dataset_cfg, class_names=None,
#             root_path=ROOT_DIR / 'data' / 'nuscenes',
#             logger=common_utils.create_logger(), training=True
#         )
#         # nuscenes_dataset.create_groundtruth_database(max_sweeps=dataset_cfg.MAX_SWEEPS)