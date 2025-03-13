import os
import time
from .kl_utils import precompute_timestamps,find_multi_sensor_data,generate_token
class KL():
    #nusc = NuScenes(version=self.dataset_cfg.VERSION, dataroot=str(self.root_path), verbose=True)
    def __init__(self,
                 version: str = 'v1.0-mini',
                 dataroot: str = '/data/sets/nuscenes',
                 verbose: bool = True,
                 map_resolution: float = 0.1):
        self.version = version
        self.dataroot = dataroot
        self.verbose = verbose
        self.label_dir=self.dataroot/version/'label'
        self.sample_dir=self.dataroot/version/'sample'
        self.sensor_names={
            'helios_front_left':'helios_front_left',
            'helios_rear_right':'helios_rear_right',
            'bp_front_left':'bp_front_left',
            'bp_front_right':'bp_front_right',
            'bp_rear_left':'bp_rear_left',
            'bp_rear_right':'bp_rear_right'}
        self.samples=[]
        start_time = time.time()
        if verbose:
            print("======\nLoading NuScenes tables for version {}...".format(self.version))
        self.generate_sample()


        if verbose:
            # for table in self.table_names:
            #     print("{} {},".format(len(getattr(self, table)), table))
            print("Done loading in {:.3f} seconds.\n======".format(time.time() - start_time))
    
    def generate_sample(self):
        for s_dir in os.listdir(self.label_dir):
            label_s_path = self.label_dir/s_dir  # label/s7
            sample_s_path = self.sample_dir/s_dir  # sample/s7

            # 检查 label 下的子目录是否存在，并且是一个目录
            if os.path.isdir(label_s_path):
                # 遍历 label/s7 下的场景目录（如 igv_1114_rain_01-century02）
                for scenario_dir in os.listdir(label_s_path):
                    label_scenario_path = label_s_path/scenario_dir  # label/s7/igv_1114_rain_01-century02
                    sample_lidar_path = sample_s_path/scenario_dir/'lidar'  # sample/s7/igv_1114_rain_01-century02/lidar
                    extrinsics_path = sample_s_path/scenario_dir/'extrinsics.json' if (sample_s_path/scenario_dir/'extrinsics.json').exists() else None
                    intrinsics_path = sample_s_path/scenario_dir/'intrinsics.json' if (sample_s_path/scenario_dir/'intrinsics.json').exists() else None
                    # 检查 label 下的场景目录label/s7/igv_1114_rain_01-century02是否存在，并且是一个目录
                    if os.path.isdir(label_scenario_path):
                        # 对于每个lidar目录，预先计算好时间，便于二分查找加速
                        sensor_files={}
                        sensor_timestamps={}
                        for sensor_name in self.sensor_names:
                            sensor_file=list((sample_lidar_path/sensor_name).glob('*.bin'))
                            sensor_timestamp=precompute_timestamps(sensor_file)
                            sensor_files[sensor_name]=sensor_file
                            sensor_timestamps[sensor_name]=sensor_timestamp
                        # 遍历 label 场景目录下的 JSON 文件
                        for json_file in os.listdir(label_scenario_path):
                            if json_file.endswith('.json'):
                                sample={}
                                # 构建 JSON 文件的完整路径
                                json_path = label_scenario_path/json_file
                                timestamp=os.path.splitext(json_file)[0]
                                matched_lidars=find_multi_sensor_data(timestamp, sensor_files,sensor_timestamps)
                                sample['token']=generate_token()
                                sample['label']=json_path
                                sample['lidars']=matched_lidars
                                sample['timestamp']=timestamp
                                sample['extrinsics_path']=extrinsics_path
                                sample['intrinsics_path']=intrinsics_path
                                self.samples.append(sample)
                                # 构建对应的 bin 文件名
    
    def get_all_sample(self):
        return self.samples

    # def compute(self):
    #     # compute KL divergence between x and y
    #     pass

if __name__ == '__main__':
    kl = KL(version='v1.0-trainval', dataroot='../data/kl', verbose=True)