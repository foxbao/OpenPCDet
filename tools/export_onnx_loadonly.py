# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import glob
import onnx
import torch
import argparse
import numpy as np

from pathlib import Path
from onnxsim import simplify
from pcdet.utils import common_utils
from pcdet.models import build_network
from pcdet.datasets import DatasetTemplate
from pcdet.config import cfg, cfg_from_yaml_file

from modify_onnx import simplify_preprocess, simplify_postprocess


class DemoDataset(DatasetTemplate):
    def __init__(
        self,
        dataset_cfg,
        class_names,
        training=True,
        root_path=None,
        logger=None,
        ext=".bin",
    ):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg,
            class_names=class_names,
            training=training,
            root_path=root_path,
            logger=logger,
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = (
            glob.glob(str(root_path / f"*{self.ext}"))
            if self.root_path.is_dir()
            else [self.root_path]
        )

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == ".bin":
            points = np.fromfile(
                self.sample_file_list[index], dtype=np.float32
            ).reshape(-1, 4)
        elif self.ext == ".npy":
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            "points": points,
            "frame_id": index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument(
        "--cfg_file",
        type=str,
        default="cfgs/kitti_models/pointpillar.yaml",
        help="specify the config for demo",
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
        "--data_path",
        type=str,
        default="data",
        help="specify the point cloud data file or directory",
    )
    parser.add_argument(
        "--ckpt", type=str, default=None, help="specify the pretrained model"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="model",
        help="specify the extension of your point cloud data file",
    )
    parser.add_argument(
        "--launcher", choices=["none", "pytorch", "slurm"], default="none"
    )

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()

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
    logger = common_utils.create_logger()
    logger.info("------ Convert OpenPCDet model for TensorRT ------")
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        training=False,
        root_path=Path(args.data_path),
        ext=".bin",
        logger=logger,
    )
    from pcdet.datasets import build_dataloader

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
        model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset
    )
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)

    model.cuda()
    model.eval()

    np.set_printoptions(threshold=np.inf)

    with torch.no_grad():

        dummy_input = dict()

        from pcdet.models import load_data_to_gpu
        for i, batch_dict in enumerate(test_loader):
            load_data_to_gpu(batch_dict)
            dummy_input["voxels"]=batch_dict["voxels"]
            dummy_input["voxel_num_points"]=batch_dict["voxel_num_points"]
            dummy_input["voxel_coords"] = batch_dict["voxel_coords"]
            dummy_input["batch_size"] = batch_dict["batch_size"]
            result=model(dummy_input)
            break

    # test onnx_raw with onnxrunner
    import onnxruntime as ort

    ort_session = ort.InferenceSession(
        os.path.join(args.out_dir, "pointpillar_raw.onnx")
    )

    # 检查输入输出是否确实支持动态维度
    for input_tensor in ort_session.get_inputs():
        print(f"Input: {input_tensor.name}, Shape: {input_tensor.shape},Type: {input_tensor.type}")
        # 应该看到类似 image_input: ['batch_size', 3, 'height', 'width'] 的输出

    for output_tensor in ort_session.get_outputs():
        print(f"Output: {output_tensor.name}, Shape: {output_tensor.shape},Type: {output_tensor.type}")
        # 应该看到类似 output: ['batch_size', 3] 的输出

    # # 8. **转换 NumPy 格式输入**
    onnx_input = dict()
    onnx_input['voxels'] = dummy_input["voxels"].cpu().numpy().astype(np.float32)
    onnx_input['voxel_num'] = dummy_input["voxel_num_points"].cpu().numpy().astype(np.float32) 
    onnx_input['voxel_idxs'] = dummy_input["voxel_coords"] .cpu().numpy().astype(np.float32)   
    onnx_input['batch_size'] = np.array(1,dtype=np.int64)          

    # onnx_input['batch_size']=np.array([1])

    # 9. **运行 ONNX 推理**
    onnx_output = ort_session.run(None, onnx_input)

    torch_output=model(dummy_input)

    torch_numpy=[]
    for tensor in torch_output.values():
        if torch.is_tensor(tensor):
            torch_numpy.append(tensor.detach().cpu().numpy())
        else:
            torch_numpy.append(tensor)

    np.save('torch_numpy.npy', torch_numpy)  # 默认扩展名是 .npy
    np.save('onnx_output.npy', onnx_output)  # 默认扩展名是 .npy
    max_error=float('-inf')
    for torch_numpy, onnx_output in zip(torch_numpy, onnx_output):
        print(np.abs(torch_numpy - onnx_output).max())
        max_error = max(np.abs(torch_numpy - onnx_output).max(),max_error)

# error = np.abs(torch_output - onnx_output[0]).max()
    print(f"最大绝对误差: {max_error:.5e}")  # 科学计数法显示

if __name__ == "__main__":
    main()
