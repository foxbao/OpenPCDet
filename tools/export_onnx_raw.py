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
        if self.dataset_mode=='kl':
            points = points[:, :self.feature_num]

        input_dict = {
            'points': points,
            'frame_id': index,
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


def prepare_batch_dict_from_ort_outputs(ort_outputs, output_names, batch_size=1, cls_preds_normalized=False,device='cuda:0'):
    """
    将ONNX Runtime的输出转换为可被post_processing处理的batch_dict
    
    Args:
        ort_outputs: ort_session.run的输出结果列表
        output_names: ONNX模型输出名称列表，顺序与ort_outputs对应
        batch_size: 批处理大小，默认为1
        cls_preds_normalized: 分类预测是否已经归一化，默认为False
    
    Returns:
        batch_dict: 符合post_processing输入要求的字典
    """
    batch_dict = {}
    
    # 将ORT输出映射到batch_dict
    for i, name in enumerate(output_names):
        if i < len(ort_outputs):
            # 将NumPy数组转换为PyTorch张量
            batch_dict[name] = torch.from_numpy(ort_outputs[i]).to(device)
    
    # 添加post_processing所需的必要字段
    batch_dict['batch_size'] = batch_size
    batch_dict['cls_preds_normalized'] = cls_preds_normalized
    return batch_dict


def main():
    # 这个代码要运行，需要把pointpillar代码改一下
        # else:
        #     # batch_dict.pop('cls_preds_normalized', None)
        #     # return batch_dict
        #     pred_dicts, recall_dicts = self.post_processing(batch_dict)
        #     return pred_dicts, recall_dicts
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
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path),  logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    from pcdet.datasets import build_dataloader
    test_dataset, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_test,
        workers=args.workers,
        logger=logger,
        training=False,
    )

    model = build_network(
        model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_dataset
    )
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)

    model.cuda()
    model.eval()

    np.set_printoptions(threshold=np.inf)

    with torch.no_grad():
        MAX_VOXELS = 12537
        dummy_voxels = torch.zeros((MAX_VOXELS, 32, 4), dtype=torch.float32, device="cuda:0")
        dummy_voxel_idxs = torch.zeros((MAX_VOXELS, 4), dtype=torch.int32, device="cuda:0")
        dummy_voxel_num = torch.zeros((MAX_VOXELS,), dtype=torch.int32, device="cuda:0")

        dummy_input = dict()
        dummy_input["voxels"] = dummy_voxels
        dummy_input["voxel_num_points"] = dummy_voxel_num
        dummy_input["voxel_coords"] = dummy_voxel_idxs
        dummy_input["batch_size"] = 1

        # result=model(dummy_input)
        # aaaa=1

        real_input=dict()
        from pcdet.models import load_data_to_gpu
        for i, batch_dict in enumerate(demo_dataset):
            data_dict = demo_dataset.collate_batch([batch_dict])
            load_data_to_gpu(data_dict)
            real_input["voxels"]=data_dict["voxels"]
            real_input["voxel_num_points"]=data_dict["voxel_num_points"].to(torch.int32)
            real_input["voxel_coords"] = data_dict["voxel_coords"].to(torch.int32)
            real_input["batch_size"] = data_dict["batch_size"]
            break
            
        # use real input to export onnx
        # dummy_input=real_input
        torch.onnx.export(
            model,  # model being run
            dummy_input,  # model input (or a tuple for multiple inputs)
            os.path.join(
                args.out_dir, "pointpillar_raw.onnx"
            ),  # where to save the model (can be a file or file-like object)
            export_params=True,  # store the trained parameter weights inside the model file
            opset_version=11,  # the ONNX version to export the model to
            do_constant_folding=True,  # whether to execute constant folding for optimization
            keep_initializers_as_inputs=True,
            input_names=[
                "voxels",
                "voxel_num",
                "voxel_idxs",
                "batch_size",
            ],  # the model's input names
            output_names=[
                "voxels2",
                "voxel_num_points2",
                "voxel_coords2",
                "batch_size2",
                "pillar_features",
                "spatial_features",
                "spatial_features_2d",
                "batch_cls_preds",
                "batch_box_preds",
                # "cls_preds_normalized"
            ],  # the model's output names
        )
        print("ONNX 模型导出成功！")

    onnx_raw = onnx.load(
        os.path.join(args.out_dir, "pointpillar_raw.onnx")
    )  # load onnx model

    # onnx_trim_post = simplify_postprocess(onnx_raw)
    # onnx.save(onnx_trim_post, os.path.join(args.out_dir, "onnx_trim_post.onnx"))

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

    # onnx_trim_post = simplify_postprocess(onnx_raw)
    # onnx.save(onnx_trim_post, os.path.join(args.out_dir, "onnx_trim_post.onnx"))
    # onnx_simp, check = simplify(onnx_trim_post)
    # assert check, "Simplified ONNX model could not be validated"

    # onnx_final = simplify_preprocess(onnx_simp)
    # onnx.save(onnx_final, os.path.join(args.out_dir, "pointpillar.onnx"))



    # # 8. **转换 NumPy 格式输入**
    onnx_input = dict()
    onnx_input['voxels'] = real_input["voxels"].cpu().numpy().astype(np.float32)
    onnx_input['voxel_num'] = real_input["voxel_num_points"].cpu().numpy().astype(np.int32) 
    onnx_input['voxel_idxs'] = real_input["voxel_coords"] .cpu().numpy().astype(np.int32)     
    onnx_input['batch_size'] = np.array(1,dtype=np.int64)          

    # onnx_input['batch_size']=np.array([1])

    # 9. **运行 ONNX 推理**
    onnx_output = ort_session.run(None, onnx_input)

    output_names = [
        "voxels",
        "voxel_num_points",
        "voxel_coords",
        "batch_size",
        "pillar_features",
        "spatial_features",
        "spatial_features_2d",
        "batch_cls_preds",
        "batch_box_preds",
    ]
    # 准备batch_dict
    onnx_output_batch = prepare_batch_dict_from_ort_outputs(
        ort_outputs=onnx_output,
        output_names=output_names,
        batch_size=1,  # 假设batch_size为1
        cls_preds_normalized=False  # 假设分类预测未归一化
    )

    pred_dicts_onnx, recall_dict_onnx = model.post_processing(onnx_output_batch)
    torch.save(pred_dicts_onnx, "pred_dicts_onnx.pt")

    
    torch_batch_out=model(real_input)
    torch_batch_out['cls_preds_normalized']=False
    pred_dicts_torch, recall_dict_torch=model.post_processing(torch_batch_out)
    torch.save(pred_dicts_torch, "pred_dicts_torch.pt")
    torch_numpy=[]
    for tensor in torch_batch_out.values():
        if torch.is_tensor(tensor):
            torch_numpy.append(tensor.detach().cpu().numpy())
        else:
            torch_numpy.append(tensor)

    # np.save('torch_numpy.npy', torch_numpy)  # 默认扩展名是 .npy
    # np.save('onnx_output.npy', onnx_output)  # 默认扩展名是 .npy
    max_error=float('-inf')
    for torch_numpy, onnx_output in zip(torch_numpy, onnx_output):
        max_error = max(np.abs(torch_numpy - onnx_output).max(),max_error)

    # error = np.abs(torch_output - onnx_output[0]).max()
    print(f"最大绝对误差: {max_error:.5e}")  # 科学计数法显示

if __name__ == "__main__":
    main()
