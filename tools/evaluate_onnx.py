import onnxruntime as ort
import numpy as np

# 加载 ONNX 模型
# session = ort.InferenceSession("model/pointpillar_raw.onnx", providers=["CPUExecutionProvider"])

providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
session = ort.InferenceSession("model/pointpillar_raw.onnx", providers=providers)

# 获取所有输入信息

for input_tensor in session.get_inputs():
    print(f"Input Name: {input_tensor.name}")
    print(f"Input Shape: {input_tensor.shape}")
    print(f"Input Type: {input_tensor.type}")


for output_tensor in session.get_outputs():
    print(f"Output Name: {output_tensor.name}")
    print(f"Output Shape: {output_tensor.shape}")
    print(f"Output Type: {output_tensor.type}")
# # 获取输出信息
# for output_tensor in session.get_outputs():
#     print(f"Output Name: {output_tensor.name}")
#     print(f"Output Shape: {output_tensor.shape}")
#     print(f"Output Type: {output_tensor.type}")

# 假设 max_pillars=12000, max_points_per_pillar=32, num_features=4
voxels = np.random.rand(10000, 32, 4).astype(np.float32)
voxel_num = np.random.randint(1, 32, size=(10000,)).astype(np.int32)
voxel_idxs = np.random.randint(0, 100, size=(12000, 4)).astype(np.int32)  # 4D: (batch, z, y, x)

# 准备输入数据
inputs = {
    'voxels': np.random.randn(10000, 32, 4).astype(np.float32),
    'voxel_num': np.array([10000], dtype=np.int32),
    'voxel_idxs': np.random.randint(0, 100, (10000, 4), dtype=np.int32),
    '3': np.array(1, dtype=np.int64)  # 添加第四个输入
}

# 进行推理
outputs = session.run(None, inputs)

# 输出预测框的形状
for name, output in zip(["box_preds", "cls_preds", "dir_preds"], outputs):
    print(f"{name} shape: {output.shape}")