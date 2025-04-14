# import numpy as np
# import open3d as o3d

# def create_oriented_bbox_from_array(bbox_array):
#     """
#     从长度为 7 的数组创建带方向的 3D 边界框
#     :param bbox_array: 长度为 7 的数组，格式为 [center_x, center_y, center_z, size_x, size_y, size_z, yaw]
#     :return: open3d.geometry.OrientedBoundingBox 对象
#     """
#     # 解析数组
#     center = bbox_array[:3]  # 前 3 个元素是中心点
#     size = bbox_array[3:6]   # 中间 3 个元素是尺寸
#     yaw = bbox_array[6]      # 最后一个元素是 yaw 角（弧度）

#     # 创建旋转矩阵（绕 z 轴旋转 yaw 角）
#     rotation_matrix = np.array([
#         [np.cos(yaw), -np.sin(yaw), 0],
#         [np.sin(yaw), np.cos(yaw), 0],
#         [0, 0, 1]
#     ])
    
#     # 创建带方向的边界框
#     bbox = o3d.geometry.OrientedBoundingBox(center, rotation_matrix, size)
#     bbox.color = [1, 0, 0]  # 设置边界框颜色为红色
#     return bbox

# # 示例：点云数据
# points = np.random.rand(1000, 3)  # 随机生成 1000 个点
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(points)

# # 示例：定义长度为 7 的边界框数组
# bbox_array = np.array([0.5, 0.5, 0.5,  # 中心点 (x, y, z)
#                        0.4, 0.3, 0.2,  # 尺寸 (长, 宽, 高)
#                        np.pi / 4])     # 方向 (yaw 角，弧度）

# # 创建边界框
# bbox = create_oriented_bbox_from_array(bbox_array)

# # 可视化点云和边界框
# o3d.visualization.draw_geometries([pcd, bbox])