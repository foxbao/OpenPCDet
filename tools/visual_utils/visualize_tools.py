import numpy as np
import open3d as o3d
import os
import torch

from visual_utils.open3d_vis_utils import box_colormap

def create_3d_box(center, size, rotation_matrix=np.eye(3), color=[1, 0, 0]):
    """
    创建3D边界框（8个顶点+12条边）
    参数:
        center: [x,y,z] 框中心坐标
        size: [长,宽,高]
        rotation_matrix: 3x3旋转矩阵
        color: RGB颜色值
    """
    # 计算半长宽高
    l, w, h = size[0]/2, size[1]/2, size[2]/2
    
    # 定义8个顶点的相对坐标
    vertices = np.array([
        [-l, -w, -h], [l, -w, -h], [l, w, -h], [-l, w, -h],
        [-l, -w, h], [l, -w, h], [l, w, h], [-l, w, h]
    ])
    
    # 应用旋转和平移
    vertices = vertices @ rotation_matrix.T + center
    
    # 定义12条边的连接关系
    lines = [
        [0,1], [1,2], [2,3], [3,0],  # 底面
        [4,5], [5,6], [6,7], [7,4],  # 顶面
        [0,4], [1,5], [2,6], [3,7]   # 侧面
    ]
    
    # 创建线框
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(vertices)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color for _ in range(len(lines))])
    
    return line_set

def create_3d_bbox(bbox_3d, color=(1, 0, 0)):
    """创建3D检测框线框"""
    center = bbox_3d[:3]
    length, width, height = bbox_3d[3:6]
    rotation = bbox_3d[6]
    
    # 计算8个角点
    corners = np.array([
        [length/2, width/2, height/2],
        [length/2, width/2, -height/2],
        [length/2, -width/2, height/2],
        [length/2, -width/2, -height/2],
        [-length/2, width/2, height/2],
        [-length/2, width/2, -height/2],
        [-length/2, -width/2, height/2],
        [-length/2, -width/2, -height/2]
    ])
    
    # 应用旋转和平移
    # rotation=0
    rot_mat = np.array([
        [np.cos(rotation), -np.sin(rotation), 0],
        [np.sin(rotation), np.cos(rotation), 0],
        [0, 0, np.cos(rotation)]
    ])

    # rot_mat=np.ones(3,3)
    corners = np.dot(corners, rot_mat.T) + center
    
    # 定义12条边
    lines = [
        [0, 1], [0, 2], [1, 3], [2, 3],
        [4, 5], [4, 6], [5, 7], [6, 7],
        [0, 4], [1, 5], [2, 6], [3, 7]]
    
    # 创建线框
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color for _ in range(len(lines))])
    return line_set

def offscreen_visualization_array(points_array, gt_boxes=None,ref_boxes=None,output_image="point_cloud.png",zoom=0.25):

    if isinstance(points_array, torch.Tensor):
        points_array = points_array.cpu().numpy()
    if isinstance(gt_boxes, torch.Tensor):
        gt_boxes = gt_boxes.cpu().numpy()
    if isinstance(ref_boxes, torch.Tensor):
        ref_boxes = ref_boxes.cpu().numpy()
    assert points_array.shape[1] >= 4, "输入数组需要至少4列（XYZ+强度）"

    xyz = points_array[:, :3]
    # 读取点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(np.ones((points_array.shape[0], 3)))  # 所有点设为白色

    bbox_geometries = []
    if gt_boxes is not None:
        for i, bbox in enumerate(gt_boxes):
            # color = cm.plasma(i / len(bboxes))[:3]  # 框的颜色渐变
            color = (1, 0, 0)  # 真值固定为荧光绿 RGB
            label=int(bbox[7])
            # color = box_colormap[label]  # 固定为绿色 RGB
            bbox_geometries.append(create_3d_bbox(bbox, color=color))

    if ref_boxes is not None:
        for i, bbox in enumerate(ref_boxes):
            # color = cm.plasma(i / len(bboxes))[:3]  # 框的颜色渐变
            # color = (0, 1, 0)  # 固定为绿色 RGB
            label=int(bbox[7])
            color = box_colormap[label]  # 固定为绿色 RGB
            bbox_geometries.append(create_3d_bbox(bbox, color=color))
    
    # 添加坐标系
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2)

    # 创建离屏可视化器
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)  # 不可见窗口
    
    # 添加几何体
    vis.add_geometry(pcd)
    vis.add_geometry(coordinate_frame)
    for bbox in bbox_geometries:
        vis.add_geometry(bbox)

    
    # 渲染设置
    opt = vis.get_render_option()
    opt.background_color = np.array([0, 0, 0])  # 黑色背景
    opt.point_size = 2.0
    opt.light_on = True   # 启用光照（增强白色点可见性）
    
    # 调整视图
    ctr = vis.get_view_control()
    ctr.set_front([0, 0, -1])
    ctr.set_up([0, -1, 0])
    ctr.set_lookat(pcd.get_center())  # ⭐️ 核心：焦点对准点云中心
    ctr.set_zoom(zoom)
    
    # 强制渲染
    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    
    # 保存图像
    # print(output_image)
    # output_path=folder+"/"+output_image+".png"
    os.makedirs(os.path.dirname(output_image), exist_ok=True)  # 自动创建文件夹
    vis.capture_screen_image(output_image, do_render=True)
    vis.destroy_window()