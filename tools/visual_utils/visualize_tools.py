import numpy as np
import open3d as o3d
import os
import torch
import pyvista as pv
# from visual_utils.open3d_vis_utils import box_colormap


class_names = {
    1: "Pedestrian",
    2: "Car",
    3: "IGV-Full",
    4: "Truck",
    5: "Trailer-Empty",
    6: "Trailer-Full",
    7: "IGV-Empty",
    8: "Crane",
    9: "OtherVehicle",
    10: "Cone",
    11: "ContainerForklift",
    12: "Forklift",
    13: "Lorry",
    14: "ConstructionVehicle",
}

def create_3d_bbox(bbox_3d, color=(1, 0, 0)):
    """创建3D检测框线框"""
    center = bbox_3d[:3]
    length, width, height = bbox_3d[3:6]
    rotation = bbox_3d[6]

    # 计算8个角点
    corners = np.array(
        [
            [length / 2, width / 2, height / 2],
            [length / 2, width / 2, -height / 2],
            [length / 2, -width / 2, height / 2],
            [length / 2, -width / 2, -height / 2],
            [-length / 2, width / 2, height / 2],
            [-length / 2, width / 2, -height / 2],
            [-length / 2, -width / 2, height / 2],
            [-length / 2, -width / 2, -height / 2],
        ]
    )

    # 应用旋转和平移
    # rotation=0
    rot_mat = np.array(
        [
            [np.cos(rotation), -np.sin(rotation), 0],
            [np.sin(rotation), np.cos(rotation), 0],
            [0, 0, np.cos(rotation)],
        ]
    )

    # rot_mat=np.ones(3,3)
    corners = np.dot(corners, rot_mat.T) + center

    # 定义12条边
    lines = [
        [0, 1],
        [0, 2],
        [1, 3],
        [2, 3],
        [4, 5],
        [4, 6],
        [5, 7],
        [6, 7],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ]

    # 创建线框
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color for _ in range(len(lines))])
    return line_set


def offscreen_visualization_array(
    points_array,
    gt_boxes=None,
    ref_boxes=None,
    output_image="point_cloud.png",
    box_colormap=None,
    zoom=0.25,
):

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
    pcd.colors = o3d.utility.Vector3dVector(
        np.ones((points_array.shape[0], 3))
    )  # 所有点设为白色

    bbox_geometries = []
    if gt_boxes is not None:
        for i, bbox in enumerate(gt_boxes):
            # color = cm.plasma(i / len(bboxes))[:3]  # 框的颜色渐变
            color = (1, 0, 0)  # 真值固定为荧光绿 RGB
            label = int(bbox[7])
            # color = box_colormap[label]  # 固定为绿色 RGB
            bbox_geometries.append(create_3d_bbox(bbox, color=color))

    if ref_boxes is not None:
        for i, bbox in enumerate(ref_boxes):
            # color = cm.plasma(i / len(bboxes))[:3]  # 框的颜色渐变
            # color = (0, 1, 0)  # 固定为绿色 RGB
            label = int(bbox[7])
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
    opt.light_on = True  # 启用光照（增强白色点可见性）

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


def visualization_array_pyvista(
    points_array,
    gt_boxes=None,
    ref_boxes=None,
    output_image="point_cloud.png",
    class_names=None,
    box_colormap=None,
    off_screen=False,
):
    if isinstance(points_array, torch.Tensor):
        points_array = points_array.cpu().numpy()
    if isinstance(gt_boxes, torch.Tensor):
        gt_boxes = gt_boxes.cpu().numpy()
    if isinstance(ref_boxes, torch.Tensor):
        ref_boxes = ref_boxes.cpu().numpy()

    assert points_array.shape[1] >= 3, "输入数组需要至少3列（XYZ）"

    xyz = points_array[:, :3]

    # 初始化 plotter
    plotter = pv.Plotter(off_screen=off_screen, window_size=(1920, 1080))
    plotter.set_background("black")

    # 添加点云
    point_cloud = pv.PolyData(xyz)
    plotter.add_points(point_cloud, color="white", point_size=1.0)

    # 绘制bbox函数
    def draw_bbox(center, size, yaw, color=(1, 0, 0), label_text=None,labe_pos_shift=0):
        cube = pv.Cube(
            center=(0, 0, 0), x_length=size[0], y_length=size[1], z_length=size[2]
        )
        cube.rotate_z(np.degrees(yaw), point=(0, 0, 0), inplace=True)
        cube.translate(center, inplace=True)
        plotter.add_mesh(cube, color=color, style="wireframe", line_width=2)
        if label_text is not None:
            label_pos = center + np.array([0, 1.0+labe_pos_shift, size[2] / 2 + 0.5])
            plotter.add_point_labels(
                np.array([label_pos]),
                [label_text],
                font_size=15,
                text_color=color,
                point_color=color,
            )

    # 添加 gt_boxes（红色）
    if gt_boxes is not None:
        for i, bbox in enumerate(gt_boxes):
            center = bbox[:3]
            size = bbox[3:6]
            yaw = bbox[6]
            label = int(bbox[7]) if bbox.shape[0] >= 8 else 0
            name = class_names[label] if class_names else f"gt_{label}"
            draw_bbox(center, size, yaw, color="red",label_text=name,labe_pos_shift=1.5)

    # 添加 ref_boxes（使用颜色映射）
    if ref_boxes is not None:
        for i, bbox in enumerate(ref_boxes):
            center = bbox[:3]
            size = bbox[3:6]
            yaw = bbox[6]
            label = int(bbox[7]) if bbox.shape[0] >= 8 else 0
            score= bbox[8] if bbox.shape[0] >= 9 else 0
            # color = box_colormap[label]
            color="lime"
            name = class_names[label] if class_names else f"pred_{label}"
            label_text=name+f":{score:.2f}"
            draw_bbox(center, size, yaw, color=color, label_text=label_text)

    # 添加坐标轴
    plotter.add_axes(line_width=2)
    
    # 设置视角
    plotter.camera_position = "xy"  # 可以是 'xy', 'xz', 'yz', 'iso'
    plotter.camera.zoom(2.0)  # 放大2倍


    # 添加两行不同颜色的文本（使用规范化坐标定位）
    # plotter.add_text(
    #     "Red is groundtruth",
    #     position=(0.02, 0.95),  # 左上角位置 (x,y)，范围[0,1]
    #     font_size=15,
    #     color="red",            # 红色文本
    #     font="arial",
    #     shadow=True
    # )

    # plotter.add_text(
    #     "Green is Prediction", 
    #     position=(0.02, 0.8),  # 上一行下方
    #     font_size=15,
    #     color="green",          # 绿色文本
    #     font="arial",
    #     shadow=True
    # )
    # 保存图像
    os.makedirs(os.path.dirname(output_image), exist_ok=True)
    # plotter.show()
    plotter.show(screenshot=output_image)
    plotter.close()
