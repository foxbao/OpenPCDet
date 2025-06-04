import open3d as o3d

pcd = o3d.io.read_point_cloud("data/.pcd")
vis = o3d.visualization.VisualizerWithEditing()
vis.create_window()
vis.add_geometry(pcd)
vis.run()  # Pick points with [shift + left click], then close window
vis.destroy_window()
picked_ids = vis.get_picked_points()
print("Picked point indices:", picked_ids)
