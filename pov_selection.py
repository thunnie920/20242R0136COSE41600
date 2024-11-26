# 시각화에 필요한 라이브러리 불러오기
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

# file_path = "E:\\Desktop\\selfdrivingCars\\COSE416_HW1_tutorial\\test_data\\1727320101-961578277.pcd"
file_path = "E:\\Desktop\\selfdrivingCars\\data\\01_straight_walk\\pcd\\pcd_000188.pcd"

"""
이미지 캡쳐 각도 설정
"""

# PCD 파일 읽기
original_pcd = o3d.io.read_point_cloud(file_path)

# Voxel Downsampling 수행
voxel_size = 0.1  # 필요에 따라 voxel 크기를 조정
downsample_pcd = original_pcd.voxel_down_sample(voxel_size=voxel_size)

# Radius Outlier Removal (ROR) 적용
cl, ind = downsample_pcd.remove_radius_outlier(nb_points=6, radius=1.2)
ror_pcd = downsample_pcd.select_by_index(ind)

# RANSAC을 사용하여 평면 추정
plane_model, inliers = ror_pcd.segment_plane(distance_threshold=0.1,
                                             ransac_n=3,
                                             num_iterations=2000)

# 도로에 속하지 않는 포인트 (outliers) 추출
final_point = ror_pcd.select_by_index(inliers, invert=True)

# DBSCAN 클러스터링 적용
with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    labels = np.array(final_point.cluster_dbscan(eps=0.6, min_points=11, print_progress=True))

# 각 클러스터를 색으로 표시
max_label = labels.max()
print(f"point cloud has {max_label + 1} clusters")

# 노이즈를 제거하고 각 클러스터에 색상 지정
colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = 0  # 노이즈는 검정색으로 표시
final_point.colors = o3d.utility.Vector3dVector(colors[:, :3])

camera_param_file = "E:\\Desktop\\selfdrivingCars\\COSE416_HW1\\camera_params.json"


def save_camera_parameters(pcd, file_name=camera_param_file, point_size=1.0, width=1920, height=1080):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height)
    vis.add_geometry(pcd)
    vis.get_render_option().point_size = point_size

    # 카메라 시야를 사용자가 설정하도록 함
    print("윈도우에서 원하는 시야를 설정한 뒤 'q'를 눌러 종료하세요.")
    vis.run()

    # 현재 카메라 파라미터 가져오기
    view_control = vis.get_view_control()
    camera_params = view_control.convert_to_pinhole_camera_parameters()

    # 파라미터 저장
    o3d.io.write_pinhole_camera_parameters(file_name, camera_params)
    print(f"카메라 파라미터가 '{file_name}'에 저장되었습니다.")

    vis.destroy_window()

# 예시: 포인트 클라우드 불러오기

save_camera_parameters(final_point, file_name=camera_param_file, point_size=2.0, width=1920, height=1080)
