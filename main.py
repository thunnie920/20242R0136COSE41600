# 시각화에 필요한 라이브러리 불러오기
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import LinearRegression

# pcd 파일 불러오기, 필요에 맞게 경로 수정
# file_path = "test_data/1727320101-665925967.pcd"
# file_path = "E:\\Desktop\\selfdrivingCars\\COSE416_HW1\\data\\01_straight_walk\\pcd\\pcd_000002.pcd"
file_path = "E:\\Desktop\\selfdrivingCars\\COSE416_HW1_tutorial\\test_data\\1727320101-665925967.pcd"

# PCD 파일 읽기
original_pcd = o3d.io.read_point_cloud(file_path)
points = np.asarray(original_pcd.points)

# 점 수가 많을 경우 일부만 랜덤하게 샘플링
sample_size = min(10000, len(points))  # 최대 10,000개의 점만 샘플링
sampled_points = points[np.random.choice(points.shape[0], sample_size, replace=False)]

# KNN 기반 거리 계산
nbrs = NearestNeighbors(n_neighbors=2).fit(sampled_points)
distances, _ = nbrs.kneighbors(sampled_points)

# 평균 거리 계산 (자기 자신과의 거리는 제외)
mean_distance = np.mean(distances[:, 1])

# voxel_size를 평균 거리의 비율로 설정
voxel_size = mean_distance * 0.5
print(f"Calculated voxel_size: {voxel_size}")

# Voxel Downsampling 수행
# voxel_size = 0.2  # 필요에 따라 voxel 크기를 조정하세요.
downsample_pcd = original_pcd.voxel_down_sample(voxel_size=voxel_size)

# 탐색할 nb_points 값 범위
nb_points_range = range(2, 15)
radius = 1.2  # 고정된 반경

optimal_nb_points = 0
best_ratio = 0

for nb_points in nb_points_range:
    _, ind = downsample_pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)
    inlier_ratio = len(ind) / len(downsample_pcd.points)  # 이상치 제거 후 남은 점의 비율

    if inlier_ratio > best_ratio:
        best_ratio = inlier_ratio
        optimal_nb_points = nb_points

print(f"최적의 nb_points: {optimal_nb_points}, 비율: {best_ratio:.2f}")

# Radius Outlier Removal (ROR) 적용
cl, ind = downsample_pcd.remove_radius_outlier(nb_points=optimal_nb_points, radius=1.2)
ror_pcd = downsample_pcd.select_by_index(ind)

# RANSAC을 사용하여 초기 평면 추정
plane_model, inliers = ror_pcd.segment_plane(distance_threshold=0.1,
                                             ransac_n=3,
                                             num_iterations=2000)

# 인라이어 점 추출
inlier_cloud = downsample_pcd.select_by_index(inliers)
inlier_points = np.asarray(inlier_cloud.points)

# 로컬 최적화 단계
# X, Y, Z 좌표를 분리
X = inlier_points[:, :2]  # X와 Y를 feature로 사용
Z = inlier_points[:, 2]   # Z는 label

# 초기 모델을 기반으로 한 Linear Regression을 통한 로컬 최적화
local_model = LinearRegression().fit(X, Z)

# 업데이트된 평면 방정식 구하기
normal_vector = np.array([-local_model.coef_[0], -local_model.coef_[1], 1])
normal_vector /= np.linalg.norm(normal_vector)
distance = -local_model.intercept_

# 최적화된 평면 모델 출력
print(f"로컬 최적화 후 평면 모델: normal={normal_vector}, distance={distance}")

# 도로에 속하지 않는 포인트 (outliers) 추출
final_point = ror_pcd.select_by_index(inliers, invert=True)

# 포인트 클라우드를 NumPy 배열로 변환
points = np.asarray(final_point.points)

# 기존 DBSCAN 적용 코드 전에 파라미터 탐색 추가
eps_values = [0.2, 0.3, 0.4, 0.5]
min_points_values = [5, 10, 15, 20]

best_labels = None
best_eps = 0
best_min_points = 0
best_cluster_count = 0

# 최적의 eps와 min_points 탐색
for eps in eps_values:
    for min_points in min_points_values:
        labels = np.array(final_point.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
        cluster_count = labels.max() + 1
        if cluster_count > best_cluster_count:  # 클러스터 수가 더 많으면 업데이트
            best_cluster_count = cluster_count
            best_labels = labels
            best_eps = eps
            best_min_points = min_points

print(f"최적의 eps: {best_eps}, 최적의 min_points: {best_min_points}")
print(f"최종 클러스터 수: {best_cluster_count}")

# 최적의 파라미터를 사용하여 최종 DBSCAN 적용
labels = best_labels if best_labels is not None else np.array(final_point.cluster_dbscan(eps=best_eps, min_points=best_min_points, print_progress=True))

# 각 클러스터를 색으로 표시
max_label = labels.max()
print(f"point cloud has {max_label + 1} clusters")


# 노이즈를 제거하고 각 클러스터에 색상 지정
colors = plt.get_cmap("tab20")(labels / (max_label + 1 if max_label > 0 else 1))
colors[labels < 0] = 0  # 노이즈는 검정색으로 표시
final_point.colors = o3d.utility.Vector3dVector(colors[:, :3])

# 필터링 기준 1. 클러스터 내 최대 최소 포인트 수
min_points_in_cluster = 5   # 클러스터 내 최소 포인트 수
max_points_in_cluster = 40  # 클러스터 내 최대 포인트 수

# 필터링 기준 2. 클러스터 내 최소 최대 Z값
min_z_value = -1.5    # 클러스터 내 최소 Z값
max_z_value = 2.5   # 클러스터 내 최대 Z값

# 필터링 기준 3. 클러스터 내 최소 최대 Z값 차이
min_height = 0.5   # Z값 차이의 최소값
max_height = 2.0   # Z값 차이의 최대값

max_distance = 30.0  # 원점으로부터의 최대 거리

# 1번, 2번, 3번 조건을 모두 만족하는 클러스터 필터링 및 바운딩 박스 생성
bboxes_1234 = []
for i in range(max_label + 1):
    cluster_indices = np.where(labels == i)[0]
    if min_points_in_cluster <= len(cluster_indices) <= max_points_in_cluster:
        cluster_pcd = final_point.select_by_index(cluster_indices)
        points = np.asarray(cluster_pcd.points)
        z_values = points[:, 2]  # Z값 추출
        z_min = z_values.min()
        z_max = z_values.max()
        if min_z_value <= z_min and z_max <= max_z_value:
            height_diff = z_max - z_min
            if min_height <= height_diff <= max_height:
                distances = np.linalg.norm(points, axis=1)
                if distances.max() <= max_distance:
                    bbox = cluster_pcd.get_axis_aligned_bounding_box()
                    bbox.color = (1, 0, 0) 
                    bboxes_1234.append(bbox)


# 포인트 클라우드 및 바운딩 박스를 시각화하는 함수
def visualize_with_bounding_boxes(pcd, bounding_boxes, window_name="Filtered Clusters and Bounding Boxes", point_size=1.0):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name)
    vis.add_geometry(pcd)
    for bbox in bounding_boxes:
        vis.add_geometry(bbox)
    vis.get_render_option().point_size = point_size
    vis.run()
    vis.destroy_window()

# 시각화 (포인트 크기를 원하는 크기로 조절 가능)
visualize_with_bounding_boxes(final_point, bboxes_1234, point_size=2.0)
