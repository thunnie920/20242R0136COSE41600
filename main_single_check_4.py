# 시각화에 필요한 라이브러리 불러오기
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import LinearRegression

# pcd 파일 불러오기, 필요에 맞게 경로 수정
# file_path = "test_data/1727320101-665925967.pcd"
file_path = "E:\\Desktop\\selfdrivingCars\\data\\04_zigzag_walk\\pcd\\pcd_000199.pcd"
#file_path = "E:\\Desktop\\selfdrivingCars\\data\\04_zigzag_walk\\pcd\\pcd_000001.pcd"
# file_path = "E:\\Desktop\\selfdrivingCars\\COSE416_HW1_tutorial\\test_data\\1727320101-961578277.pcd"

"""
클러스터 구분 확인용
"""
# 1. 지형 기반 땅 모델 생성 (KD-Tree 기반 지역적 평면 모델)
def generate_ground_model(pcd, cell_size=1.0):
    points = np.asarray(pcd.points)
    ground_model = {}
    
    # 포인트 클라우드를 cell_size로 슬라이싱
    for x in np.arange(points[:, 0].min(), points[:, 0].max(), cell_size):
        for y in np.arange(points[:, 1].min(), points[:, 1].max(), cell_size):
            # 해당 셀의 포인트 추출
            mask = (points[:, 0] >= x) & (points[:, 0] < x + cell_size) & \
                   (points[:, 1] >= y) & (points[:, 1] < y + cell_size)
            cell_points = points[mask]
            
            if len(cell_points) < 10:  # 최소 점 개수 확인
                continue
            
            # RANSAC으로 해당 셀의 평면 추정
            ransac = RANSACRegressor()
            X = cell_points[:, :2]  # X와 Y
            Z = cell_points[:, 2]  # Z
            ransac.fit(X, Z)
            
            # 평면 방정식 저장 (z = ax + by + c)
            coef = ransac.estimator_.coef_
            intercept = ransac.estimator_.intercept_
            ground_model[(x, y)] = (coef, intercept)
    
    return ground_model

# 2. 땅 접촉 여부 판단
def is_cluster_on_ground(cluster_points, ground_model, cell_size=1.0, max_distance=0.2):
    for point in cluster_points:
        x, y, z = point
        cell_x = int(x // cell_size) * cell_size
        cell_y = int(y // cell_size) * cell_size
        
        if (cell_x, cell_y) in ground_model:
            coef, intercept = ground_model[(cell_x, cell_y)]
            ground_z = coef[0] * x + coef[1] * y + intercept
            if abs(z - ground_z) <= max_distance:  # z 값 차이가 임계값 이내인지 확인
                return True
    return False


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
voxel_size = mean_distance * 0.1
print(f"Calculated voxel_size: {voxel_size}")

# Voxel Downsampling 수행
# voxel_size = 0.2  # 필요에 따라 voxel 크기를 조정하세요.
downsample_pcd = original_pcd.voxel_down_sample(voxel_size=voxel_size)


# Radius Outlier Removal (ROR) 적용
#cl, ind = downsample_pcd.remove_radius_outlier(nb_points=optimal_nb_points, radius=1.2)
cl, ind = downsample_pcd.remove_radius_outlier(nb_points=5, radius=2.0)
ror_pcd = downsample_pcd.select_by_index(ind)

# RANSAC을 사용하여 초기 평면 추정
plane_model, inliers = ror_pcd.segment_plane(distance_threshold=0.1,
                                             ransac_n=5,
                                             num_iterations=2000)

[a, b, c, d] = plane_model  # 평면 방정식 계수
ground_z_mean = -d / c      # z = -(ax + by + d)/c

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
eps_values = [0.1, 0.2, 0.3]  # 기존보다 작은 값으로 설정
min_points_values = [5, 10]  # 클러스터를 더 세밀하게 정의

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
colors[labels < 0] = 256  # 노이즈는 검정색으로 표시
final_point.colors = o3d.utility.Vector3dVector(colors[:, :3])

# 조건에 따라 클러스터 필터링
min_height = 0.2  # 최소 높이 (사람)
max_height = 2.0  # 최대 높이 (사람)
    
min_z_value = -1.0 # 지면에서부터의 최소 높이
max_z_value = 6.0  # 지면에서부터의 최대 높이
min_width_x = 0.2
max_width_x = 0.8

min_width_y = 0.2
max_width_y = 0.8
    
# 땅 모델 생성
cell_size = 1.0  # 셀 크기 (필요에 따라 조정 가능)
ground_model = generate_ground_model(final_point, cell_size=cell_size)

# 필터링된 클러스터만 처리
filtered_centroids = []
filtered_bboxes = []

for i in range(max_label + 1):
    # 클러스터 포인트 추출
    cluster_indices = np.where(labels == i)[0]
    cluster_pcd = final_point.select_by_index(cluster_indices)
    cluster_points = np.asarray(cluster_pcd.points)
    
    # 1. 땅 접촉 여부 확인
    if not is_cluster_on_ground(cluster_points, ground_model, cell_size=1.0, max_distance=0.5):
        continue
    
    # 2. 기존 필터링 조건 (높이, 너비 등)
    # Z 값 범위 확인 (지면에서의 최소/최대 높이)
    z_min, z_max = cluster_points[:, 2].min(), cluster_points[:, 2].max()
    z_center = (z_min + z_max) / 2

    # 상대적 Z축 값으로 필터링
    relative_z_min = z_min - ground_z_mean
    relative_z_max = z_max - ground_z_mean
    if not (min_height <= relative_z_max - relative_z_min <= max_height):
        continue
    
    # 높이 계산 (클러스터 내 Z축 범위)
    height = z_max - z_min
    if not (min_height <= height <= max_height): 
        continue

    # X 축 너비 계산
    x_min, x_max = cluster_points[:, 0].min(), cluster_points[:, 0].max()
    width_x = x_max - x_min
    if not (min_width_x <= width_x <= max_width_x): 
        continue

    # Y 축 깊이 계산
    y_min, y_max = cluster_points[:, 1].min(), cluster_points[:, 1].max()
    width_y = y_max - y_min
    if not (min_width_y <= width_y <= max_width_y):
        continue
    
    bbox = cluster_pcd.get_axis_aligned_bounding_box()
    bbox.color = (1, 0, 0)  # 빨간색
    filtered_bboxes.append(bbox)
    

def apply_camera_view(vis, filename="camera_params.json"):
    # 저장된 카메라 매개변수 불러오기
    camera_params = o3d.io.read_pinhole_camera_parameters(filename)
    vis.create_window(width=camera_params.intrinsic.width, height=camera_params.intrinsic.height)
    
    # ViewControl에 적용
    view_control = vis.get_view_control()
    view_control.convert_from_pinhole_camera_parameters(camera_params, allow_arbitrary=True)
    print(f"Camera parameters loaded from {filename}")

def visualize_with_bounding_boxes(pcd, bounding_boxes, window_name="Filtered Clusters and Bounding Boxes", point_size=1.0, width=1920, height=1080, filename="camera_params.json"):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, height=height, width=width)
    vis.add_geometry(pcd)
    for bbox in bounding_boxes:
        vis.add_geometry(bbox)

    # 렌더링 옵션 설정 (포인트 크기)
    render_option = vis.get_render_option()
    render_option.point_size = point_size

    try:
        # 기존 시야각 적용
        apply_camera_view(vis, filename)
    except Exception as e:
        print(f"Could not apply saved camera view: {e}")
        print("Using default view.")

    vis.run()
    vis.destroy_window()

# 시각화 (포인트 크기를 원하는 크기로 조절 가능)
visualize_with_bounding_boxes(final_point, filtered_bboxes, point_size=2.0, width=1920, height=1080, filename="camera_params.json")
