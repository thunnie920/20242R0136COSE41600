# 시각화에 필요한 라이브러리 불러오기
import open3d as o3d
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import LinearRegression
from scipy.spatial import KDTree

# 경로 설정
data_path = "E:\\Desktop\\selfdrivingCars\\data\\06_straight_crawl\\pcd"  # PCD 파일들이 있는 폴더
# data_path = "E:\\Desktop\\selfdrivingCars\\COSE416_HW1_tutorial\\test_data"
output_folder = "E:\\Desktop\\selfdrivingCars\\COSE416_HW1\\visualized_pcd\\ver13\\06_straight_crawl2" # 저장할 폴더
# output_folder = "E:\\Desktop\\selfdrivingCars\\COSE416_HW1\\visualized_pcd\\test_data_3"


def apply_camera_view(vis, filename="camera_params.json"):
    # 저장된 카메라 매개변수 불러오기
    camera_params = o3d.io.read_pinhole_camera_parameters(filename)
    vis.create_window(width=camera_params.intrinsic.width, height=camera_params.intrinsic.height)
    
    # ViewControl에 적용
    view_control = vis.get_view_control()
    view_control.convert_from_pinhole_camera_parameters(camera_params, allow_arbitrary=True)
    print(f"Camera parameters loaded from {filename}")


def visualize_with_bounding_boxes(pcd, bounding_boxes, window_name="Filtered Clusters and Bounding Boxes", point_size=1.0, filename="camera_params.json"):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, height=1080, width=1920)
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

    vis.poll_events()
    vis.update_renderer()

    # 캡처된 이미지를 numpy 배열로 변환
    image = vis.capture_screen_float_buffer(do_render=True)
    vis.destroy_window()

    # 이미지 저장
    image_array = np.asarray(image) * 255  # 0-1 범위의 float 이미지를 0-255 범위로 변환
    plt.imsave(output_path, image_array.astype(np.uint8))
    print(f"Visualization saved at '{output_path}'.")

# test_data 폴더의 모든 PCD 파일 처리
pcd_files = [f for f in os.listdir(data_path) if f.endswith('.pcd')]

centroid_history = {}  # 이전 프레임의 중심점 정보를 저장할 딕셔너리
bbox_history = {}  # 이전 프레임의 바운딩 박스 정보를 저장할 딕셔너리

for idx, pcd_file in enumerate(pcd_files):
    input_path = os.path.join(data_path, pcd_file)
    output_path = os.path.join(output_folder, pcd_file.replace('.pcd', '.png'))

    # PCD 파일 읽기
    current_pcd = o3d.io.read_point_cloud(input_path)
    points = np.asarray(current_pcd.points)

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
    downsample_pcd = current_pcd.voxel_down_sample(voxel_size=voxel_size)
    
    # Radius Outlier Removal (ROR) 적용
    cl, ind = downsample_pcd.remove_radius_outlier(nb_points=8, radius=1.5)
    ror_pcd = downsample_pcd.select_by_index(ind)
        
    # RANSAC을 사용하여 초기 평면 추정
    plane_model, inliers = ror_pcd.segment_plane(distance_threshold=0.1,
                                                ransac_n=3,
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

    # 사용자 지정 파라미터
    eps = 0.1  # 사용자가 지정한 eps 값
    min_points = 5  # 사용자가 지정한 min_points 값

    # 클러스터링 수행
    labels = np.array(final_point.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))

    # 노이즈를 제외한 클러스터 개수 계산
    cluster_count = len(set(labels)) - (1 if -1 in labels else 0)

    # 결과 출력
    print(f"eps: {eps}, min_points: {min_points}, 클러스터 수: {cluster_count}")
    
    # 각 클러스터를 색으로 표시
    max_label = labels.max()
    print(f"point cloud has {max_label + 1} clusters")

    clusters = [final_point.select_by_index(np.where(labels == i)[0]) for i in range(max_label + 1)]
    
    # 노이즈를 제거하고 각 클러스터에 색상 지정
    colors = plt.get_cmap("tab20")(labels / (max_label + 1 if max_label > 0 else 1))
    colors[labels < 0] = 256  # 노이즈는 흰색으로 표시
    final_point.colors = o3d.utility.Vector3dVector(colors[:, :3])
    
    # 조건에 따라 사람 클러스터 필터링
    min_points_in_cluster = 10
    max_points_in_cluster = 65
    
    min_height = 0.2  # 최소 높이 (사람)
    max_height = 2.0  # 최대 높이 (사람)
    min_width_x = 0.2
    max_width_x = 0.8
    min_width_y = 0.2
    max_width_y = 0.8
    min_density = 0.1  # 밀도의 최소값 (필터링 기준)

    person_bboxes = []  # 필터링된 클러스터의 바운딩 박스를 저장
    person_centroids = []  # 필터링된 클러스터의 중심점 저장

    for i in range(max_label + 1):
        cluster_indices = np.where(labels == i)[0]
        cluster_pcd = final_point.select_by_index(cluster_indices)
        cluster_points = np.asarray(cluster_pcd.points)

        # Z 값 범위 확인 (지면에서의 최소/최대 높이)
        z_min, z_max = cluster_points[:, 2].min(), cluster_points[:, 2].max()
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

        # 밀도 계산
        bbox = cluster_pcd.get_axis_aligned_bounding_box()
        volume = bbox.volume()
        density = len(cluster_points) / volume if volume > 0 else 0

        if density < min_density:
            continue

        bbox.color = (1, 0, 0)  # 빨간색으로 표시
        person_bboxes.append(bbox)

        # 조건을 만족하는 클러스터만 저장
        person_centroids.append(cluster_points.mean(axis=0))  # 중심점 저장

        
    # 현재 프레임 정보 저장
    centroid_history[idx] = np.array(person_centroids)
    bbox_history[idx] = person_bboxes
    
    if idx < 40:
        visualize_with_bounding_boxes(final_point, [], output_path)
        
    else:
        # 이전 프레임과 비교 처리
        compare_idx = idx - 40
        if compare_idx in centroid_history:
            previous_centroids = centroid_history[compare_idx]

            # 이동(Movement) 계산
            movements = []
            for curr_idx, curr_centroid in enumerate(person_centroids):
                distances = np.linalg.norm(previous_centroids - curr_centroid, axis=1)
                min_distance_idx = np.argmin(distances)
                min_distance = distances[min_distance_idx]
                movements.append((curr_idx, min_distance_idx, min_distance))

            # 이동 정보 출력
            print("Movement Information:")
            for curr_idx, prev_idx, distance in movements:
                print(f"Cluster {curr_idx} moved from previous Cluster {prev_idx} by {distance:.2f} units.")

            # 이동량 기준으로 정렬하여 상위 5개 선택
            movements_sorted = sorted(movements, key=lambda x: x[2], reverse=True)  # 이동량(distance)을 기준으로 내림차순 정렬
            top_moving_bboxes = [person_bboxes[movement[0]] for movement in movements_sorted[:10]]

            # 상위 5개의 이동 클러스터를 빨간색으로 설정
            for bbox in top_moving_bboxes:
                bbox.color = (1, 0, 0)  # 빨간색으로 설정
            visualize_with_bounding_boxes(final_point, top_moving_bboxes, output_path)

print("All visualizations with bounding boxes have been saved.")