# 시각화에 필요한 라이브러리 불러오기
import open3d as o3d
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import LinearRegression

# 경로 설정
data_path = "E:\\Desktop\\selfdrivingCars\\data\\06_straight_crawl\\pcd"  # PCD 파일들이 있는 폴더
# data_path = "E:\\Desktop\\selfdrivingCars\\COSE416_HW1_tutorial\\test_data"
output_folder = "E:\\Desktop\\selfdrivingCars\\COSE416_HW1\\visualized_pcd\\06_straight_crawl_sample"  # 저장할 폴더
# output_folder = "E:\\Desktop\\selfdrivingCars\\COSE416_HW1\\visualized_pcd\\test_data_3" 

def calculate_centroid(cluster_pcd):
    # 클러스터의 중심점을 계산
    points = np.asarray(cluster_pcd.points)
    return points.mean(axis=0)

def match_bounding_boxes(prev_centroids, current_centroids, threshold=1.0):
    # 이전과 현재 클러스터 중심점을 비교하여 매칭
    matched_indices = []
    for i, prev_centroid in enumerate(prev_centroids):
        distances = np.linalg.norm(current_centroids - prev_centroid, axis=1)
        if np.min(distances) < threshold:
            matched_indices.append((i, np.argmin(distances)))
    return matched_indices

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
movement_threshold = 0.5  # 움직임 감지 임계값

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
    voxel_size = mean_distance * 0.5
    print(f"Calculated voxel_size: {voxel_size}")

    # Voxel Downsampling 수행
    # voxel_size = 0.2  # 필요에 따라 voxel 크기를 조정하세요.
    downsample_pcd = current_pcd.voxel_down_sample(voxel_size=voxel_size)
    
    # Radius Outlier Removal (ROR) 적용
    #cl, ind = downsample_pcd.remove_radius_outlier(nb_points=optimal_nb_points, radius=1.2)
    cl, ind = downsample_pcd.remove_radius_outlier(nb_points=8, radius=1.5)
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

    clusters = [final_point.select_by_index(np.where(labels == i)[0]) for i in range(max_label + 1)]
    current_centroids = np.array([np.asarray(cluster.points).mean(axis=0) for cluster in clusters])
    current_bboxes = [cluster.get_axis_aligned_bounding_box() for cluster in clusters]

    # 노이즈를 제거하고 각 클러스터에 색상 지정
    colors = plt.get_cmap("tab20")(labels / (max_label + 1 if max_label > 0 else 1))
    colors[labels < 0] = 0  # 노이즈는 검정색으로 표시
    final_point.colors = o3d.utility.Vector3dVector(colors[:, :3])

    if idx < 50:
        # 모든 클러스터의 바운딩 박스를 저장할 리스트
        filtered_bboxes = []  

        for i, cluster_bbox in enumerate(current_bboxes):
            cluster_indices = np.where(labels == i)[0]
            cluster_pcd = final_point.select_by_index(cluster_indices)
            points = np.asarray(cluster_pcd.points)

            # 바운딩 박스를 파란색으로 설정
            cluster_bbox.color = (0, 0, 1)  # 파란색
            filtered_bboxes.append(cluster_bbox)
            
            
        # 50번째 프레임 이전에는 클러스터만 표시
        visualize_with_bounding_boxes(final_point, current_bboxes, output_path)
        # 현재 프레임 정보 저장
        centroid_history[idx] = current_centroids
        bbox_history[idx] = current_bboxes
    else:
        # 50번째 이후에는 현재 프레임과 50 프레임 전의 프레임 비교
        compare_idx = idx - 50  # 비교 대상 프레임
        if compare_idx in centroid_history:
            previous_centroids = centroid_history[compare_idx]
            previous_bboxes = bbox_history[compare_idx]

            for curr_idx, curr_centroid in enumerate(current_centroids):
                # 50 프레임 전의 중심점과 비교하여 움직임 감지
                distances = np.linalg.norm(previous_centroids - curr_centroid, axis=1)
                closest_prev_idx = np.argmin(distances)
                movement = distances[closest_prev_idx]

                if movement > 1:  # movement가 10 이상인 경우만 표시
                    print(f"Movement detected between frame {idx} and frame {compare_idx} in cluster {curr_idx}: {movement}")
                    current_bboxes[curr_idx].color = (1, 0, 0)  # 움직임이 있는 클러스터는 빨간색으로 표시
                else:
                    print(f"No significant movement detected for cluster {curr_idx} (Movement: {movement})")
                    continue  # 움직임이 없는 경우 시각화에서 제외

        # 시각화 및 저장
        visualize_with_bounding_boxes(final_point, current_bboxes, output_path)

        # 현재 프레임 정보 저장
        centroid_history[idx] = current_centroids
        bbox_history[idx] = current_bboxes

print("All visualizations with bounding boxes have been saved.")