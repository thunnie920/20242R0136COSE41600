# 시각화에 필요한 라이브러리 불러오기
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

# pcd 파일 불러오기, 필요에 맞게 경로 수정
# 
# file_path = "test_data/1727320101-665925967.pcd"
# file_path = "E:\\Desktop\\selfdrivingCars\\COSE416_HW1_tutorial\\test_data\\1727320101-961578277.pcd"
file_path = "E:\\Desktop\\selfdrivingCars\\data\\01_straight_walk\\pcd\\pcd_000288.pcd"

# pcd 파일 불러오고 시각화하는 함수
def load_and_visualize_pcd(file_path, point_size=1.0):
    # pcd 파일 로드
    pcd = o3d.io.read_point_cloud(file_path)
    print(f"Point cloud has {len(pcd.points)} points.")
    
    # 시각화 설정
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.get_render_option().point_size = point_size
    vis.run()
    vis.destroy_window()

    # 카메라 뷰 조정
    view_control = vis.get_view_control()
    
    # 카메라 설정 변경 (예시: 초기화 및 특정 위치로 설정)
    view_control.set_zoom(0.7)  # 줌 비율 조정
    view_control.set_front([0.0, -0.2, -1.0])  # 카메라의 앞쪽 방향 벡터 설정
    view_control.set_lookat([0.0, -1.0, 0.0])  # 카메라의 중심점을 설정
    view_control.set_up([0.0, 1.0, 0.0])  # 카메라의 위쪽 방향 벡터 설정


# PCD 파일 불러오기 및 데이터 확인 함수
def load_and_inspect_pcd(file_path):
    # PCD 파일 로드
    pcd = o3d.io.read_point_cloud(file_path)
    
    # 점 구름 데이터를 numpy 배열로 변환
    points = np.asarray(pcd.points)
    
    # 점 데이터 개수 및 일부 점 확인
    print(f"Number of points: {len(points)}")
    print("First 5 points:")
    print(points[:5])  # 처음 5개의 점 출력
    
    # 점의 x, y, z 좌표의 범위 확인
    print("X coordinate range:", np.min(points[:, 0]), "to", np.max(points[:, 0]))
    print("Y coordinate range:", np.min(points[:, 1]), "to", np.max(points[:, 1]))
    print("Z coordinate range:", np.min(points[:, 2]), "to", np.max(points[:, 2]))

# pcd 시각화 테스트
load_and_visualize_pcd(file_path, 0.5)
load_and_inspect_pcd(file_path)
