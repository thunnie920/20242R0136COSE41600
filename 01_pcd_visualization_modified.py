import open3d as o3d
import numpy as np
import os
import matplotlib.pyplot as plt

# PCD 파일을 불러와 시각화하고 이미지를 저장하는 함수
def visualize_and_save_pcd(file_path, output_folder, point_size=1.0):
    # PCD 파일 로드
    pcd = o3d.io.read_point_cloud(file_path)
    
    # 시각화 설정 및 이미지 저장
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)  # 창을 표시하지 않고 시각화
    vis.add_geometry(pcd)
    vis.get_render_option().point_size = point_size

    # 카메라 뷰 조정
    view_control = vis.get_view_control()
    
    # 카메라 설정 변경 (예시: 초기화 및 특정 위치로 설정)
    view_control.set_zoom(0.7)  # 줌 비율 조정
    view_control.set_front([0.0, -0.2, -1.0])  # 카메라의 앞쪽 방향 벡터 설정
    view_control.set_lookat([0.0, -1.0, 0.0])  # 카메라의 중심점을 설정
    view_control.set_up([0.0, 1.0, 0.0])  # 카메라의 위쪽 방향 벡터 설정
    
    # 카메라와 시각화 창 초기화
    vis.poll_events()
    vis.update_renderer()
    
    # 파일명에서 확장자를 제거하고 이미지 파일명 설정
    filename = os.path.splitext(os.path.basename(file_path))[0] + ".png"
    save_path = os.path.join(output_folder, filename)
    
    # 이미지를 캡처하고 저장
    vis.capture_screen_image(save_path)
    print(f"Image saved at {save_path}")
    vis.destroy_window()

# 폴더의 모든 PCD 파일을 처리하는 함수
def process_pcd_folder(input_folder, output_folder, point_size=1.0):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 폴더 내 모든 파일 처리
    for filename in os.listdir(input_folder):
        if filename.endswith('.pcd'):
            file_path = os.path.join(input_folder, filename)
            visualize_and_save_pcd(file_path, output_folder, point_size)

# 경로 설정
#file_path = "E:\\Desktop\\selfdrivingCars\\data\\01_straight_walk\\pcd\\pcd_000001.pcd"
input_folder = "E:\\Desktop\\selfdrivingCars\\COSE416_HW1\\data\\01_straight_walk\\pcd"
output_folder = "E:\\Desktop\\selfdrivingCars\\COSE416_HW1\\visualized_pcd\\01_straight_walk"

# 전체 PCD 파일 시각화 및 이미지 저장 실행
process_pcd_folder(input_folder, output_folder, 0.5)
