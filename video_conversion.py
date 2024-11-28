import cv2
import os

# 경로 설정
image_folder = "E:\\Desktop\\selfdrivingCars\\COSE416_HW1\\visualized_pcd\\02_straight_duck_walk"
output_video = "E:\\Desktop\\selfdrivingCars\\COSE416_HW1\\converted_video\\02_straight_duck_walk.mp4"  # 생성할 비디오 파일 이름
fps = 10  # 초당 프레임 수 (FPS)

def create_video_from_images(image_folder, output_video, fps=30):
    # 이미지 파일 이름 가져오기
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images.sort()  # 파일 이름 순으로 정렬

    # 첫 번째 이미지로 프레임 크기 설정
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, _ = frame.shape

    # VideoWriter 객체 생성
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # mp4v 코덱 사용
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # 이미지들을 비디오에 추가
    for image in images:
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)
        video.write(frame)

    # 작업 완료 후 리소스 해제
    video.release()
    print(f"비디오 저장 완료: {output_video}")

create_video_from_images(image_folder, output_video, fps)
