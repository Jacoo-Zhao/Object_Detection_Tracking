from ultralytics import YOLO
import cv2

# Single Prediction Template
def yolo_model_demot():
    model = YOLO('yolov8n.pt')  # 加载官方模型
    results = model('https://ultralytics.com/images/bus.jpg')  # 对图像进行预测
    return results


def frame_subtraction_from_video(video_path=''):
    video_path = "../data/videos/1_2023_10_11_14_34_trim.mp4"
    video = cv2.VideoCapture(video_path)
    assert video.isOpened(), "Video could not be read, check with os.path.exists()"

    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    print(frame_count)

    count = 0
    while True:
        tf, frame = video.read()
        if not tf:
            break
        if count % 2 == 0:
            cv2.imwrite('../data/imgs1_trim/1_trim_frame_' + str(count) + '.png', frame)
            print('Frame ' + str(count) + ' saved.')
        count += 1
    return 0
