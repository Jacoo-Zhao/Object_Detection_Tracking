from ultralytics import YOLO
import cv2
import os


# Single Prediction Template
def yolo_model_demot():
    model = YOLO('yolov8n.pt')  # 加载官方模型
    results = model('https://ultralytics.com/images/bus.jpg')  # 对图像进行预测
    return results


def frame_subtraction_from_video(video_path=''):
    video_path = video_path
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
            cv2.imwrite('data/imgs_286_trim_1/frame_' + str(count) + '.png', frame)
            print('Frame ' + str(count) + ' saved.')
        count += 1
    return 0

def rename_files(folder_path):
    # 遍历文件夹中的文件
    for filename in os.listdir(folder_path):
        # 检查文件名是否符合要求
        if filename.startswith("#286_frame_") and filename.endswith(".png"):
            # 提取出n的值
            n = filename.split("_")[-1].split(".")[0]

            # 构造新的文件名
            new_filename = f"frame_{n}.png"

            # 构造完整的文件路径
            old_filepath = os.path.join(folder_path, filename)
            new_filepath = os.path.join(folder_path, new_filename)

            # 重命名文件
            os.rename(old_filepath, new_filepath)
            print(f"已将文件 {filename} 重命名为 {new_filename}")


if __name__ == "__main__":

    frame_subtraction_from_video(video_path='data/videos/286_trim_1.mp4')

    # rename_files(folder_path = "data/imgs_286")

