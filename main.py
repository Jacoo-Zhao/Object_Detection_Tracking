import argparse
import copy
import csv
import datetime
import json
import os
import re

import cv2
import numpy as np

import track


# ANSI颜色代码
class Color:
    GREEN = '\033[92m'  # 绿色
    YELLOW = '\033[93m'  # 黄色
    RED = '\033[91m'  # 红色
    RESET = '\033[0m'  # 重置颜色

def print_colored(message, color):
    print(f"{color}{message}{Color.RESET}")

def print_heading(message, color):
    print("\n", "#" * 40)
    print_colored(message, color)
    print("#" * 40)

def first_img_crop(img_orig_path=''):
    print_heading("Executing Function first_img_crop", Color.GREEN)
    image = cv2.imread(img_orig_path)

    # 显示图像，等待用户选择ROI
    roi = cv2.selectROI('Roi Selection', image, False)
    cv2.destroyAllWindows()

    # 提取ROI
    x, y, w, h = roi
    selected_roi = image[int(y):int(y + h), int(x):int(x + w)]

    # 保存ROI图像
    current_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    output_path = img_orig_path.replace('.png', f'_cropped_{current_time}.png')
    # pdb.set_trace()
    cv2.imwrite(output_path, selected_roi)

    # 显示选择的ROI
    cv2.imshow('Selected ROI', selected_roi)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(f"处理后的图像已保存到: {output_path}")
    return output_path


def class_selection(img_orig_cropped_path="", detected_objects_path="", draw=False):
    print_heading("Executing Function class_selection", Color.GREEN)

    # 读取图像
    image = cv2.imread(img_orig_cropped_path)
    ori_h, ori_w, c = image.shape

    data_list = []  # 用于存储字典的列表
    with open(detected_objects_path, 'r') as file:
        reader = csv.DictReader(file)

        for idx, row in enumerate(reader):
            data_dict = {
                'id': idx,
                'path': row['path'],
                'class_name': row['class_name'],
                'class_id': int(row['class_id']),
                'confidence': int(row['confidence']),
                'box_coord': eval(row['box_coord'])  # 注意：使用eval函数解析字符串为列表
            }

            data_list.append(data_dict)

    classes = []
    imgs = []
    count = len(data_list)
    for i in range(count): imgs.append(copy.deepcopy(image))

    i = 0
    for data_dict in data_list:
        classes.append(data_dict['class_id'])
        print("INFO: Class {} with Id {} detected.".format(data_dict['class_id'], data_dict['id']))

        box_coord = data_dict['box_coord']
        # 定义矩形框的中心点坐标、宽度和高度
        center_x, center_y = ori_w * box_coord[0], ori_h * box_coord[1]
        width, height = ori_w * box_coord[2], ori_h * box_coord[3]

        # 计算矩形框的左上角和右下角坐标
        x1, y1 = int(center_x - width / 2), int(center_y - height / 2)
        x2, y2 = int(center_x + width / 2), int(center_y + height / 2)

        # 画出ROI
        cv2.rectangle(imgs[i], (x1, y1), (x2, y2), (0, 255, 0), 2)  # 绿色边框，线宽为2
        i += 1
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # # 显示图像
    # if draw:
    #     imgs_show = np.hstack(imgs)
    #     cv2.imshow('Image with ROI', imgs_show)
    #     cv2.waitKey(30)
    #     cv2.destroyAllWindows()
    #
    while True:
        try:
            user_input_cls, user_input_id = map(int,
                                                input("输入目标对象类别和id（用空格隔开): ").replace(',', ' ').split())

            if int(user_input_cls) in classes:
                break
            else:
                print("输入不符合规定，请重新输入。")
        except Exception as e:
            print(f"发生错误：{e}")

    print(f"选定的对象类别:{user_input_cls}, 对象ID:{user_input_id}")

    template_path = img_orig_cropped_path.replace('.png', '_template.png')

    cv2.imwrite(template_path, imgs[user_input_id])
    print("Template Image Saved:", template_path)

    template_json_path = detected_objects_path.replace('.csv', '.json')
    with open(template_json_path, 'w') as json_file:
        json.dump(data_list[user_input_id], json_file)

    print(f"Template saved as JSON file: {template_json_path}")

    return data_list[user_input_id], template_json_path

def main(args):

    # 打印结果
    print(f"template_json: {args.template_json_path}")
    print(f"input_folder: {args.input_folder}")

    # 循环处理照片
    for i in range(0, 5, 2):
        filename = os.path.join(args.input_folder, f"frame_{i}.png")
        if os.path.exists(filename):
            os.system(f"python track.py --yolo-model yolov8n --source {filename} --template {args.template_json_path}")
        else:
            print(f"File not found: {filename}")

    # 定义正则表达式模式
    pattern = r"frame_(\d+)\.png"
    # 使用 tqdm 显示循环进度
    for filename in os.listdir(args.input_folder):
        if re.match(pattern, filename):
            file_path = os.path.join(args.input_folder, filename)
            os.system(f"python track.py --yolo-model yolov8n --source {file_path} --template {args.template_json_path}")
        else:
            print(f"Skipping non-PNG file: {filename}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--first_frame_path", help="path to input first frame")
    parser.add_argument("--input_folder", help="Path to the input images folder")
    args = parser.parse_args()

    img_orig_cropped = first_img_crop(img_orig_path=args.first_frame_path)

    print_heading("Executing Function track.run", Color.YELLOW)
    opt = track.parse_opt()
    opt.source = img_orig_cropped
    detected_objects_path = track.run(opt)

    bbox_template, args.template_json_path = class_selection(img_orig_cropped_path=img_orig_cropped,
                                    detected_objects_path=detected_objects_path,
                                    draw=True)

    print_heading("Executing Function track.py", Color.GREEN)
    main(args)


