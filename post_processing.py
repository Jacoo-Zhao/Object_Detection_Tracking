"""
Template Match:template pool + algorithm pool
"""
import csv, os, glob
import json

import cv2
import pandas as pd
from skimage.metrics import structural_similarity as ssim
import numpy as np


def get_csv_file_names(folder_path):
    # 拼接文件夹路径和匹配模式
    pattern = os.path.join(folder_path, '*.csv')

    # 使用glob模块获取匹配的文件列表
    csv_files = glob.glob(pattern)

    # 提取文件名部分
    file_names = [os.path.basename(file) for file in csv_files]

    return file_names


def resize_image(image, target_size):
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)


def calculate_ssim(template, target):
    return ssim(template, target)


def produce_first_template(template_json_path):
    """
    produce the first object template
    :param image_ori: first frame cropped
    :param template_path: template json file path
    :return: real_object:pe_numpy.ndarray
    """
    with open(template_json_path, "r") as json_file:
        data = json.load(json_file)

    image = cv2.imread(data['path'])
    ori_h, ori_w, c = image.shape
    box_coord = data["box_coord"]

    # 定义矩形框的中心点坐标、宽度和高度
    center_x, center_y = ori_w * box_coord[0], ori_h * box_coord[1]
    width, height = ori_w * box_coord[2], ori_h * box_coord[3]

    # 计算矩形框的左上角和右下角坐标
    x1, y1 = int(center_x - width / 2), int(center_y - height / 2)
    x2, y2 = int(center_x + width / 2), int(center_y + height / 2)

    # 画出ROI
    real_object = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    template_orig = [real_object]
    return real_object


def produce_candidates(detected_objects_path=""):
    """
    Return list of numpy_array-formatted candidates from detected objects from one frame.
    :param detected_objects_path: sys root.
    :return: candidates: [ndarray,...,]
    """
    candidates = []
    data_list = []
    box_coord = []

    image = cv2.imread(pd.read_csv(detected_objects_path, usecols=[0], nrows=1).iloc[0, 0])
    ori_h, ori_w, c = image.shape

    with open(detected_objects_path, "r") as file:
        reader = csv.DictReader(file)
        for idx, row in enumerate(reader):
            data_dict = {
                "id": idx,
                "path": row["path"],
                "class_name": row["class_name"],
                "class_id": int(row["class_id"]),
                "confidence": int(row["confidence"]),
                "box_coord": eval(row["box_coord"]),  # 注意：使用eval函数解析字符串为列表
            }
            box_coord.append(data_dict["box_coord"])
            data_list.append(data_dict)

    for box in box_coord:
        # 定义矩形框的中心点坐标、宽度和高度
        center_x, center_y = ori_w * box[0], ori_h * box[1]
        width, height = ori_w * box[2], ori_h * box[3]

        # 计算矩形框的左上角和右下角坐标
        x1, y1 = int(center_x - width / 2), int(center_y - height / 2)
        x2, y2 = int(center_x + width / 2), int(center_y + height / 2)

        candidates.append(cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY))

    return candidates


def find_most_similar_bbox(template_pool, candidates, ssim_threshold):
    similarity_indexs = []
    ids = []
    ssim_values = []

    max_ssim = -1
    most_similar_candidate = None

    # 遍历候选图像
    for id, candidate in enumerate(candidates):
        # 调整候选图像大小与模板图像相同
        candidate = resize_image(candidate, template_pool[0].shape[::-1])

        for template in template_pool:
            # 计算结构相似性指数 (SSIM)
            ssim_value, _ = ssim(template, candidate, full=True)

            ssim_values.append(ssim_value)

        # 计算相似度的平均值
        average_ssim = np.mean(ssim_values)

        # log
        similarity_indexs.append(average_ssim)
        ids.append(id)

        # 如果相似度低于阈值，进行相应处理（这里只是打印一个提醒）
        if average_ssim < ssim_threshold:
            print("Warning: Bounding box similarity below threshold!")

        # 更新最相似的图像
        if average_ssim > max_ssim:
            max_ssim = average_ssim
            most_similar_candidate = candidate
            res = id

    return ids, similarity_indexs, res, most_similar_candidate


def update_template_pool(template_pool, new_template, ssim_threshold=0.8):
    # 判断新的 template 是否与初始 template 相似度较高
    similarity_to_initial = calculate_ssim(template_pool[0], new_template)

    # 如果相似度较高，则添加新的 template 到 template pool
    if similarity_to_initial > ssim_threshold:
        template_pool.append(new_template)

        # 如果 template pool 超过一定大小， 移除相似度最低的 template
        max_pool_size = 10
        if len(template_pool) > max_pool_size:
            # 计算新的 template 与初始 template 的相似度
            similarities = [calculate_ssim(template_pool[0], template) for template in template_pool]
            min_similarity_index = np.argmin(similarities)
            template_pool.pop(min_similarity_index)

    return template_pool


if __name__ == "__main__":
    template_ori = 'runs/predicted_labels_1704431183327_template.json'
    first_template = produce_first_template(template_ori)
    template_pool = [first_template]

    # detected_objects = 'runs/predicted_labels_1704431263155.csv'
    img_list = get_csv_file_names('runs/')

    # 使用检测算法获取每张图片的 bounding box 列表
    for detected_objects_in_img in img_list:
        # print(template_pool)

        print(detected_objects_in_img)

        is_csv_empty = lambda file_path: not any(csv.reader(open(file_path, 'r'), delimiter=','))
        if is_csv_empty('runs/'+detected_objects_in_img):
            print("NO OBJECT DETECTED")
            continue

        candidates = produce_candidates('runs/' + detected_objects_in_img)

        # 找到最相似的 bounding box
        ids, similarity_indexs, res, most_similar_candidate = find_most_similar_bbox(template_pool=template_pool,
                                                                                     candidates=candidates,
                                                                                     ssim_threshold=0.3)
        # print('Similarity:{}'.format(res))
        # 将最相似的 bounding box 更新到 template pool
        template_pool = update_template_pool(template_pool, most_similar_candidate)

        # print(f"The most similar candidate is: {ids}")
        # print(f"Similarity Score: {similarity_indexs}")
        cv2.imshow('Most similar candidate', candidates[res])
        cv2.waitKey(0)

        # for id in ids:
        #     cv2.imshow('Most similar candidate', candidates[id])
        #     cv2.waitKey(0)
