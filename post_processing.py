"""
Template Match:template pool + algorithm pool
"""
import csv
import json

import cv2
import pandas as pd
from skimage.metrics import structural_similarity as ssim


def resize_image(image, target_size):
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)


def produce_first_template(template_path):
    """
    produce the first object template
    :param image_ori: first frame cropped
    :param template_path: template json file path
    :return: real_object:pe_numpy.ndarray
    """
    with open(template_path, "r") as json_file:
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


def find_most_similar_template(template, candidates):

    similarity_indexs = []
    ids = []

    max_ssim = -1
    most_similar_candidate = None
    # 遍历候选图像
    for id, candidate in enumerate(candidates):
        # 调整候选图像大小与模板图像相同
        candidate = resize_image(candidate, template.shape[::-1])

        # 计算相似性
        similarity_index, _ = ssim(template, candidate, full=True)

        # results log
        similarity_indexs.append(similarity_index)
        ids.append(id)

        # 更新最相似的图像
        if similarity_index > max_ssim:
            max_ssim = similarity_index
            most_similar_candidate = candidate
            res = id
    # return res, most_similar_candidate, max_ssim
    return ids, most_similar_candidate, similarity_indexs

def collect_template_pool(confidence, template_json):
    template_pool = []
    template_pool.append('runs/predicted_labels_1704422987243_template.json')


if __name__ == "__main__":
    template_json = 'runs/predicted_labels_1704431183327_template.json'
    detected_objects = 'runs/predicted_labels_1704431263155.csv'

    confidence = 0.7

    first_template = produce_first_template(template_json)
    candidates = produce_candidates(detected_objects)

    # candidate_paths = []

    ids, most_similar_candidate, similarity_indexs = find_most_similar_template(first_template, candidates)
    #
    print(f"The most similar candidate is: {ids}")
    print(f"Similarity Score: {similarity_indexs}")
    # cv2.imshow('Most similar candidate', candidates[ids])
    # cv2.waitKey(0)

    for id in ids:
        cv2.imshow('Most similar candidate', candidates[id])
        cv2.waitKey(0)

    # collect_templete_pool(confidence, template_json)