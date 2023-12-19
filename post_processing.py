"""
Template Match:template pool + algorithm pool
"""
import json
import cv2
import numpy as np
import copy
from skimage.metrics import structural_similarity as ssim


def produce_first_template(image_ori, template_path):
    """
    produce the first object template
    :param image_ori: first frame cropped
    :param template_path: template json file path
    :return: numpy.ndarray real_object
    """
    image = cv2.imread(image_ori)
    template_path =template_path
    with open(template_path, 'r') as json_file:
        data = json.load(json_file)

    ori_h, ori_w, c = image.shape
    box_coord = data['box_coord']

    # 定义矩形框的中心点坐标、宽度和高度
    center_x, center_y = ori_w * box_coord[0], ori_h * box_coord[1]
    width, height = ori_w * box_coord[2], ori_h * box_coord[3]

    # 计算矩形框的左上角和右下角坐标
    x1, y1 = int(center_x - width / 2), int(center_y - height / 2)
    x2, y2 = int(center_x + width / 2), int(center_y + height / 2)

    # 画出ROI
    real_object = image[y1:y2, x1:x2]

    return real_object

def resize_image(image, target_size):
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)


def produce_candidates(img_path="", label_path="", draw=False):
    return candidates_path

def find_most_similar_template(first_template, candidate_paths):
    # 读取模板图像
    # template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    template = first_template
    max_ssim = -1
    most_similar_candidate = None

    # 遍历候选图像
    for candidate_path in candidate_paths:
        # 读取候选图像
        candidate = cv2.imread(candidate_path, cv2.IMREAD_GRAYSCALE)

        # 调整候选图像大小与模板图像相同
        candidate = resize_image(candidate, template.shape[::-1])

        # 计算相似性
        similarity_index, _ = ssim(template, candidate, full=True)

        # 更新最相似的图像
        if similarity_index > max_ssim:
            max_ssim = similarity_index
            most_similar_candidate = candidate_path

    return most_similar_candidate, max_ssim

def find_most_similar_candidate_2(template_path, candidate_paths):
    # 读取模板图像
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)

    max_corr = -1
    most_similar_candidate = None

    # 遍历候选图像
    for candidate_path in candidate_paths:
        # 读取候选图像
        candidate = cv2.imread(candidate_path, cv2.IMREAD_GRAYSCALE)

        # 调整候选图像大小与模板图像相同
        candidate = resize_image(candidate, template.shape[::-1])

        # 使用 cv2.matchTemplate 计算归一化互相关
        result = cv2.matchTemplate(candidate, template, cv2.TM_CCOEFF_NORMED)

        # 获取最大的相关性值和其位置
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        # 更新最相似的图像
        if max_val > max_corr:
            max_corr = max_val
            most_similar_candidate = candidate_path

    return most_similar_candidate, max_corr


if __name__ == "__main__":
    template_json = 'runs/predicted_labels_1702630580064.json'
    image_cropped = 'data/imgs_1_trim/frame_000_cropped_2023-12-15-19-56-18_template.png'
    first_template = produce_first_template(image_cropped,template_json)
    # candidate_paths = produce_candidates(draw=True)
    # candidate_paths = []

    most_similar_candidate, similarity_score = find_most_similar_template(first_template, candidate_paths)

    print(f"The most similar candidate is: {most_similar_candidate}")
    print(f"Similarity Score: {similarity_score}")

