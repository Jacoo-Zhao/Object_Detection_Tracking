import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage import feature
from scipy.spatial.distance import cosine
import imagehash
from PIL import Image
from skimage import io, color
import time

from nearpy import Engine
from nearpy.hashes import RandomBinaryProjections


def compare_img_hist(img1, img2):
    # Get the histogram data of image 1, then using normalize the picture for better compare
    img1_hist = cv2.calcHist([img1], [1], None, [256], [0, 256])
    img1_hist = cv2.normalize(img1_hist, img1_hist, 0, 1, cv2.NORM_MINMAX, -1)

    img2_hist = cv2.calcHist([img2], [1], None, [256], [0, 256])
    img2_hist = cv2.normalize(img2_hist, img2_hist, 0, 1, cv2.NORM_MINMAX, -1)
    similarity = cv2.compareHist(img1_hist, img2_hist, 0)
    print(f"直方图算法准确率：{similarity:.3f}")
    return similarity

def ssim_similarity(imageA, imageB):
    start_time = time.time()
    image1_gray = color.rgb2gray(imageA)
    image2_gray = color.rgb2gray(imageB)
    # SSIM
    ssim_value, _ = ssim(image1_gray, image2_gray, full=True, data_range=image2_gray.max() - image2_gray.min())
    execution_time = time.time() - start_time
    print(f"执行时间: {execution_time:.3f} 秒")
    return ssim_value

def mse_similarity(imageA, imageB):
    start_time = time.time()

    # MSE
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    execution_time = time.time() - start_time
    print(f"执行时间: {execution_time:.3f} 秒")
    return err

def ncc_similarity(imageA, imageB):
    start_time = time.time()

    # Normalized Cross-Correlation (NCC)
    correlation = np.correlate(imageA.flatten(), imageB.flatten())
    execution_time = time.time() - start_time
    print(f"执行时间: {execution_time:.3f} 秒")
    return correlation[0] / (np.linalg.norm(imageA) * np.linalg.norm(imageB))

def histogram_similarity(imageA, imageB):
    start_time = time.time()
    # 直方图相似度
    histA = cv2.calcHist([imageA], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    histB = cv2.calcHist([imageB], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    similarity = cv2.compareHist(histA, histB, cv2.HISTCMP_CORREL)
    execution_time = time.time() - start_time
    print(f"执行时间: {execution_time:.3f} 秒")
    return similarity

def perceptual_hash_similarity(imageA, imageB):
    start_time = time.time()

    # 感知哈希
    hashA = imagehash.phash(Image.fromarray(cv2.cvtColor(imageA, cv2.COLOR_BGR2RGB)))
    hashB = imagehash.phash(Image.fromarray(cv2.cvtColor(imageB, cv2.COLOR_BGR2RGB)))
    similarity = 1 - (hashA - hashB) / len(hashA.hash)  # Normalized Hamming distance
    execution_time = time.time() - start_time
    print(f"执行时间: {execution_time:.3f} 秒")
    return similarity

# def local_sensitive_hash_similarity(imageA, imageB):
#     # 局部敏感哈希
#     # 这里使用了 10 个哈希表，每个哈希表使用 256 位的二进制投影
#     rbp = RandomBinaryProjections('rbp', 10)
#     engine = Engine(imageA.size, lshashes=[rbp])
#
#     hashA = engine.hash_vector(imageA.flatten())
#     hashB = engine.hash_vector(imageB.flatten())
#
#     similarity = np.mean(hashA == hashB)
#     return similarity

def cosine_similarity(imageA, imageB):
    start_time = time.time()

    # 余弦相似度
    vectorA = imageA.flatten()
    vectorB = imageB.flatten()
    similarity = 1 - cosine(vectorA, vectorB)
    execution_time = time.time() - start_time
    print(f"执行时间: {execution_time:.3f} 秒")
    return similarity

def local_binary_pattern_similarity(imageA, imageB):
    start_time = time.time()
    # 局部二进制模式
    lbpA = feature.local_binary_pattern(cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY), 8, 1, method="uniform")
    lbpB = feature.local_binary_pattern(cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY), 8, 1, method="uniform")
    similarity = np.sum(lbpA == lbpB) / float(lbpA.size)
    execution_time = time.time() - start_time
    print(f"执行时间: {execution_time:.3f} 秒")
    return similarity

source = 'data/img_sim_com/'
# 用法示例
imageA = cv2.imread("data/img_sim_com/img.png")
# imageA = cv2.imread("data/imgs_1_trim/frame_000.png")
# imageB = cv2.imread("data/imgs_1_trim/frame_000_cropped_2023-12-15-19-56-18.png")
imageB = cv2.imread("data/img_sim_com/img_3.png")

compare_img_hist(imageA, imageB)


# Measure similarity using each algorithm
ssim_result = ssim_similarity(imageA, imageB)
print(f"SSIM Similarity: {ssim_result:.4f}")

mse_result = mse_similarity(imageA, imageB)
print(f"MSE Similarity: {mse_result:.4f}")

ncc_result = ncc_similarity(imageA, imageB)
print(f"NCC Similarity: {ncc_result:.4f}")

hist_similarity = histogram_similarity(imageA, imageB)
print(f"Histogram Similarity: {hist_similarity:.4f}")

hash_similarity = perceptual_hash_similarity(imageA, imageB)
print(f"Perceptual Hash Similarity: {hash_similarity:.4f}")

# lsh_similarity = local_sensitive_hash_similarity(imageA, imageB)
# print(f"Local Sensitive Hash Similarity: {lsh_similarity:.4f}")

cosine_sim = cosine_similarity(imageA, imageB)
print(f"Cosine Similarity: {cosine_sim:.4f}")

lbp_similarity = local_binary_pattern_similarity(imageA, imageB)
print(f"Local Binary Pattern Similarity: {lbp_similarity:.4f}")


