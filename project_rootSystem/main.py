import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
import os

from project_rootSystem.project_rootSystem.models.Time.getFeatures import GetFeatures
from project_rootSystem.project_rootSystem.models.Time.segment import preprocess,denoise,db_bfs


def main():

    # 设置根系特征的数据结构（用字典存储）
    rootFeatures = {
        'Length': [],  # 主根长
        'TotalLength': [],  # 总根长
        'SurfArea': [],  # 表面积
        'Depth': [],  # 根深
        'Width': [],  # 根宽
        'WDRatio': [],  # 深宽比
        'Branches': [],  # 分支数
        'Density': [],  # 根密度
        'EnvelopeArea':[],# 凸包面积
    }

    folder_path = r'D:\PyCharm\Py_Projects\project_rootSystem\project_rootSystem\data\raw'
    for filename in os.listdir(folder_path):
        # 构造完整的文件路径
        file_path = os.path.join(folder_path, filename)
        # 使用 OpenCV 读取图片
        img = cv2.imread(file_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)# 转换为灰度图像

        # 对图片进行预处理
        preImage = preprocess(gray)

        # 二值分割
        _, binary_image = cv2.threshold(preImage, 40, 255, cv2.THRESH_BINARY)  # 通常设为90

        # 去噪
        kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)) #5,5
        eroded = cv2.erode(binary_image, kernel1, iterations=1)
        kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))#7,7
        dilated = cv2.dilate(eroded, kernel2, iterations=1)
        # eroded = cv2.dilate(dilated, kernel1, iterations=1)
        # dilated = cv2.dilate(eroded, kernel2, iterations=1)
        denoise_image = denoise(dilated)

        #可视化
        plt.figure(figsize=(20, 5))
        plt.subplot(1, 4, 1)
        plt.imshow(img, cmap='gray')
        plt.title('PreImage')
        plt.axis('off')
        plt.subplot(1, 4, 2)
        plt.imshow(binary_image, cmap='gray')
        plt.title('Denoise_Image')
        plt.axis('off')
        plt.subplot(1, 4, 3)
        plt.imshow(preImage, cmap='gray')
        plt.title('PreImage')
        plt.axis('off')
        plt.subplot(1, 4, 4)
        plt.imshow(denoise_image, cmap='gray')
        plt.title('Denoise_Image')
        plt.axis('off')
        plt.show()


        # 骨架化处理
        denoise_image[denoise_image == 255] = 1
        denoise_image = denoise_image.astype(np.uint8)
        skeleton = skeletonize(denoise_image)

        # 将图像转换为特征向量
        #提取图像坐标和矩阵
        img_arr = np.array(skeleton)
        img_arr = img_arr.astype(float)  # 图像的二值矩阵
        # 提取像素值为1的坐标
        coords = np.where(img_arr == 1)
        coords_normal = list(zip(coords[0], coords[1]))  # 像素值为1的坐标列表
        array_coords = np.array(coords_normal)  # 转换为矩阵

        # 语义分割(返回主根、侧根路径）
        all_paths = db_bfs(img, img_arr, array_coords, coords_normal,filename)

        # 获取根系特征
        GetFeatures(all_paths,rootFeatures,denoise_image)
        print(file_path)

    print(rootFeatures)

    # 将rootFeatures保存为表格
    df = pd.DataFrame(rootFeatures)
    #保存为CSV文件
    df.to_csv(".\\rootFeatures2.csv", index=False, header=False)


if __name__ == "__main__":

    main()