import cv2
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
import numpy as np
import pandas as pd
from collections import deque
import os
from Py_Projects.project_rootSystem.project_rootSystem.models.Scan.getFeatures import GetFeatures

"""
    第一步：找出边缘点    
    第二步：利用BFS算法找出第一个边缘点到其他边缘点的路径
    第三步：截出所有路径中重复的坐标，作为主根路径（主根路径颜色加粗）
    第四步：给所有路径渲染颜色
"""
def mayiti(image,img_arr,A,coods): #img_arr为图像的二值矩阵，A为图像中像素为1的坐标二维矩阵

    # 记录每个点周围八个方向是否有邻域点
    def bafang(key):
        list_round = []  # 存储一个点周围八个方向，有哪些点
        for i in range(A.shape[0]):
            if np.linalg.norm(A[key, :] - A[i, :]) <= np.sqrt(2):  # 计算两空间点的距离：dist = np.linalg.norm(pt2 - pt1)
                if key != i:
                    list_round.append(i)
                else:
                    pass
            else:
                pass
        return list_round

    # BFS求路径
    def bfs_path(matrix, start, end):
        # 定义八个方向的移动
        directions = [
            (-1, 0), (1, 0), (0, -1), (0, 1),  # 上下左右
            (-1, -1), (-1, 1), (1, -1), (1, 1)  # 四个对角线
        ]
        rows, cols = len(matrix), len(matrix[0])
        queue = deque([start])  # 队列,存放遍历的点
        visited = set([start])  # 集合，标记访问过的的点
        prev = {start: None}  # 前驱节点字典

        while queue:
            current = queue.popleft()
            # 检查是否到达终点
            if current == end:
                break
            # 探索八个方向
            for d in directions:
                new_row, new_col = current[0] + d[0], current[1] + d[1]
                if 0 <= new_row < rows and 0 <= new_col < cols and \
                        (new_row, new_col) not in visited and matrix[new_row][new_col] == 1:
                    visited.add((new_row, new_col))
                    prev[(new_row, new_col)] = current
                    queue.append((new_row, new_col))
        # 重建路径
        path = []
        if (end in prev):
            while current:
                path.append(current)
                current = prev[current]
            path.reverse()
        return path

    # 筛选出主根路径
    def mainRoot(all_paths):
        max_length = 0
        main_root_list = []

        # 遍历all_paths，找出最长的序列
        for path in all_paths:
            if len(path) > max_length:
                max_length = len(path)
                main_root_list = path

        # 打印结果
        print("最长的序列即主根是：", main_root_list)
        return main_root_list

    # 将主根路径的坐标以及它们左右两侧的坐标渲染为红色
    def render_main_root(path, image):
        for coord in path:
            y, x = coord
            # 确保坐标在图像范围内
            if 0 <= x < width and 0 <= y < height:
                image[y, x] = (255, 0, 0)  # 红色
                if x > 0: image[y, x - 1] = (255, 0, 0)  # 左侧坐标
                if x < width - 1: image[y, x + 1] = (255, 0, 0)  # 右侧坐标

    # 将cleaned_paths里的所有序列坐标渲染成红色以外的随机颜色
    def render_Branches_paths(paths, image):
        # color = (0, 53, 255)  # 蓝色
        color = (255, 255, 255)  # 白色
        for path in paths:
            for coord in path:
                y, x = coord
                # 确保坐标在图像范围内
                if 0 <= x < width and 0 <= y < height:
                    image[y, x] = color


    #第一步：找出所有边缘点
    list_1=[]
    for i in range(A.shape[0]):
        if len(bafang(i))==1: #找到所有边缘点，分类数>=边缘点个数
            list_1.append(i)
        else:pass

    #第二步：寻找第一个边缘点到其他边缘点的路径
    all_paths = []
    start = coods[list_1[0]]
    for i in list_1:
        if i!=0:
            end=coods[i]
            path = bfs_path(img_arr, start, end)#使用bfs算法寻找
            all_paths.append(path)


    #第三步：分出主根路径与侧根路径
    main_root=mainRoot(all_paths) #主根路径
    # 创建一个集合，包含主根序列中的所有坐标
    main_root_set = set()
    for coord in main_root:
        main_root_set.add(coord)
    allBranchs_paths=[]
    for path in all_paths:
        allBranchs_path = [coord for coord in path if coord not in main_root_set]
        if allBranchs_path:  # 只有当清理后的路径不为空时才添加
            allBranchs_paths.append(allBranchs_path)
    # print("allBranchs_paths:",allBranchs_paths) #最终的所有路径

    #第四步：渲染主根和侧根
    height, width = image.shape[:2]  # 获取原始图像的尺寸
    # 创建一个尺寸与原始图像相同且全黑的RGB图像,所有值都设置为0
    fenge_image = np.zeros((height, width, 3), dtype=np.uint8)
    # 渲染主根路径
    render_main_root(main_root, fenge_image)
    # 渲染清理后的路径
    render_Branches_paths(allBranchs_paths, fenge_image)

    #第五步：可视化原图和语义分割的结果
    plt.figure(figsize=(10, 5))
    # 原图
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Oraginal Image')
    plt.axis('off')
    # 语义分割的结果
    plt.subplot(1, 2, 2)
    plt.imshow(fenge_image, cmap='gray')
    plt.title('Semantic Segmentation Image')
    plt.axis('off')

    plt.show()

    return True


def db_bfs(image,img_arr,A,coods,filename): #img_arr为图像的二值矩阵，A为图像中像素为1的坐标二维矩阵

    # 记录每个点周围八个方向是否有邻域点
    def bafang(key):
        list_round = []  # 存储一个点周围八个方向，有哪些点
        for i in range(A.shape[0]):
            if np.linalg.norm(A[key, :] - A[i, :]) <= np.sqrt(2):  # 计算两空间点的距离：dist = np.linalg.norm(pt2 - pt1)
                if key != i:
                    list_round.append(i)
                else:
                    pass
            else:
                pass
        return list_round

    # BFS求路径
    def bfs_path(matrix, start, end):
        # 定义八个方向的移动
        directions = [
            (-1, 0), (1, 0), (0, -1), (0, 1),  # 上下左右
            (-1, -1), (-1, 1), (1, -1), (1, 1)  # 四个对角线
        ]
        rows, cols = len(matrix), len(matrix[0])
        queue = deque([start])  # 队列,存放遍历的点
        visited = set([start])  # 集合，标记访问过的的点
        prev = {start: None}  # 前驱节点字典

        while queue:
            current = queue.popleft()
            # 检查是否到达终点
            if current == end:
                break
            # 探索八个方向
            for d in directions:
                new_row, new_col = current[0] + d[0], current[1] + d[1]
                if 0 <= new_row < rows and 0 <= new_col < cols and \
                        (new_row, new_col) not in visited and matrix[new_row][new_col] == 1:
                    visited.add((new_row, new_col))
                    prev[(new_row, new_col)] = current
                    queue.append((new_row, new_col))
        # 重建路径
        path = []
        if (end in prev):
            while current:
                path.append(current)
                current = prev[current]
            path.reverse()
        return path

    # 筛选出主根路径
    def mainRoot(all_paths):
        max_length = 0
        main_root_list = []

        # 遍历all_paths，找出最长的序列
        for path in all_paths:
            if len(path) > max_length:
                max_length = len(path)
                main_root_list = path

        # 打印结果
        # print("最长的序列即主根是：", main_root_list)
        return main_root_list

    # 将主根路径的坐标以及它们左右两侧的坐标渲染为绿色
    def render_main_root(path, image):
        color = (0, 255, 0)  # RGB绿色
        for coord in path:
            y, x = coord
            # 确保坐标在图像范围内
            if 0 <= x < width and 0 <= y < height:
                image[y, x] = color
                if x > 4:
                    image[y, x - 1] = color  # 左侧坐标
                    image[y, x - 2] = color
                    image[y, x - 3] = color
                    image[y, x - 4] = color
                    # image[y, x - 5] = color

                if x < width - 1:
                    image[y, x + 1] = color  # 右侧坐标
                    # image[y, x + 2] = color


    # # 将cleaned_paths里的所有序列坐标渲染成红色
    # def render_Branches_paths(paths, image):
    #     color = (255, 0, 0)  # 红色
    #     p=0
    #     for path in paths:
    #         for coord in path:
    #             y, x = coord
    #             # 确保坐标在图像范围内
    #             if 0 <= x < width and 0 <= y < height:
    #                 image[y, x] = color
    #                 if x > 3 and y >4:
    #                     image[y, x - 1] = color  # 左侧坐标
    #                     image[y, x - 2] = color
    #                     image[y, x - 3] = color
    #                     image[y - 1, x] = color
    #                     image[y - 2, x] = color
    #                     image[y - 3, x] = color
    #                     image[y - 4, x] = color
    #
    #                 if x < width - 2 and y < height - 3:
    #                     image[y, x + 1] = color  # 右侧坐标
    #         p=p+1

    # 将cleaned_paths里的所有序列坐标渲染成红色以外的随机颜色
    def render_Branches_paths(paths, image):
        num_clusters = len(paths)
        colors = np.random.randint(0, 255, size=(num_clusters, 3), dtype=np.uint8)
        # color = (255, 255, 255)  # 白色
        p = 0
        for path in paths:
            color = colors[p]
            for coord in path:
                y, x = coord
                # 确保坐标在图像范围内
                if 0 <= x < width and 0 <= y < height:
                    image[y, x] = color
                    if x > 3 and y > 4:
                        image[y, x - 1] = color  # 左侧坐标
                        image[y, x - 2] = color
                        image[y, x - 3] = color
                        image[y - 1, x] = color
                        image[y - 2, x] = color
                        image[y - 3, x] = color
                        image[y - 4, x] = color

                    if x < width - 2:
                        image[y, x + 1] = color  # 右侧坐标
                        image[y, x + 2] = color  # 右侧坐标
            p = p + 1


    #第一步：找出所有边缘点
    list_1=[]
    for i in range(A.shape[0]):
        if len(bafang(i))==1: #找到所有边缘点，分类数>=边缘点个数
            list_1.append(i)
        else:pass

    #第二步：寻找第一个边缘点到其他边缘点的路径
    all_paths = []
    start = coods[list_1[0]]
    for i in list_1:
        if i!=0:
            end=coods[i]
            path = bfs_path(img_arr, start, end)#使用bfs算法寻找
            all_paths.append(path)


    #第三步：分出主根路径与侧根路径
    main_root=mainRoot(all_paths) #主根路径
    # 创建一个集合，包含主根序列中的所有坐标
    main_root_set = set()
    for coord in main_root:
        main_root_set.add(coord)
    allBranchs_paths=[]
    for path in all_paths:
        allBranchs_path = [coord for coord in path if coord not in main_root_set]
        if len(allBranchs_path)>15:  # 只有当清理后的路径不为空时才添加
            allBranchs_paths.append(allBranchs_path)
    # print("allBranchs_paths:",allBranchs_paths) #最终的所有路径

    #第四步：渲染主根和侧根
    height, width = image.shape[:2]  # 获取原始图像的尺寸
    # 创建一个尺寸与原始图像相同且全黑的RGB图像,所有值都设置为0
    fenge_image = np.zeros((height, width, 3), dtype=np.uint8)
    # 渲染清理后的路径
    render_Branches_paths(allBranchs_paths, fenge_image)
    # 渲染主根路径
    render_main_root(main_root, fenge_image)

    # #第五步：可视化原图和语义分割的结果
    # plt.figure(figsize=(10, 5))
    # # 原图
    # plt.subplot(1, 2, 1)
    # plt.imshow(image, cmap='gray')
    # plt.title('Oraginal Image')
    # plt.axis('off')
    # # 语义分割的结果
    # plt.subplot(1, 2, 2)
    # plt.imshow(fenge_image, cmap='gray')
    # plt.title('Semantic Segmentation Image')
    # plt.axis('off')
    # plt.show()

    #保存语义分割图
    # save_path = r"C:\Users\xsy\Desktop\testImage\all_Image\diff_seg\root_seg"
    save_path=r"C:\Users\xsy\Desktop\testImage\test\seg"
    # 检查保存路径是否存在，如果不存在则创建
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # 构建完整的保存路径
    full_save_path = os.path.join(save_path, "seg_" + filename)
    # 保存图像
    fenge_image_rgb = cv2.cvtColor(fenge_image, cv2.COLOR_BGR2RGB) #转换格式
    # print(fenge_image[87, 847])  #[255   0   0]
    # print(fenge_image_rgb[87, 847]) #[  0   0 255]
    cv2.imwrite(full_save_path, fenge_image_rgb)#显示成红色，因为保存时要以BGR格式读入
    print(f"Image saved to {full_save_path}")
    # 返回所有路径
    allBranchs_paths.append(main_root)
    paths=allBranchs_paths

    return paths

"""图片预处理"""
def preprocess(image):
    # 1.顶帽变化
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))  # 定义结构元素，这里使用一个较大的核，以便捕捉根系的细节
    tophat1 = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)  # 应用顶帽变换
    tophat1= cv2.normalize(tophat1, None, 0, 255, cv2.NORM_MINMAX)   # 调整对比度以更好地显示结果

    # 2.调整伽马值
    gamma = 2.0# 定义伽马值
    gamma_table = [np.power(i / 255.0, gamma) * 255.0 for i in np.arange(256)] # 创建查找表
    gamma_table = np.array(gamma_table).astype("uint8")
    corrected_image = cv2.LUT(tophat1, gamma_table) # 应用伽马校正

    # 3.调整对比度和亮度
    alpha = 3.5  # 增加对比度，如果是小于1的值则减少对比度
    beta = 55 # 减少亮度，如果是负值则减少亮度
    adjusted_image = cv2.convertScaleAbs(corrected_image, alpha=alpha, beta=beta)

    # 可视化阈值分割和骨架化结果
    plt.figure(figsize=(20, 5))
    # 原图
    plt.subplot(1, 4, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Oraginal image')
    plt.axis('off')
    # 顶帽变换结果
    plt.subplot(1, 4, 2)
    plt.imshow(tophat1, cmap='gray')
    plt.title('tophat image')
    plt.axis('off')
    # gamma调整结果
    plt.subplot(1, 4, 3)
    plt.imshow(corrected_image, cmap='gray')
    plt.title('Gamma image')
    plt.axis('off')
    # 亮度和对比度调整结果
    plt.subplot(1, 4, 4)
    plt.imshow(adjusted_image, cmap='gray')
    plt.title('adjusted image')
    plt.axis('off')

    plt.show()

    # cv2.imshow('Oraginal', image)
    # cv2.imshow('tophat1', tophat1)
    # cv2.imshow('corrected_image', corrected_image)
    # cv2.imshow('adjusted_image', adjusted_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return adjusted_image

"""去除图片噪声"""
def denoise(image):
    # 找到所有的连通分量及其统计信息
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image, 8, cv2.CV_32S)

    # 找到最大连通分量的索引
    max_area = 0
    max_area_index = -1
    for i in range(1, num_labels):  # 从1开始，因为0是背景
        if stats[i, cv2.CC_STAT_AREA] > max_area:
            max_area = stats[i, cv2.CC_STAT_AREA]
            max_area_index = i

    # 创建一个全黑的图像
    result_image = np.zeros_like(image)
    # 将最大连通分量的区域填充为白色
    result_image[labels == max_area_index] = 255
    # 将其他连通分量的区域变为黑色
    result_image[labels != max_area_index] = 0

    return  result_image

#进行语义分割的预测
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
        'Density': []  # 根密度
        # 'EnvelopeArea': [],  # 凸包面积
    }

    # input_path = r"C:\Users\xsy\Desktop\testImage\all_Image\sample"
    input_path = r"C:\Users\xsy\Desktop\testImage\test\sample"
    files = sorted(os.listdir(input_path))
    print(files)

    for filename in files:
        full_ori_path=os.path.join(input_path,  filename)
        # 读取图像
        img=cv2.imread(full_ori_path)
        # 转换为灰度图像
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 二值分割
        _, binary_image = cv2.threshold(gray,70, 255, cv2.THRESH_BINARY) #通常设为95
        denoise_image1=denoise(binary_image)

        kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        eroded = cv2.erode(binary_image, kernel1, iterations=1)
        kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        dilated = cv2.dilate(eroded, kernel2, iterations=1)
        eroded = cv2.dilate(dilated, kernel1, iterations=1)
        dilated = cv2.dilate(eroded, kernel2, iterations=1)
        denoise_image = denoise(dilated)

        # 骨架化处理
        denoise_image[denoise_image==255] = 1
        denoise_image = denoise_image.astype(np.uint8)
        skeleton = skeletonize(denoise_image)

        # 将图像转换为特征向量
        img_arr = np.array(skeleton)
        img_arr = img_arr.astype(float) #图像的二值矩阵
        # 提取像素值为1的坐标
        coords = np.where(img_arr == 1)
        # 坐标为(y,x)形式
        coords = list(zip(coords[0], coords[1])) #像素值为1的坐标列表
        # print(coords)
        array_coords = np.array(coords)#转换为矩阵

        # 语义分割
        filename2=filename.replace('.jpg','.png')
        all_paths = db_bfs(img, img_arr, array_coords,coords,filename2)

        # 获取根系特征
        # GetFeatures(all_paths, rootFeatures, denoise_image1)
    # df = pd.DataFrame(rootFeatures)
    # # 保存为CSV文件
    # df.to_csv(".\\rootFeatures_Scan2.csv", index=False, header=False)

if __name__ == '__main__':
    main()
