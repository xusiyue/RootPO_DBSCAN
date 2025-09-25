import os
import cv2
import numpy as np
from collections import deque
import distinctipy

"""图片预处理"""
def preprocess(image):
    # 1.顶帽变化
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (27, 27))  # 定义结构元素，这里使用一个较大的核，以便捕捉根系的细节
    tophat1 = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)  # 应用顶帽变换
    tophat1 = cv2.normalize(tophat1, None, 0, 255, cv2.NORM_MINMAX)  # 调整对比度以更好地显示结果

    return tophat1

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

"""
    语义分割（分出主根和侧根）
    第一步：找出边缘点    
    第二步：利用BFS算法找出第一个边缘点到其他边缘点的路径
    第三步：遍历所有路径，找出最长的路径作为主根路径（主根路径颜色加粗）
    第四步：给所有路径渲染颜色
"""
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

    # 找出paths中缺少的coods坐标，组成新的分支路径
    def find_missing_coords(paths, coods):
        # 将所有paths中的坐标收集到一个集合中
        all_path_coords = set()
        for path in paths:
            all_path_coords.update(path)

        # 找出coods中不在paths中的坐标,作为一个分支
        missing_branches = [coord for coord in coods if tuple(coord) not in all_path_coords]
        return missing_branches

    # 生成差异度比较大的颜色
    def generate_distinct_colors_distinctipy(num_clusters):
        colors_rgb = distinctipy.get_colors(num_clusters)
        colors_255 = [(int(r * 255), int(g * 255), int(b * 255)) for r, g, b in colors_rgb]
        return np.array(colors_255, dtype=np.uint8)

    # 将主根路径的坐标以及它们左右两侧的坐标渲染为红色
    def render_main_root(path, image):
        color = (255, 0, 0)
        for coord in path:
            y, x = coord
            # 确保坐标在图像范围内
            if 0 <= x < width and 0 <= y < height:
                image[y, x] = color  # 红色
                if x > 5:
                    image[y, x - 1] = color  # 左侧坐标
                    image[y, x - 2] = color
                    image[y, x - 3] = color
                    image[y, x - 4] = color
                    image[y, x - 5] = color
                if x < width - 5:
                    image[y, x + 1] = color  # 右侧坐标
                    image[y, x + 2] = color
                    image[y, x + 3] = color
                    image[y, x + 4] = color
                    image[y, x + 5] = color

    # 将cleaned_paths里的所有序列坐标渲染成红色以外的随机颜色
    def render_Branches_paths(paths, image):
        num_clusters = len(paths)
        colors = generate_distinct_colors_distinctipy(num_clusters)
        p=0
        for path in paths:
            color=colors[p]
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
                        image[y - 1, x] = color
                        image[y - 2, x] = color
                        image[y - 3, x] = color
                    if x < width - 4:
                        image[y, x + 1] = color  # 右侧坐标
                        image[y, x + 2] = color
                        image[y, x + 3] = color
                        image[y, x + 4] = color
            p=p+1

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
    # paths保存所有根的路径坐标
    missing_coords = find_missing_coords(allBranchs_paths + [main_root], coods)
    allBranchs_paths.append(missing_coords)
    paths = allBranchs_paths
    paths.append(main_root)
    # print("allBranchs_paths:",allBranchs_paths) #最终的所有路径

    #第四步：渲染主根和侧根
    height, width = image.shape[:2]  # 获取原始图像的尺寸
    # 创建一个尺寸与原始图像相同且全黑的RGB图像,所有值都设置为0
    fenge_image = np.zeros((height, width, 3), dtype=np.uint8)
    # 渲染主根路径
    render_main_root(main_root, fenge_image)
    # 渲染清理后的路径
    render_Branches_paths(allBranchs_paths, fenge_image)


    #保存语义分割图
    save_path = r"C:\Users\xsy\Desktop\testImage\test3\seg"
    # 检查保存路径是否存在，如果不存在则创建
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # 构建完整的保存路径
    full_save_path = os.path.join(save_path, "seg_" + filename)
    # 保存图像
    fenge_image_rgb = cv2.cvtColor(fenge_image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(full_save_path, fenge_image_rgb)
    print(f"Image saved to {full_save_path}")

    # 返回所有路径
    allBranchs_paths.append(main_root)
    paths=allBranchs_paths

    return paths
