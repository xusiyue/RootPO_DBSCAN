import numpy as np
from scipy.spatial import ConvexHull
from Py_Projects.project_rootSystem.project_rootSystem.models.Time.calibrate import Calibrate
import matplotlib.pyplot as plt

"""
    获取根系表型特征
    1.主根长（mm）
    2.总根长(mm)
    3.根表面积(mm2）
    4.根深(mm)
    5.根宽(mm)
    6.根宽深比
    7.分支数（侧根数）
    8.根密度
    [根系连接分析（研究根系分支角度、连通性等形态特征）、
    根系拓扑分析（研究根系连接数量、路径长度）、
    根系伸展分析（记录根系整体等分布情况）]
"""
def GetFeatures(all_paths,rootFeatures,denoise_image):
    #获取像素对应的mm
    pixel_size_mm=float(Calibrate())

    # 1.主根长
    numOfRoot = len(all_paths)
    mainPath = all_paths[numOfRoot - 1]
    mainLength= len(mainPath) * pixel_size_mm
    rootFeatures['Length'].append(format(mainLength, '.3g'))

    # 2.总根长=每条根的长度相加
    length = 0
    for path in all_paths:
        length = length + len(path)
    totalLength=length * pixel_size_mm
    rootFeatures['TotalLength'].append(format(totalLength, '.3g'))

    # 3.根的表面积=根的所有像素点总数
    img_arr = np.array(denoise_image)
    img_arr = img_arr.astype(float)  # 图像的二值矩阵
    coords =np.where(img_arr == 1) # 提取像素值为1的坐标
    coords_normal = list(zip(coords[0], coords[1]))  # 像素值为1的坐标列表
    surfArea = len(coords_normal) * pixel_size_mm * pixel_size_mm
    rootFeatures['SurfArea'].append(format(surfArea, '.3g'))

    # 4.根深=竖直方向两端像素点的距离（取根的纵坐标最大值与最小值之差）
    y_coords = [x for x, y in coords_normal]  # 提取所有y坐标
    y_diff = (max(y_coords) - min(y_coords))* pixel_size_mm # 计算y坐标的最大值和最小值之差,即为根深
    rootFeatures['Depth'].append(format(y_diff, '.3g'))

    # 5.根宽=水平方向两端像素点的距离（取根的横坐标最大值与最小值之差）
    x_coords = [y for x, y in coords_normal]  # 提取所有x坐标
    x_diff = (max(x_coords) - min(x_coords)) * pixel_size_mm # 计算x坐标的最大值和最小值之差
    rootFeatures['Width'].append(format(x_diff, '.3g'))

    # 6.根宽深比
    rootFeatures['WDRatio'].append(format(x_diff/y_diff, '.3g'))

    # 7.分支数
    rootFeatures['Branches'].append(numOfRoot-1)

    # 8.根密度：单位土壤体积中根的总长度
    # soil_coords = np.where(img_arr == 0)  # 提取像素值为0的坐标(图像代表的土壤体积）
    soil_coords = np.column_stack(np.where(img_arr == 0))
    rootFeatures['Density'].append(format((length)/float(len(soil_coords)),'.4g'))

    #9.凸包面积
    coords1 = np.column_stack(np.where(img_arr == 1))

    # def polygon_area(points):
    #     """计算多边形的面积"""
    #     n = len(points)
    #     area = 0.0
    #     for i in range(n):
    #         j = (i + 1) % n
    #         area += points[i][0] * points[j][1]
    #         area -= points[i][1] * points[j][0]
    #     area = abs(area) / 2.0
    #     return area
    #
    # envelopeArea=polygon_area(coords1) * pixel_size_mm

    hull = ConvexHull(coords1)
    # 计算凸包的面积
    envelopeArea = (hull.area) * pixel_size_mm  #

    rootFeatures['EnvelopeArea'].append(format(envelopeArea, '.3g'))

    # 绘制原始点集
    plt.plot(coords1[:, 0], coords1[:, 1], 'o')

    # 绘制凸包的顶点
    for simplex in hull.simplices:
        plt.plot(coords1[simplex, 0], coords1[simplex, 1], 'k-')

    # 标记凸包的顶点
    plt.plot(coords1[:, 0], coords1[:, 1], 'ko')

    # 设置图表标题和标签
    plt.title("Convex Hull")
    plt.xlabel("X")
    plt.ylabel("Y")

    # 显示图表
    plt.show()

    print("SurfArea:",len(coords_normal))
    print("EnvelopeArea:",hull.area)

    return True