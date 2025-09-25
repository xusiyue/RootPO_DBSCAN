import cv2
import numpy as np
import matplotlib.pyplot as plt


"""
    相机标定，得到每像素代表的实际距离（毫米）
"""
def Calibrate():
    # 棋盘格的实际尺寸
    square_size_mm = 27  # 每个正方形的宽度（毫米）
    board_width = 8  # 棋盘格宽度（正方形数量）
    board_height = 5  # 棋盘格高度（正方形数量）

    # 准备对象点（世界坐标系中的点）
    objp = np.zeros((board_width * board_height, 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_width, 0:board_height].T.reshape(-1, 2)
    objp *= square_size_mm / 1000.0  # 转换为米

    # 读取图片
    img = cv2.imread('D:\PyCharm\Py_Projects\project_rootSystem\project_rootSystem\data\calibrateImage\Pic_1.bmp')#时序图每像素代表的实际距离（毫米）: 0.094315mm
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 调整对比度和亮度
    alpha = 3.5  # 增加对比度，如果是小于1的值则减少对比度
    beta = 55 # 减少亮度，如果是负值则减少亮度
    adjusted_image = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)

    # 寻找棋盘格的候选角点
    ret, corners = cv2.findChessboardCorners(adjusted_image, (board_width, board_height), None)

    # 如果找到角点，计算像素与世界距离的转换比例
    if ret:
        #精确定位角点
        cv2.cornerSubPix(adjusted_image, corners, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

        # 假设我们关注的是水平方向上的距离
        # 计算像素尺寸
        distances = [] # 初始化一个列表来存储距离

        # 遍历每一行的角点
        for p in range (board_height-1):
            for i in range (p * board_width, (p+1) * (board_width - 1), 1):
                # 计算欧氏距离
                distance = np.linalg.norm(corners[i+1] - corners[i])
                # 将距离添加到列表中
                distances.append(distance)
        pixel_size_mm = square_size_mm / np.mean(distances)
        # pixel_size_mm1 = square_size_mm / np.linalg.norm(corners[0]-corners[board_width])

        # 打印转换比例
        print(f"每像素代表的实际距离（毫米）: {format(pixel_size_mm, '.5g')}mm")

        # # 绘制并显示角点
        # img = cv2.drawChessboardCorners(img, (board_width, board_height), corners, ret)
        # plt.figure(figsize=(8, 6))
        # plt.subplot(1, 1, 1)
        # plt.imshow(img, cmap='gray')
        # plt.title('Chessboard Corners')
        # plt.axis('off')
        # plt.show()

    return format(pixel_size_mm, '.5g')

