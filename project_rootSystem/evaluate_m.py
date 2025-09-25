import numpy as np
import os
from PIL import Image
import cv2
from sklearn.metrics import precision_score, recall_score, f1_score


# 计算每个类别的 IoU、Precision、Recall
def compute_metrics(gt, pred, num_classes):
    iou_list = []
    precision_list = []
    recall_list = []
    f1_list = []

    for cls in range(num_classes):
        # 计算 TP, FP, FN
        tp = np.sum((pred == cls) & (gt == cls))
        fp = np.sum((pred == cls) & (gt != cls))
        fn = np.sum((pred != cls) & (gt == cls))

        # 计算 IoU
        union = tp + fp + fn
        iou = tp / union if union > 0 else 0
        iou_list.append(iou)

        # 计算 Precision 和 Recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision_list.append(precision)
        recall_list.append(recall)

        # 计算 F1-Score
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0
        f1_list.append(f1)

    # 计算 mIoU, 平均 Precision, Recall 和 F1
    mIoU = np.mean(iou_list)
    avg_precision = np.mean(precision_list)
    avg_recall = np.mean(recall_list)
    avg_f1 = np.mean(f1_list)

    return {
        "IoU": iou_list,
        "mIoU": mIoU,
        "Precision": avg_precision,
        "Recall": avg_recall,
        "F1-Score": avg_f1
    }


# 读取图像文件
def load_image(file_path):
    img = Image.open(file_path)
    return np.array(img)

# 将预测图片由彩色图转换为索引图
def color_to_index(rgb_image, color_map):
    """
    将 RGB 彩色图映射回类别索引图。

    参数:
        rgb_image (np.ndarray): 输入的 RGB 图像，形状为 (H, W, 3)。
        color_map (dict): 颜色到类别索引的映射字典，格式为 {类别索引: [R, G, B]}。

    返回:
        label_image (np.ndarray): 输出的类别索引图，形状为 (H, W)。
    """
    # 初始化类别索引图
    height, width, _ = rgb_image.shape
    label_image = np.zeros((height, width), dtype=np.uint8)  # 默认值为 0（背景）

    # 遍历颜色映射字典
    for color, class_id in color_map.items():
        # 找到与当前颜色匹配的像素
        match_pixels = np.all(rgb_image == color, axis=-1)  # 形状为 (H, W)
        # 将匹配的像素设置为当前类别索引
        label_image[match_pixels] = class_id

    return label_image

# 主函数：计算整个数据集的评估指标
def evaluate_model(pred_folder, gt_folder, num_classes=3):
    all_iou = []
    all_precision = []
    all_recall = []
    all_f1 = []

    color_map = {
        (0, 0, 0): 0,  # 背景为黑色
        (0, 255, 0): 1,  # 主根 为绿色
        (0, 0, 255): 2  # 侧根 为红色
    }

    pred_files = sorted(os.listdir(pred_folder))
    gt_files = sorted(os.listdir(gt_folder))

    assert len(pred_files) == len(gt_files), "预测和标签图像数量不匹配"

    # 遍历每一张图片
    for pred_file, gt_file in zip(pred_files, gt_files):
        pre_fullpath=os.path.join(pred_folder, pred_file)
        pre_img = cv2.imread(pre_fullpath)
        pred_img = color_to_index(pre_img, color_map)  # 将颜色映射回类别索引
        gt_fullpath=os.path.join(gt_folder, gt_file)
        gt_img = cv2.imread(gt_fullpath, cv2.IMREAD_GRAYSCALE)
        # 确保预测图和标签图形状一致
        assert pred_img.shape == gt_img.shape, f"预测图和标签图形状不一致: {pred_file}, {gt_file}"

        # 计算每张图的评估指标
        metrics = compute_metrics(gt_img, pred_img, num_classes)

        # 保存每张图片的评估结果
        all_iou.append(metrics["IoU"])
        all_precision.append(metrics["Precision"])
        all_recall.append(metrics["Recall"])
        all_f1.append(metrics["F1-Score"])

    # 计算全数据集的平均值
    avg_iou = np.mean(all_iou, axis=0)
    avg_miou = np.mean(avg_iou)
    avg_precision = np.mean(all_precision)
    avg_recall = np.mean(all_recall)
    avg_f1 = np.mean(all_f1)

    return {
        "IoU": avg_iou,
        "mIoU": avg_miou,
        "Precision": avg_precision,
        "Recall": avg_recall,
        "F1-Score": avg_f1
    }


# 设置文件夹路径
pred_folder = r"C:\Users\xsy\Desktop\testImage\all_Image\diff_seg\GMM"  # 预测图像文件夹路径
gt_folder = r"C:\Users\xsy\Desktop\testImage\all_Image\Annotation\label"  # 标签图像文件夹路径

# 评估模型
metrics = evaluate_model(pred_folder, gt_folder, num_classes=3)

# 输出结果
print("IoU per class:", metrics["IoU"])
print("Mean IoU (mIoU):", metrics["mIoU"])
print("Average Precision:", metrics["Precision"])
print("Average Recall:", metrics["Recall"])
print("Average F1-Score:", metrics["F1-Score"])