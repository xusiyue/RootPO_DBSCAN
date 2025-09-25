import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
data = pd.read_csv('.\\rootFeatures.csv')

# 定义特征列表
featureNames = ['Length', 'TotalLength', 'SurfArea', 'Depth', 'Width', 'WDRatio', 'Branches', 'Density']

# 特征名称列表
features = data.columns

p=0
# 遍历每个特征并绘制图表
for feature in features:
    plt.figure(figsize=(10, 5))  # 创建一个新的图形实例，并设置图形大小
    plt.plot(data.index, data[feature], marker='o', linestyle='-', color='b')  # 绘制折线图
    plt.title(f' {featureNames[p]}')  # 设置图表标题
    plt.xlabel('imgNumber')  # 设置x轴标签
    plt.ylabel('mm')  # 设置y轴标签
    plt.grid(True)  # 显示网格
    plt.show()  # 显示图表
    p=p+1