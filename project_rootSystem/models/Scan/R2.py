import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np

# 读取两个Excel文件
df1 = pd.read_excel(r"C:\Users\xsy\Desktop\30RootData_RootSeg.xlsx", sheet_name="Sheet1")  # 手动测量数据
df2 = pd.read_excel(r"C:\Users\xsy\Desktop\30RootData_WinRIZO.xlsx", sheet_name="Sheet1")  # 半自动测量数据

# 确保按图片序号对齐
df_merged = pd.merge(df1, df2, on="图片序号", suffixes=('_RootSeg', '_WinRIZO'))

# 示例：分析 Branches 列
x = df_merged["Branches_RootSeg"]  # 全自动方法
y = df_merged["Branches_WinRIZO"]  # 半自动方法

# 计算Pearson相关系数和p值
pearson_r, pearson_p = stats.pearsonr(x, y)
# 计算R²
r_squared = pearson_r ** 2

plt.figure(figsize=(8, 6), dpi=300)  # 高清分辨率
sns.set_style("whitegrid")  # 白色网格背景
sns.set_context("paper", font_scale=1.2)  # 论文字体大小

# 散点图
scatter = sns.scatterplot(
    x=x, y=y,
    s=80, alpha=0.7, edgecolor='k', linewidth=0.5,
    color='#1f77b4'  # 自定义颜色
)

# 线性拟合线 + 95%置信区间
sns.regplot(
    x=x, y=y,
    scatter=False, ci=95,
    line_kws={'color': 'red', 'lw': 2},
    truncate=False
)

# 添加1:1参考线（虚线）
max_val = max(np.max(x), np.max(y))
plt.plot([0, max_val], [0, max_val], 'k--', lw=1, alpha=0.5, label="1:1 Line")

# 标注统计信息
text = f"$R^2$ = {r_squared:.2f}\n$p$ = {pearson_p:.3f}"
plt.text(0.05, 0.85, text, transform=plt.gca().transAxes,
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

# 坐标轴标签和标题
plt.xlabel("Branches (Manual Measurement)", fontsize=10)
plt.ylabel("Branches (Semi-automated WinRIZO)", fontsize=10)
plt.title("Correlation Analysis of Root Branches", fontsize=12, pad=20)

# 调整边距和图例
plt.tight_layout()
plt.legend(loc='upper left')
plt.show()
plt.savefig(r"C:\Users\xsy\Desktop\Correlation_Branches.png", dpi=300, bbox_inches='tight')