import MDAnalysis as mda
from MDAnalysis.analysis.leaflet import LeafletFinder
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 加载体系和轨迹
u = mda.Universe('0ns.gro')

# 对当前帧（如第0帧）进行分析
# 设定一个尝试性的截止距离，例如 1.0 nm
leaflets = LeafletFinder(u,'name PO4', cutoff=15) # 注意：单位是埃（Å）

# 获取两个叶层
upper_leaflet = leaflets.group(0)  # 第一个大簇
lower_leaflet = leaflets.group(1)  # 第二个大簇

print(f"上叶层脂质数量：{upper_leaflet.n_residues}")
print(f"下叶层脂质数量：{lower_leaflet.n_residues}")

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# 获取坐标数据
upper_pos = upper_leaflet.positions  # 上叶层原子坐标
lower_pos = lower_leaflet.positions  # 下叶层原子坐标

# 绘制上叶层（蓝色）
if len(upper_pos) > 0:
    ax.scatter(upper_pos[:, 0], upper_pos[:, 1], upper_pos[:, 2],
               c='blue', s=20, alpha=0.7, label=f'Upper Leaflet ({len(upper_pos)} atoms)', depthshade=True)

# 绘制下叶层（红色）
if len(lower_pos) > 0:
    ax.scatter(lower_pos[:, 0], lower_pos[:, 1], lower_pos[:, 2],
               c='red', s=20, alpha=0.7, label=f'Lower Leaflet ({len(lower_pos)} atoms)', depthshade=True)

# 设置坐标轴标签
ax.set_xlabel('X (Å)', fontsize=12, labelpad=10)
ax.set_ylabel('Y (Å)', fontsize=12, labelpad=10)
ax.set_zlabel('Z (Å)', fontsize=12, labelpad=10)

# 设置标题
ax.set_title('Lipid Bilayer Leaflets 3D Visualization\n(PO4 Atoms)', fontsize=14, pad=20)

# 添加图例
ax.legend(fontsize=10, loc='upper right')

# 设置视角，以便更好地观察双层结构
# 不同的视角有助于观察膜的双层特性
ax.view_init(elev=20, azim=45)  # elev=仰角, azim=方位角

# 添加网格
ax.grid(True, alpha=0.3)

# 添加颜色条说明
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='blue', alpha=0.7, label='Upper Leaflet'),
    Patch(facecolor='red', alpha=0.7, label='Lower Leaflet')
]
ax.legend(handles=legend_elements, loc='upper left')

# 显示图形
plt.tight_layout()
plt.show()
