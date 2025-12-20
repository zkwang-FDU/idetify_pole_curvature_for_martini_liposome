import MDAnalysis as mda
from MDAnalysis.analysis.leaflet import LeafletFinder
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')

print("\n简化测试:")


# 创建人工双层膜数据测试LeafletFinder
def test_artificial_bilayer():
    """创建人工双层膜测试LeafletFinder是否工作"""
    import MDAnalysis as mda
    from MDAnalysis.analysis.leaflet import LeafletFinder
    import numpy as np

    # 创建上下两层点
    n_points = 100
    # 上层
    upper = np.random.rand(n_points, 3) * 10
    upper[:, 2] = 5.0 + np.random.randn(n_points) * 0.2
    # 下层
    lower = np.random.rand(n_points, 3) * 10
    lower[:, 2] = 0.0 + np.random.randn(n_points) * 0.2

    positions = np.vstack([upper, lower])

    # 创建Universe
    from MDAnalysis.coordinates.memory import MemoryReader
    u = mda.Universe.empty(n_points * 2)
    u.load_new(positions, format=MemoryReader)

    # 测试
    atoms = u.atoms
    lf = LeafletFinder(u, atoms, cutoff=2.0, pbc=False)

    print(f"人工膜测试: 找到 {len(lf)} 个叶")
    if len(lf) == 2:
        print("✓ LeafletFinder工作正常")
        return True
    else:
        print("✗ LeafletFinder可能有问题")
        return False


test_artificial_bilayer()

# 1. 加载GRO文件
u = mda.Universe("0ns.gro")

# 2. 选择磷脂的磷酸基团
# 根据你的体系选择适当的原子
# 常见的选择方式：
lipids = u.select_atoms("name PO4")  # 磷酸基团的P原子

if len(lipids) == 0:
    print("错误：没有选择到磷脂原子！请检查选择语句。")
    print("可用的原子类型:", set(u.atoms.names))
    print("可用的残基名:", set(u.atoms.resnames))
    exit()

# 3. 使用LeafletFinder识别上下叶
try:
    cutoff = 15
    leaflets = LeafletFinder(u, lipids, cutoff=cutoff, pbc=False)

    print(f"\n找到 {len(leaflets)} 个叶")

    if len(leaflets) >= 2:
        upper_leaflet = leaflets.group(0)  # 上叶
        lower_leaflet = leaflets.group(1)  # 下叶

        print(f"上叶原子数: {len(upper_leaflet)}")
        print(f"下叶原子数: {len(lower_leaflet)}")
    else:
        print("警告：只找到了一个叶，可能选择参数需要调整")
        exit()

except Exception as e:
    print(f"LeafletFinder错误: {e}")
    print("尝试不同的原子选择或调整cutoff值")
    exit()

# 4. 创建3D图形
fig = plt.figure(figsize=(15, 5))

# 子图1: 3D散点图
ax1 = fig.add_subplot(131, projection='3d')

# 绘制上叶（红色）
ax1.scatter(upper_leaflet.positions[:, 0],
            upper_leaflet.positions[:, 1],
            upper_leaflet.positions[:, 2],
            c='red', s=30, alpha=0.7, label='Upper Leaflet', depthshade=True)

# 绘制下叶（蓝色）
ax1.scatter(lower_leaflet.positions[:, 0],
            lower_leaflet.positions[:, 1],
            lower_leaflet.positions[:, 2],
            c='blue', s=30, alpha=0.7, label='Lower Leaflet', depthshade=True)

ax1.set_xlabel('X (nm)', fontsize=10)
ax1.set_ylabel('Y (nm)', fontsize=10)
ax1.set_zlabel('Z (nm)', fontsize=10)
ax1.set_title('3D Scatter Plot\nRed: Upper, Blue: Lower', fontsize=12)
ax1.legend()
ax1.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 6. 输出统计信息
print("\n" + "="*50)
print("膜叶结构统计信息")
print("="*50)
print(f"体系尺寸 (nm): {u.dimensions[:3]}")
print(f"上叶Z坐标范围: [{upper_leaflet.positions[:, 2].min():.3f}, {upper_leaflet.positions[:, 2].max():.3f}] nm")
print(f"下叶Z坐标范围: [{lower_leaflet.positions[:, 2].min():.3f}, {lower_leaflet.positions[:, 2].max():.3f}] nm")
print(f"膜厚度(质心距离): {abs(upper_center[2] - lower_center[2]):.3f} nm")
print(f"上叶质心: ({upper_center[0]:.3f}, {upper_center[1]:.3f}, {upper_center[2]:.3f}) nm")
print(f"下叶质心: ({lower_center[0]:.3f}, {lower_center[1]:.3f}, {lower_center[2]:.3f}) nm")
print(f"磷脂分子密度(上叶): {len(upper_leaflet)/(u.dimensions[0]*u.dimensions[1]):.3f} molecules/nm²")
print(f"磷脂分子密度(下叶): {len(lower_leaflet)/(u.dimensions[0]*u.dimensions[1]):.3f} molecules/nm²")

# 7. 可选：如果LeafletFinder无法正确识别，尝试不同的cutoff值
def find_optimal_cutoff():
    """尝试不同的cutoff值找到最佳分叶"""
    print("\n尝试不同的cutoff值...")
    cutoffs = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5]

    for cutoff in cutoffs:
        try:
            leaflets = LeafletFinder(u, lipids, cutoff=cutoff, pbc=True)
            if len(leaflets) == 2:
                print(f"cutoff = {cutoff} Å: 成功找到2个叶 "
                      f"(上叶: {len(leaflets.group(0))}, 下叶: {len(leaflets.group(1))})")
            else:
                print(f"cutoff = {cutoff} Å: 找到{len(leaflets)}个叶")
        except:
            print(f"cutoff = {cutoff} Å: 失败")

# 如果需要测试不同的cutoff值，取消下面的注释
# find_optimal_cutoff()
