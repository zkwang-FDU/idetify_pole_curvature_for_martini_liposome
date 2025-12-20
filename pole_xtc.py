import MDAnalysis as mda
import numpy as np
import pyvista as pv
import networkx as nx
from scipy.spatial import cKDTree
import open3d as o3d
import matplotlib.pyplot as plt
from tqdm import tqdm  # 进度条
import warnings

# 忽略一些不必要的警告
warnings.filterwarnings('ignore')


# =============================================================================
# 1. 核心算法：MLS 投影 (静默版)
# =============================================================================
def project_points_to_midplane(points, radius=20.0, iterations=10):
    current_points = points.copy()
    n_points = len(points)
    final_normals = np.zeros_like(points)

    # KDTree 复用优化: 如果移动不大，可以复用 tree，但为了准确性这里每轮重建
    for it in range(iterations):
        tree = cKDTree(current_points)
        neighbors_list = tree.query_ball_point(current_points, r=radius)
        new_positions = np.zeros_like(current_points)
        valid_mask = np.zeros(n_points, dtype=bool)
        current_normals = np.zeros_like(current_points)

        for i in range(n_points):
            idx = neighbors_list[i]
            if len(idx) < 5:
                new_positions[i] = current_points[i]
                current_normals[i] = [0, 0, 1]
                continue

            local_cloud = current_points[idx]
            centroid = np.mean(local_cloud, axis=0)
            centered = local_cloud - centroid
            cov = centered.T @ centered
            evals, evecs = np.linalg.eigh(cov)
            normal = evecs[:, 0]

            vec = current_points[i] - centroid
            dist = np.dot(vec, normal)
            new_positions[i] = current_points[i] - dist * normal
            current_normals[i] = normal
            valid_mask[i] = True

        shift = np.linalg.norm(new_positions[valid_mask] - current_points[valid_mask], axis=1).mean()
        current_points[valid_mask] = new_positions[valid_mask]
        final_normals[valid_mask] = current_normals[valid_mask]

        if shift < 0.05: break

    return current_points, final_normals


# =============================================================================
# 2. 核心算法：BPA 构网 (静默版)
# =============================================================================
def mesh_using_ball_pivoting(points, normals, bpa_radii):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)

    # 保持法向一致性
    pcd.orient_normals_consistent_tangent_plane(k=15)

    radii = o3d.utility.DoubleVector(bpa_radii)
    try:
        o3d_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, radii)
    except Exception:
        return None

    vertices = np.asarray(o3d_mesh.vertices)
    triangles = np.asarray(o3d_mesh.triangles)

    if len(triangles) == 0:
        return None

    faces = np.hstack([np.full((len(triangles), 1), 3), triangles]).flatten()
    pv_mesh = pv.PolyData(vertices, faces)

    # 清理并只保留最大连通面
    pv_mesh = pv_mesh.clean()
    pv_mesh = pv_mesh.connectivity(largest=True)

    return pv_mesh


# =============================================================================
# 3. 核心算法：原始边缘提取 (静默版)
# =============================================================================
def extract_raw_largest_boundary(surface_mesh):
    # 提取边缘
    edges = surface_mesh.extract_feature_edges(boundary_edges=True,
                                               feature_edges=False,
                                               manifold_edges=False)
    edges = edges.clean()

    if edges.n_lines == 0:
        return 0.0

    # 构建图
    lines = edges.lines
    points = edges.points
    G = nx.Graph()

    i = 0
    while i < len(lines):
        p1_idx = lines[i + 1]
        p2_idx = lines[i + 2]
        dist = np.linalg.norm(points[p1_idx] - points[p2_idx])
        G.add_edge(p1_idx, p2_idx, weight=dist)
        i += 3

    # 寻找最大连通分量
    comps = list(nx.connected_components(G))
    max_perimeter = 0.0

    for comp in comps:
        subgraph = G.subgraph(comp)
        length = subgraph.size(weight="weight")
        if length > max_perimeter:
            max_perimeter = length

    return max_perimeter


# =============================================================================
# 主程序：轨迹分析
# =============================================================================
def analyze_trajectory(gro_file, xtc_file, selection, mls_r, bpa_radii, stride=1, output_txt="perimeter_data.txt"):
    print(f"Loading topology: {gro_file}")
    print(f"Loading trajectory: {xtc_file}")

    u = mda.Universe(gro_file, xtc_file)
    atoms = u.select_atoms(selection)

    times = []
    perimeters = []

    print(f"Start processing {len(u.trajectory)} frames (Stride={stride})...")

    # 打开文件准备写入
    with open(output_txt, "w") as f:
        f.write("# Time(ps) Perimeter(Angstrom)\n")

        # 使用 tqdm 显示进度条
        for ts in tqdm(u.trajectory[::stride], unit="frame"):
            current_time = ts.time
            points = atoms.positions

            try:
                # 1. 坍缩
                collapsed, normals = project_points_to_midplane(points, radius=mls_r, iterations=8)  # 稍微减少迭代以加速

                # 2. 构网
                surf = mesh_using_ball_pivoting(collapsed, normals, bpa_radii)

                # 3. 测长
                if surf and surf.n_cells > 0:
                    L = extract_raw_largest_boundary(surf)
                else:
                    L = 0.0  # 构网失败视作无开口或闭合

            except Exception as e:
                # 如果某帧出错，不中断，记为 NaN
                L = np.nan

            # 记录数据
            times.append(current_time)
            perimeters.append(L)

            # 实时写入文件（防止程序中途崩溃数据丢失）
            f.write(f"{current_time:.2f} {L:.4f}\n")
            f.flush()

    return np.array(times), np.array(perimeters)


if __name__ == "__main__":
    # --- 文件配置 ---
    GRO_FILE = "0ns.gro"  # 拓扑文件
    XTC_FILE = "nojump.xtc"  # 轨迹文件 (请修改这里)

    # --- 参数配置 ---
    # 你的原子选择
    ATOM_SELECTION = "name C4B or name C4A or name D3B or name D3A"

    # 算法参数 (保持你觉得效果好的那一组)
    MLS_RADIUS = 25.0
    BPA_RADII = [15.0,17.0,20.0]

    # 每隔几帧分析一次？(1=每帧都算, 10=每10帧算一次)
    STRIDE = 10

    # --- 运行分析 ---
    times, values = analyze_trajectory(
        GRO_FILE,
        XTC_FILE,
        ATOM_SELECTION,
        MLS_RADIUS,
        BPA_RADII,
        stride=STRIDE,
        output_txt="liposome_opening.txt"
    )

    # --- 绘图 ---
    print("\nPlotting results...")
    plt.figure(figsize=(10, 6))

    # 绘制曲线
    plt.plot(times / 1000.0, values, color='b', linewidth=1.5, label='Perimeter')

    # 可选：绘制平滑平均线
    # window = 10
    # if len(values) > window:
    #     smooth = np.convolve(values, np.ones(window)/window, mode='valid')
    #     plt.plot(times[window-1:] / 1000.0, smooth, color='r', linewidth=2, label='Moving Avg')

    plt.title(f"Liposome Opening Perimeter vs Time\n(MLS_R={MLS_RADIUS}, BPA={BPA_RADII})")
    plt.xlabel("Time (ns)")
    plt.ylabel("Perimeter ($\AA$)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # 保存图片
    plt.savefig("liposome_opening_curve.png", dpi=300)
    plt.show()

    print("Done. Data saved to 'liposome_opening.txt' and graph to 'liposome_opening_curve.png'.")