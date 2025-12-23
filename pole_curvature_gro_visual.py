import MDAnalysis as mda
import numpy as np
import pyvista as pv
import networkx as nx
from scipy.spatial import cKDTree
import open3d as o3d
import time

# =============================================================================
# 1. 核心算法：MLS 投影 (带法线)
# =============================================================================
def project_points_to_midplane(points, radius=20.0, iterations=10):
    current_points = points.copy()
    n_points = len(points)
    final_normals = np.zeros_like(points)

    print(f"\n[1/4] MLS 投影坍缩 (N={n_points}, R={radius})...")

    for it in range(iterations):
        start = time.perf_counter()
        tree = cKDTree(current_points)
        neighbors_list = tree.query_ball_point(current_points, r=radius)
        new_positions = np.zeros_like(current_points)
        valid_mask = np.zeros(n_points, dtype=bool)
        current_normals = np.zeros_like(current_points)
        elapsed1 = time.perf_counter() - start
        print(f"耗时: {elapsed1:.4f}秒")
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
        elapsed1 = time.perf_counter() - start
        print(f"{it}耗时: {elapsed1:.4f}秒")

        shift = np.linalg.norm(new_positions[valid_mask] - current_points[valid_mask], axis=1).mean()
        current_points[valid_mask] = new_positions[valid_mask]
        final_normals[valid_mask] = current_normals[valid_mask]
        elapsed1 = time.perf_counter() - start
        print(f"迭代{it}耗时: {elapsed1:.4f}秒")
        print(f"      Iter {it+1}: 平均位移 = {shift:.4f} A")
        if shift < 0.05: break

    return current_points, final_normals

# =============================================================================
# 2. 核心算法：BPA 构网
# =============================================================================
def mesh_using_ball_pivoting(points, normals, bpa_radii):
    start = time.perf_counter()
    print(f"\n[2/4] BPA 滚球法构网...")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)

    # 保持法向一致性 (重要)
    pcd.orient_normals_consistent_tangent_plane(k=15)

    radii = o3d.utility.DoubleVector(bpa_radii)
    o3d_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, radii)

    vertices = np.asarray(o3d_mesh.vertices)
    triangles = np.asarray(o3d_mesh.triangles)

    if len(triangles) == 0:
        return None

    faces = np.hstack([np.full((len(triangles), 1), 3), triangles]).flatten()
    pv_mesh = pv.PolyData(vertices, faces)
    pv_mesh = pv_mesh.clean()

    # 这一步非常重要：移除离散的小碎片，只保留最大的那个曲面
    pv_mesh = pv_mesh.connectivity(largest=True)
    elapsed1 = time.perf_counter() - start
    print(f"耗时: {elapsed1:.4f}秒")
    return pv_mesh

# =============================================================================
# 3. 核心算法：原始边缘提取 (无平滑，仅过滤噪点)
# =============================================================================
def extract_raw_largest_boundary(surface_mesh):
    """
    只返回最长的那个连通边缘，不进行任何平滑或插值。
    """
    start = time.perf_counter()
    print(f"\n[3/4] 提取原始最大边缘...")

    # 1. 提取所有边缘
    edges = surface_mesh.extract_feature_edges(boundary_edges=True,
                                               feature_edges=False,
                                               manifold_edges=False)

    # 清理一下，防止有点没线的情况
    edges = edges.clean()

    if edges.n_lines == 0:
        print("      未检测到边缘。")
        return 0.0, None

    # 2. 构建图
    lines = edges.lines
    points = edges.points # 这里的 points 索引对应 lines 里的 id
    G = nx.Graph()

    # PyVista lines: [2, id1, id2, 2, id3, id4 ...]
    i = 0
    while i < len(lines):
        p1_idx = lines[i+1]
        p2_idx = lines[i+2]
        # 计算原始物理距离
        dist = np.linalg.norm(points[p1_idx] - points[p2_idx])
        G.add_edge(p1_idx, p2_idx, weight=dist)
        i += 3

    # 3. 分离连通分量 (区分大环和小噪点)
    comps = list(nx.connected_components(G))

    max_perimeter = 0.0
    best_edges = []

    print(f"      检测到 {len(comps)} 个边缘回路。正在剔除噪点...")

    for comp in comps:
        subgraph = G.subgraph(comp)
        # 这一步计算的是最真实的物理周长（所有线段长度之和）
        length = subgraph.size(weight="weight")

        if length > max_perimeter:
            max_perimeter = length
            best_edges = list(subgraph.edges())

    print(f"      锁定最大边缘，原始周长: {max_perimeter:.2f} Angstrom")

    if not best_edges:
        return 0.0, None

    # 4. 重建原始 PolyData (不排序，不平滑，原样显示)
    # 格式: [2, p1, p2, 2, p3, p4 ...]
    lines_flat = []
    for u, v in best_edges:
        lines_flat.extend([2, u, v])

    # 注意：这里必须使用 edges.points，因为 best_edges 里的索引是基于它的
    rim_mesh = pv.PolyData(points, lines=lines_flat)
    elapsed1 = time.perf_counter() - start
    print(f"耗时: {elapsed1:.4f}秒")
    return max_perimeter, rim_mesh

# =============================================================================
# 主程序
# =============================================================================
if __name__ == "__main__":
    # --- 配置 ---
    GRO_FILE = "60ns.gro"
    ATOM_SELECTION = "name C4B or name C4A or name D3B or name D3A"
    MLS_RADIUS = 25.0
    BPA_RADII = [15.0,17.0,20.0]

    try:
        # 1. Load
        u = mda.Universe(GRO_FILE)
        atoms = u.select_atoms(ATOM_SELECTION)
        points = atoms.positions
        print(f"加载原子数: {len(points)}")

        # 2. MLS
        collapsed, normals = project_points_to_midplane(points, MLS_RADIUS)

        # 3. Mesh (BPA)
        surf = mesh_using_ball_pivoting(collapsed, normals, BPA_RADII)

        if surf:
            # 4. 边缘提取
            L, rim = extract_raw_largest_boundary(surf)

            # =========================================================================
            # 5. [修正版] 计算表面曲率 & 输出核心统计指标
            # =========================================================================
            print(f"\n[4/4] 计算表面曲率...")

            # 1. 计算平均曲率 (Mean Curvature)
            mean_curv_values = surf.curvature(curv_type='mean')
            surf.point_data["Mean_Curvature"] = mean_curv_values

            # 2. 计算高斯曲率 (Gaussian Curvature)
            gauss_curv_values = surf.curvature(curv_type='gaussian')
            surf.point_data["Gaussian_Curvature"] = gauss_curv_values

            # -------------------------------------------------------------------------
            # >>> 新增：直接输出中位数统计 <<<
            # -------------------------------------------------------------------------
            median_mean = np.median(mean_curv_values)
            median_gauss = np.median(gauss_curv_values)

            print("\n" + "=" * 50)
            print("【核心统计结果 / Key Statistics】")
            print(f"  > 原始平均曲率中位数 (Median Mean Curvature):   {median_mean:.8f} (1/A)")
            print(f"  > 原始高斯曲率中位数 (Median Gaussian Curvature): {median_gauss:.8f} (1/A^2)")
            print("=" * 50 + "\n")
            # -------------------------------------------------------------------------

            # --- 诊断步骤：看看真实的数值分布 (用于确定可视化范围) ---
            # 忽略极端的边缘噪点，看中间90%的数据长什么样
            p5 = np.percentile(mean_curv_values, 5)
            p95 = np.percentile(mean_curv_values, 95)

            # --- 可视化 ---
            pl = pv.Plotter()

            # 关键技巧：设置 clim 为分位点范围，或者手动设一个很小的数
            # 这样超过范围的强曲率（边缘）会饱和显示，而中间的微弱曲率会显现出来
            bound_limit = max(abs(p5), abs(p95))

            pl.add_text(f"Visualizing Range: +/- {bound_limit:.4f}\nMedian Mean: {median_mean:.4f}", font_size=12)

            pl.add_mesh(surf,
                        scalars="Mean_Curvature",
                        cmap="coolwarm",  # 蓝-白-红 配色
                        clim=[-bound_limit, bound_limit],
                        show_edges=False,
                        label="Surface (Mean Curv)")

            if rim:
                pl.add_mesh(rim, color="black", line_width=4, render_lines_as_tubes=True)

            pl.add_scalar_bar(title="Mean Curvature (1/A)")
            pl.show()

    except Exception as e:
        print(f"Err: {e}")
        import traceback
        traceback.print_exc()