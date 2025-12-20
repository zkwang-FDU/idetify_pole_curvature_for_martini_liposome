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

        print(f"      Iter {it+1}: 平均位移 = {shift:.4f} A")
        if shift < 0.05: break

    return current_points, final_normals

# =============================================================================
# 2. 核心算法：BPA 构网
# =============================================================================
def mesh_using_ball_pivoting(points, normals, bpa_radii):
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

    return pv_mesh

# =============================================================================
# 3. 核心算法：原始边缘提取 (无平滑，仅过滤噪点)
# =============================================================================
def extract_raw_largest_boundary(surface_mesh):
    """
    只返回最长的那个连通边缘，不进行任何平滑或插值。
    """
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

        # ... (接上文代码)

        if surf:
            # 4. 边缘提取
            L, rim = extract_raw_largest_boundary(surf)
            # =========================================================================
            # 5. [双重修正版] 计算并可视化双曲率
            # =========================================================================
            print(f"\n[4/4] 计算表面曲率 (自动动态范围修正)...")

            # 1. 计算原始数据
            mean_vals = surf.curvature(curv_type='mean')
            gauss_vals = surf.curvature(curv_type='gaussian')

            # 必须赋值给 point_data 才能绘图
            surf.point_data["Mean_Curvature"] = mean_vals
            surf.point_data["Gaussian_Curvature"] = gauss_vals


            # -------------------------------------------------------------------------
            # 辅助函数：计算合适的 clim 范围 (排除 5% 的极值噪点)
            # -------------------------------------------------------------------------
            def get_robust_clim(data_array):
                # 移除 NaN
                data = data_array[~np.isnan(data_array)]
                # 获取 5% 和 95% 分位点
                p5 = np.percentile(data, 5)
                p95 = np.percentile(data, 95)

                # 找到绝对值较大的那个作为边界
                limit = max(abs(p5), abs(p95))

                # 防止完全平面的情况 (limit=0)
                if limit < 1e-9:
                    limit = 0.001

                return limit, p5, p95


            # -------------------------------------------------------------------------
            # 2. 统计诊断
            # -------------------------------------------------------------------------
            m_lim, m_p5, m_p95 = get_robust_clim(mean_vals)
            g_lim, g_p5, g_p95 = get_robust_clim(gauss_vals)

            print(f"   [诊断 - 平均曲率] P5: {m_p5:.5f} | P95: {m_p95:.5f} -> 设定范围: +/- {m_lim:.5f}")
            print(f"   [诊断 - 高斯曲率] P5: {g_p5:.6f} | P95: {g_p95:.6f} -> 设定范围: +/- {g_lim:.6f}")
            print("   (注意：高斯曲率通常比平均曲率小 1-2 个数量级，这是正常的)")

            # =========================================================================
            # 3. 可视化 (左右对比)
            # =========================================================================
            pl = pv.Plotter(shape=(1, 2))

            # --- 左图：平均曲率 (弯曲程度) ---
            pl.subplot(0, 0)
            pl.add_text(f"Mean Curvature\nRange: +/- {m_lim:.4f}", font_size=10)
            # coolwarm: 红色=凸, 蓝色=凹, 白色=平
            pl.add_mesh(surf, scalars="Mean_Curvature", cmap="coolwarm",
                        clim=[-m_lim, m_lim], show_edges=False)
            if rim:
                pl.add_mesh(rim, color="black", line_width=3, render_lines_as_tubes=True)

            # --- 右图：高斯曲率 (几何形状) ---
            pl.subplot(0, 1)
            pl.add_text(f"Gaussian Curvature\nRange: +/- {g_lim:.5f}", font_size=10)
            # RdBu_r (Red-Blue-Reversed):
            # 红色 (>0) = 碗形/球形 (Elliptic)
            # 蓝色 (<0) = 马鞍形 (Hyperbolic)
            # 白色 (=0) = 平面/圆柱 (Parabolic)
            pl.add_mesh(surf, scalars="Gaussian_Curvature", cmap="RdBu_r",
                        clim=[-g_lim, g_lim], show_edges=False)
            if rim:
                pl.add_mesh(rim, color="black", line_width=3, render_lines_as_tubes=True)

            pl.link_views()
            pl.show()
    except Exception as e:
        print(f"Err: {e}")
        import traceback
        traceback.print_exc()