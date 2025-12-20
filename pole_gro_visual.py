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

        if surf:
            # 4. Rim Extraction (Raw & Robust)
            L, rim = extract_raw_largest_boundary(surf)

            # --- 可视化 ---
            pl = pv.Plotter()
            pl.set_background('white')

            # 绘制曲面
            pl.add_mesh(surf, color="cyan", opacity=0.4, show_edges=False, label="Surface")

            # 绘制边缘
            if rim:
                # render_lines_as_tubes=True 只是为了让线变粗好被看见，不会改变数据的原始性
                pl.add_mesh(rim, color="red", line_width=6, render_lines_as_tubes=True, label=f"Max Rim (L={L:.1f})")

            pl.add_legend()
            pl.show()

    except Exception as e:
        print(f"Err: {e}")
        import traceback
        traceback.print_exc()