import MDAnalysis as mda
import numpy as np
import pyvista as pv
import networkx as nx
from scipy.spatial import cKDTree
import open3d as o3d
import matplotlib.pyplot as plt
import os


# =============================================================================
# 1. 核心算法：MLS 投影 (逻辑保持原样，仅移除 Print)
# =============================================================================
def project_points_to_midplane(points, radius=20.0, iterations=10):
    current_points = points.copy()
    n_points = len(points)
    final_normals = np.zeros_like(points)

    # print(f"\n[1/4] MLS 投影坍缩 (N={n_points}, R={radius})...")

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

        # print(f"      Iter {it+1}: 平均位移 = {shift:.4f} A")
        if shift < 0.05: break

    return current_points, final_normals


# =============================================================================
# 2. 核心算法：BPA 构网 (逻辑保持原样，仅移除 Print)
# =============================================================================
def mesh_using_ball_pivoting(points, normals, bpa_radii):
    # print(f"\n[2/4] BPA 滚球法构网...")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)

    # 保持法向一致性
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

    # 移除离散的小碎片，只保留最大的那个曲面
    pv_mesh = pv_mesh.connectivity(largest=True)

    return pv_mesh


# =============================================================================
# 3. 核心算法：原始边缘提取 (逻辑保持原样，仅移除 Print)
# =============================================================================
def extract_raw_largest_boundary(surface_mesh):
    # print(f"\n[3/4] 提取原始最大边缘...")

    # 1. 提取所有边缘
    edges = surface_mesh.extract_feature_edges(boundary_edges=True,
                                               feature_edges=False,
                                               manifold_edges=False)
    edges = edges.clean()

    if edges.n_lines == 0:
        return 0.0, None

    # 2. 构建图
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

    # 3. 分离连通分量
    comps = list(nx.connected_components(G))

    max_perimeter = 0.0
    best_edges = []

    for comp in comps:
        subgraph = G.subgraph(comp)
        length = subgraph.size(weight="weight")

        if length > max_perimeter:
            max_perimeter = length
            best_edges = list(subgraph.edges())

    if not best_edges:
        return 0.0, None

    # 4. 重建原始 PolyData (为了保持返回值一致，虽然纯计算不需要Mesh对象，但保持算法结构)
    lines_flat = []
    for u, v in best_edges:
        lines_flat.extend([2, u, v])

    rim_mesh = pv.PolyData(points, lines=lines_flat)

    return max_perimeter, rim_mesh


# =============================================================================
# 4. 批量处理与绘图主程序
# =============================================================================
if __name__ == "__main__":
    # --- 配置 ---
    GRO_FILE = "ref.gro"  # 拓扑文件
    XTC_FILE = "traj.xtc"  # 轨迹文件
    ATOM_SELECTION = "name C4B or name C4A or name D3B or name D3A"

    # 算法参数
    MLS_RADIUS = 25.0
    BPA_RADII = [15.0, 17.0, 20.0]

    # 输出文件名
    OUT_PERIMETER = "data_perimeter.txt"
    OUT_MEAN_CURV = "data_mean_curvature.txt"
    OUT_GAUSS_CURV = "data_gaussian_curvature.txt"

    # 数据容器
    time_list = []
    perimeter_list = []
    mean_curv_median_list = []
    gauss_curv_median_list = []

    print(f"开始处理轨迹: {XTC_FILE} ...")

    try:
        # 加载 Universe
        u = mda.Universe(GRO_FILE, XTC_FILE)
        atoms = u.select_atoms(ATOM_SELECTION)

        total_frames = len(u.trajectory)

        for ts in u.trajectory:
            current_time = ts.time
            frame_idx = ts.frame

            # 简单的进度提示
            if frame_idx % 10 == 0:
                print(f"Processing Frame {frame_idx}/{total_frames} (Time: {current_time:.1f} ps)...")

            points = atoms.positions

            # --- 步骤 1: MLS ---
            collapsed, normals = project_points_to_midplane(points, MLS_RADIUS)

            # --- 步骤 2: BPA ---
            surf = mesh_using_ball_pivoting(collapsed, normals, BPA_RADII)

            # 初始化当前帧结果 (默认 0 或 NaN)
            L = 0.0
            med_mean = 0.0
            med_gauss = 0.0

            if surf is not None and surf.n_points > 0:
                # --- 步骤 3: 边缘提取 ---
                L, _ = extract_raw_largest_boundary(surf)

                # --- 步骤 4: 曲率计算 ---
                # 平均曲率
                k_mean = surf.curvature(curv_type='mean')
                med_mean = np.median(k_mean) if len(k_mean) > 0 else 0.0

                # 高斯曲率
                k_gauss = surf.curvature(curv_type='gaussian')
                med_gauss = np.median(k_gauss) if len(k_gauss) > 0 else 0.0

            # 记录数据
            time_list.append(current_time)
            perimeter_list.append(L)
            mean_curv_median_list.append(med_mean)
            gauss_curv_median_list.append(med_gauss)

        # =====================================================================
        # 保存数据到文件
        # =====================================================================
        data_arr = np.array([time_list, perimeter_list, mean_curv_median_list, gauss_curv_median_list]).T

        # 分别保存
        np.savetxt(OUT_PERIMETER, np.column_stack((time_list, perimeter_list)),
                   header="Time(ps) Perimeter(A)", fmt="%.4f")

        np.savetxt(OUT_MEAN_CURV, np.column_stack((time_list, mean_curv_median_list)),
                   header="Time(ps) Median_Mean_Curvature(1/A)", fmt="%.8f")

        np.savetxt(OUT_GAUSS_CURV, np.column_stack((time_list, gauss_curv_median_list)),
                   header="Time(ps) Median_Gaussian_Curvature(1/A^2)", fmt="%.8f")

        print("\n计算完成。数据已保存。正在绘图...")

        # =====================================================================
        # 绘图 (Matplotlib)
        # =====================================================================
        # 1. 周长图
        plt.figure(figsize=(10, 6))
        plt.plot(time_list, perimeter_list, color='black', linewidth=1.5)
        plt.title("Pore Perimeter vs Time")
        plt.xlabel("Time (ps)")
        plt.ylabel("Perimeter ($\AA$)")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig("plot_perimeter.png", dpi=300)
        plt.close()

        # 2. 平均曲率中位数图
        plt.figure(figsize=(10, 6))
        plt.plot(time_list, mean_curv_median_list, color='blue', linewidth=1.5)
        plt.title("Median Mean Curvature vs Time")
        plt.xlabel("Time (ps)")
        plt.ylabel("Mean Curvature ($1/\AA$)")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig("plot_mean_curvature.png", dpi=300)
        plt.close()

        # 3. 高斯曲率中位数图
        plt.figure(figsize=(10, 6))
        plt.plot(time_list, gauss_curv_median_list, color='red', linewidth=1.5)
        plt.title("Median Gaussian Curvature vs Time")
        plt.xlabel("Time (ps)")
        plt.ylabel("Gaussian Curvature ($1/\AA^2$)")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig("plot_gaussian_curvature.png", dpi=300)
        plt.close()

        print("绘图完成: plot_perimeter.png, plot_mean_curvature.png, plot_gaussian_curvature.png")

    except Exception as e:
        print(f"发生错误: {e}")
        import traceback

        traceback.print_exc()