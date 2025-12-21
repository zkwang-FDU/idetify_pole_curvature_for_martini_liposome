import MDAnalysis as mda
import numpy as np
import pyvista as pv
import networkx as nx
from scipy.spatial import cKDTree
import open3d as o3d
import matplotlib.pyplot as plt
from tqdm import tqdm  # 引入进度条库
import os


# =============================================================================
# 1. 核心算法：MLS 投影 (保持原样)
# =============================================================================
def project_points_to_midplane(points, radius=20.0, iterations=10):
    current_points = points.copy()
    n_points = len(points)
    final_normals = np.zeros_like(points)

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
# 2. 核心算法：BPA 构网 (保持原样)
# =============================================================================
def mesh_using_ball_pivoting(points, normals, bpa_radii):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)

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
    pv_mesh = pv_mesh.connectivity(largest=True)

    return pv_mesh


# =============================================================================
# 3. 核心算法：原始边缘提取 (保持原样)
# =============================================================================
def extract_raw_largest_boundary(surface_mesh):
    edges = surface_mesh.extract_feature_edges(boundary_edges=True,
                                               feature_edges=False,
                                               manifold_edges=False)
    edges = edges.clean()

    if edges.n_lines == 0:
        return 0.0, None

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

    lines_flat = []
    for u, v in best_edges:
        lines_flat.extend([2, u, v])

    rim_mesh = pv.PolyData(points, lines=lines_flat)

    return max_perimeter, rim_mesh


# =============================================================================
# 4. 主程序：支持 tqdm 和 实时保存
# =============================================================================
if __name__ == "__main__":
    # --- 配置 ---
    GRO_FILE = "0ns.gro"
    XTC_FILE = "nojump.xtc"
    ATOM_SELECTION = "name C4B or name C4A or name D3B or name D3A"

    MLS_RADIUS = 25.0
    BPA_RADII = [15.0, 17.0, 20.0]

    OUT_PERIMETER = "data_perimeter.txt"
    OUT_MEAN_CURV = "data_mean_curvature.txt"
    OUT_GAUSS_CURV = "data_gaussian_curvature.txt"

    # 用于绘图的内存缓存
    plot_time = []
    plot_perimeter = []
    plot_mean = []
    plot_gauss = []

    try:
        print(f"Loading Universe: {GRO_FILE} | {XTC_FILE}")
        u = mda.Universe(GRO_FILE, XTC_FILE)
        atoms = u.select_atoms(ATOM_SELECTION)

        # 1. 打开文件句柄 (Write mode)
        f_p = open(OUT_PERIMETER, "w")
        f_m = open(OUT_MEAN_CURV, "w")
        f_g = open(OUT_GAUSS_CURV, "w")

        # 2. 写入表头
        f_p.write("# Time(ps) Perimeter(A)\n")
        f_m.write("# Time(ps) Median_Mean_Curvature(1/A)\n")
        f_g.write("# Time(ps) Median_Gaussian_Curvature(1/A^2)\n")

        print("Start processing trajectory...")

        # 3. 循环遍历 (使用 tqdm 显示进度条)
        # total=u.trajectory.n_frames 帮助 tqdm 预估剩余时间
        for ts in tqdm(u.trajectory, total=u.trajectory.n_frames, unit="frame"):
            current_time = ts.time
            points = atoms.positions

            # --- 算法执行 ---
            try:
                # A. MLS
                collapsed, normals = project_points_to_midplane(points, MLS_RADIUS)

                # B. Mesh
                surf = mesh_using_ball_pivoting(collapsed, normals, BPA_RADII)

                L = 0.0
                med_mean = 0.0
                med_gauss = 0.0

                if surf is not None and surf.n_points > 0:
                    # C. 边缘提取
                    L, _ = extract_raw_largest_boundary(surf)

                    # D. 曲率
                    k_mean = surf.curvature(curv_type='mean')
                    if len(k_mean) > 0:
                        med_mean = np.median(k_mean)

                    k_gauss = surf.curvature(curv_type='gaussian')
                    if len(k_gauss) > 0:
                        med_gauss = np.median(k_gauss)

            except Exception as e_inner:
                # 如果某一帧算法出错，记录为 0 并继续，不要中断整个程序
                # print(f"Frame {ts.frame} error: {e_inner}")
                L, med_mean, med_gauss = 0.0, 0.0, 0.0

            # --- 核心：实时写入文件并刷新 ---
            f_p.write(f"{current_time:.4f} {L:.4f}\n")
            f_m.write(f"{current_time:.4f} {med_mean:.8f}\n")
            f_g.write(f"{current_time:.4f} {med_gauss:.8f}\n")

            # Flush 确保数据立即写入硬盘，防止断电数据丢失
            f_p.flush()
            f_m.flush()
            f_g.flush()

            # --- 添加到列表以便最后画图 ---
            plot_time.append(current_time)
            plot_perimeter.append(L)
            plot_mean.append(med_mean)
            plot_gauss.append(med_gauss)

        # 4. 关闭文件
        f_p.close()
        f_m.close()
        f_g.close()

        print("\nCalculation finished. Data saved. Generating plots...")

        # =====================================================================
        # 绘图
        # =====================================================================
        # 图1: 周长
        plt.figure(figsize=(10, 6))
        plt.plot(plot_time, plot_perimeter, color='black', linewidth=1)
        plt.title("Pore Perimeter vs Time")
        plt.xlabel("Time (ps)")
        plt.ylabel("Perimeter ($\AA$)")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig("plot_perimeter.png", dpi=300)
        plt.close()

        # 图2: 平均曲率
        plt.figure(figsize=(10, 6))
        plt.plot(plot_time, plot_mean, color='blue', linewidth=1)
        plt.title("Median Mean Curvature vs Time")
        plt.xlabel("Time (ps)")
        plt.ylabel("Mean Curvature ($1/\AA$)")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig("plot_mean_curvature.png", dpi=300)
        plt.close()

        # 图3: 高斯曲率
        plt.figure(figsize=(10, 6))
        plt.plot(plot_time, plot_gauss, color='red', linewidth=1)
        plt.title("Median Gaussian Curvature vs Time")
        plt.xlabel("Time (ps)")
        plt.ylabel("Gaussian Curvature ($1/\AA^2$)")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig("plot_gaussian_curvature.png", dpi=300)
        plt.close()

        print("All done!")

    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        import traceback

        traceback.print_exc()
        # 即使报错退出，由于前面用了 flush，已经跑完的帧数据依然在txt里