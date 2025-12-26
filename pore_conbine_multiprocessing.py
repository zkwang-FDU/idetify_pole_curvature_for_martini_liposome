import MDAnalysis as mda
import numpy as np
import pyvista as pv
import networkx as nx
from scipy.spatial import cKDTree
import open3d as o3d
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing
import os
import datetime  # [新增] 用于时间记录


# =============================================================================
# 1. 核心算法函数
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

        diff = new_positions[valid_mask] - current_points[valid_mask]
        shift = np.linalg.norm(diff, axis=1).mean() if len(diff) > 0 else 0.0

        current_points[valid_mask] = new_positions[valid_mask]
        final_normals[valid_mask] = current_normals[valid_mask]

        if shift < 0.05: break

    return current_points, final_normals


def mesh_using_ball_pivoting(points, normals, bpa_radii):
    # Open3D几何构建
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

    # 修复 Deprecation Warning
    pv_mesh = pv_mesh.connectivity(extraction_mode='largest')

    return pv_mesh


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
# 2. Worker 函数 (多进程子任务)
# =============================================================================
def analyze_single_frame(data_package):
    time_val, points, mls_r, bpa_r = data_package
    try:
        collapsed, normals = project_points_to_midplane(points, mls_r)
        surf = mesh_using_ball_pivoting(collapsed, normals, bpa_r)

        L = 0.0
        med_mean = 0.0
        med_gauss = 0.0

        if surf is not None and surf.n_points > 0:
            L, _ = extract_raw_largest_boundary(surf)
            k_mean = surf.curvature(curv_type='mean')
            if len(k_mean) > 0: med_mean = np.median(k_mean)
            k_gauss = surf.curvature(curv_type='gaussian')
            if len(k_gauss) > 0: med_gauss = np.median(k_gauss)

        return (time_val, L, med_mean, med_gauss)

    except Exception as e:
        return (time_val, 0.0, 0.0, 0.0)


# =============================================================================
# 3. 主程序
# =============================================================================
if __name__ == "__main__":
    # --- [新增] Log 文件配置 ---
    LOG_FILE = "run_history.log"

    # 获取开始时间
    start_dt = datetime.datetime.now()
    start_str = start_dt.strftime('%Y-%m-%d %H:%M:%S')

    # 写入开始信息
    with open(LOG_FILE, "w") as f_log:
        f_log.write("========================================\n")
        f_log.write(f"Task Started : {start_str}\n")
        f_log.write("========================================\n")
        f_log.write(f"Parameters:\n")

    # --- 配置 ---
    GRO_FILE = "0ns.gro"
    XTC_FILE = "nojump.xtc"
    ATOM_SELECTION = "name C4B or name C4A or name D3B or name D3A"
    MLS_RADIUS = 25.0
    BPA_RADII = [15.0, 17.0, 20.0]
    STRIDE = 1
    WRITE_BATCH_SIZE = 100
    NUM_WORKERS = max(1, multiprocessing.cpu_count() - 2)

    OUT_PERIMETER = "data_perimeter_fast.txt"
    OUT_MEAN_CURV = "data_mean_curvature_fast.txt"
    OUT_GAUSS_CURV = "data_gaussian_curvature_fast.txt"

    # 将参数也写入 Log (可选，但推荐)
    with open(LOG_FILE, "a") as f_log:
        f_log.write(f"  Workers     : {NUM_WORKERS}\n")
        f_log.write(f"  Stride      : {STRIDE}\n")
        f_log.write(f"  MLS Radius  : {MLS_RADIUS}\n")
        f_log.write("----------------------------------------\n\n")

    print(f"Loading Universe... Using {NUM_WORKERS} workers.")
    u = mda.Universe(GRO_FILE, XTC_FILE)
    atoms = u.select_atoms(ATOM_SELECTION)

    # 1. 准备数据任务
    tasks = []
    print("Preparing tasks (loading trajectories into memory)...")
    for ts in u.trajectory[::STRIDE]:
        tasks.append((ts.time, atoms.positions.copy(), MLS_RADIUS, BPA_RADII))

    total_frames = len(tasks)
    print(f"Total frames to process: {total_frames}")

    # 2. 打开数据文件
    f_p = open(OUT_PERIMETER, "w")
    f_m = open(OUT_MEAN_CURV, "w")
    f_g = open(OUT_GAUSS_CURV, "w")

    f_p.write("# Time(ps) Perimeter(A)\n")
    f_m.write("# Time(ps) Median_Mean_Curvature(1/A)\n")
    f_g.write("# Time(ps) Median_Gaussian_Curvature(1/A^2)\n")

    # 3. 运行多进程处理
    batch_buffer = []
    plot_time = []
    plot_perimeter = []
    plot_mean = []
    plot_gauss = []

    print("Starting parallel processing...")

    with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
        iterator = pool.imap(analyze_single_frame, tasks)

        for i, result in tqdm(enumerate(iterator), total=total_frames):
            batch_buffer.append(result)

            if len(batch_buffer) >= WRITE_BATCH_SIZE or i == total_frames - 1:
                for item in batch_buffer:
                    t, p, m, g = item
                    f_p.write(f"{t:.4f} {p:.4f}\n")
                    f_m.write(f"{t:.4f} {m:.8f}\n")
                    f_g.write(f"{t:.4f} {g:.8f}\n")

                    plot_time.append(t)
                    plot_perimeter.append(p)
                    plot_mean.append(m)
                    plot_gauss.append(g)

                f_p.flush()
                f_m.flush()
                f_g.flush()
                batch_buffer = []

    f_p.close()
    f_m.close()
    f_g.close()

    print("Calculation finished. Generating plots...")

    # =====================================================================
    # 绘图
    # =====================================================================
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(plot_time, plot_perimeter, color='black', linewidth=1)
        plt.title("Pore Perimeter vs Time")
        plt.xlabel("Time (ps)")
        plt.ylabel("Perimeter ($\AA$)")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig("plot_perimeter_fast.png", dpi=300)
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(plot_time, plot_mean, color='blue', linewidth=1)
        plt.title("Median Mean Curvature vs Time")
        plt.xlabel("Time (ps)")
        plt.ylabel("Mean Curvature ($1/\AA$)")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig("plot_mean_curvature_fast.png", dpi=300)
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(plot_time, plot_gauss, color='red', linewidth=1)
        plt.title("Median Gaussian Curvature vs Time")
        plt.xlabel("Time (ps)")
        plt.ylabel("Gaussian Curvature ($1/\AA^2$)")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig("plot_gaussian_curvature_fast.png", dpi=300)
        plt.close()

        print("All done!")

    except Exception as e:
        print(f"Error during plotting: {e}")

    # --- [新增] 计算结束时间并写入 Log ---
    end_dt = datetime.datetime.now()
    end_str = end_dt.strftime('%Y-%m-%d %H:%M:%S')
    duration = end_dt - start_dt  # 计算时间差

    print(f"\n[Log] Processed {total_frames} frames in {duration}")

    with open(LOG_FILE, "a") as f_log:
        f_log.write(f"\nTask Finished: {end_str}\n")
        f_log.write(f"Total Duration: {duration}\n")
        f_log.write("========================================\n")