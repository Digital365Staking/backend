# =========================================================
# RGB-D SLAM + NeRF Hybrid Pipeline
# Fixes weak-texture issues (walls, ceilings, floors)
# =========================================================
# Architecture:
#   1. SLAM odometry + loop closure → optimized poses
#   2. TSDF fusion → geometry skeleton
#   3. Instant-NGP NeRF trained on same poses → fills
#      weak-texture regions via view synthesis
#   4. Weak-texture detection → identify where TSDF failed
#   5. NeRF point cloud extracted at weak regions
#   6. Merge TSDF + NeRF → final cleaned mesh
# =========================================================
# Runtime: T4 GPU | RAM: High
# !pip uninstall -y jax jaxlib cupy-cuda12x shap rasterio xarray-einstats tobler pytensor
# !pip install open3d opencv-python-headless scipy numpy==1.26.4
# =========================================================

from google.colab import drive
drive.mount('/content/drive')

import os, json, glob, time, shutil, subprocess, sys
import numpy as np
import cv2
import open3d as o3d
from scipy.spatial.transform import Rotation as R

# ---- PATHS -----------------------------------------------
DATASET    = "/content/drive/MyDrive/3D"
COLOR_DIR  = f"{DATASET}/cam0/data"
DEPTH_DIR  = f"{DATASET}/depth0/data"
CALIB_FILE = f"{DATASET}/calib/camera_imu.json"
OUTPUT_DIR = f"{DATASET}/tsdf_nerf_hybrid"

NERF_DIR        = "/content/instant-ngp"
NERF_SCENE_DIR  = "/content/nerf_scene"
POSES_CKPT      = os.path.join(OUTPUT_DIR, "poses_checkpoint.json")

os.makedirs(OUTPUT_DIR,   exist_ok=True)
os.makedirs(NERF_SCENE_DIR, exist_ok=True)
os.makedirs(f"{NERF_SCENE_DIR}/images", exist_ok=True)

# ---- CONFIG ----------------------------------------------
DEPTH_SCALE  = 1000.0
DEPTH_TRUNC  = 3.5
VOXEL_SIZE   = 0.02

# Weak-texture detection thresholds
LAPLACIAN_THRESHOLD = 80.0   # frames below this variance = weak texture
TSDF_DENSITY_THRESH = 0.05   # vertices with low point-cloud support = weak region

# Loop closure
KEYFRAME_STEP       = 20
LOOP_FITNESS_THRESH = 0.50
MAX_LOOP_PAIRS      = 2000

# NeRF
NERF_STEPS          = 5000   # training steps — raise to 10000 for better quality
USE_EVERY_N_NERF    = 3      # subsample frames for NeRF training
# ----------------------------------------------------------


# =========================================================
# SECTION 1 — Intrinsics & File Matching
# =========================================================
with open(CALIB_FILE) as f:
    calib = json.load(f)

fx, fy, cx, cy = calib["cam0"]["intrinsics"]
W, H           = calib["cam0"]["resolution"]
intrinsic      = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)
print(f"Intrinsics: fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}  {W}x{H}")

color_files_raw = sorted(glob.glob(f"{COLOR_DIR}/*.png"))
if not color_files_raw:
    color_files_raw = sorted(glob.glob(f"{COLOR_DIR}/*.jpg"))
depth_files_raw = sorted(glob.glob(f"{DEPTH_DIR}/*depth.png"))

def ts_key(fname, suffix=''):
    stem = os.path.splitext(os.path.basename(fname))[0]
    return stem.replace(suffix, '') if suffix else stem

color_ts  = {ts_key(f): f           for f in color_files_raw}
depth_ts  = {ts_key(f, '_depth'): f for f in depth_files_raw}
common_ts = sorted(set(color_ts.keys()) & set(depth_ts.keys()))
color_files = [color_ts[t] for t in common_ts]
depth_files = [depth_ts[t] for t in common_ts]
N = len(common_ts)
print(f"Matched pairs: {N}")
if N == 0:
    raise RuntimeError("No matched pairs.")


# =========================================================
# SECTION 2 — RGBD Loader
# =========================================================
def load_rgbd(i, preprocess=True):
    color_bgr = cv2.imread(color_files[i])
    if color_bgr is None:
        return None
    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    depth_raw = cv2.imread(depth_files[i], cv2.IMREAD_UNCHANGED)
    if depth_raw is None:
        return None
    if preprocess:
        kernel    = np.ones((5, 5), np.uint8)
        depth_raw = cv2.morphologyEx(depth_raw, cv2.MORPH_CLOSE, kernel)
        depth_f32 = depth_raw.astype(np.float32)
        depth_f32 = cv2.bilateralFilter(depth_f32, d=5, sigmaColor=10, sigmaSpace=10)
        depth_f32[depth_f32 > DEPTH_TRUNC * DEPTH_SCALE] = 0
    else:
        depth_f32 = depth_raw.astype(np.float32)
    return o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(color_rgb),
        o3d.geometry.Image(depth_f32),
        depth_scale=DEPTH_SCALE,
        depth_trunc=DEPTH_TRUNC,
        convert_rgb_to_intensity=False
    )

def rgbd_to_pcd(rgbd):
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
    pcd = pcd.voxel_down_sample(VOXEL_SIZE * 2)
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=VOXEL_SIZE * 4, max_nn=30)
    )
    return pcd


# =========================================================
# SECTION 3 — Weak-Texture Detection
# =========================================================
def measure_texture_variance(frame_idx):
    """
    Compute Laplacian variance of the color frame.
    Low variance = flat/uniform = weak texture region.
    These are the frames NeRF needs to fill in.
    """
    img = cv2.imread(color_files[frame_idx], cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    return cv2.Laplacian(img, cv2.CV_64F).var()

print("Analyzing texture variance across frames...")
texture_scores = []
for i in range(N):
    texture_scores.append(measure_texture_variance(i))
texture_scores = np.array(texture_scores)

weak_texture_frames = np.where(texture_scores < LAPLACIAN_THRESHOLD)[0]
strong_texture_frames = np.where(texture_scores >= LAPLACIAN_THRESHOLD)[0]

print(f"  Strong-texture frames: {len(strong_texture_frames)} "
      f"({len(strong_texture_frames)/N*100:.1f}%)")
print(f"  Weak-texture frames:   {len(weak_texture_frames)} "
      f"({len(weak_texture_frames)/N*100:.1f}%)")
print(f"  Variance range: [{texture_scores.min():.1f}, {texture_scores.max():.1f}]")


# =========================================================
# SECTION 4 — SLAM Odometry + Loop Closure
# =========================================================
def save_poses(poses, path):
    with open(path, 'w') as f:
        json.dump([p.tolist() for p in poses], f)
    print(f"Poses saved → {path}")

def load_poses(path, n):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        poses = [np.array(p) for p in json.load(f)]
    while len(poses) < n:
        poses.append(poses[-1].copy())
    return poses[:n]

poses = load_poses(POSES_CKPT, N)

if poses is None:
    print(f"\nRunning SLAM odometry on {N} frames...")
    pose_graph = o3d.pipelines.registration.PoseGraph()
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.eye(4)))

    curr_pose = np.eye(4)
    prev_rgbd = load_rgbd(0)
    poses     = [np.eye(4)]
    keyframes = []   # list of (frame_idx, pcd, world_pose)
    failed    = 0

    if prev_rgbd is None:
        raise RuntimeError("Failed to load frame 0.")

    for i in range(1, N):
        curr_rgbd = load_rgbd(i)
        if curr_rgbd is None:
            poses.append(poses[-1].copy())
            pose_graph.nodes.append(
                o3d.pipelines.registration.PoseGraphNode(curr_pose.copy())
            )
            failed += 1
            continue

        success, trans, info = o3d.pipelines.odometry.compute_rgbd_odometry(
            prev_rgbd, curr_rgbd,
            intrinsic,
            np.eye(4),
            o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm()
        )

        if success:
            curr_pose = curr_pose @ trans
        else:
            info  = np.eye(6) * 1e-6
            trans = np.eye(4)
            failed += 1

        poses.append(curr_pose.copy())
        pose_graph.nodes.append(
            o3d.pipelines.registration.PoseGraphNode(curr_pose.copy())
        )
        pose_graph.edges.append(
            o3d.pipelines.registration.PoseGraphEdge(
                i-1, i, trans, info, uncertain=not success
            )
        )

        # Loop closure at keyframe intervals
        if i % KEYFRAME_STEP == 0:
            curr_pcd = rgbd_to_pcd(curr_rgbd)
            lc_added = 0
            for kf_idx, kf_pcd, kf_pose in keyframes:
                if lc_added >= 3:   # max 3 loop edges per keyframe
                    break
                init_guess = np.linalg.inv(kf_pose) @ curr_pose
                result = o3d.pipelines.registration.registration_icp(
                    curr_pcd, kf_pcd,
                    VOXEL_SIZE * 3,
                    init_guess,
                    o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
                )
                if result.fitness > LOOP_FITNESS_THRESH:
                    lc_info = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
                        curr_pcd, kf_pcd, VOXEL_SIZE * 3, result.transformation
                    )
                    pose_graph.edges.append(
                        o3d.pipelines.registration.PoseGraphEdge(
                            kf_idx, i, result.transformation, lc_info, uncertain=True
                        )
                    )
                    lc_added += 1
            keyframes.append((i, curr_pcd, curr_pose.copy()))

        prev_rgbd = curr_rgbd
        if i % 50 == 0:
            print(f"  Odometry {i}/{N}  failures={failed}")

    print(f"Odometry done. failures={failed}")

    # Global pose graph optimization
    print("Optimizing pose graph...")
    o3d.pipelines.registration.global_optimization(
        pose_graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        o3d.pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance=VOXEL_SIZE * 1.5,
            edge_prune_threshold=0.25,
            reference_node=0
        )
    )
    poses = [node.pose.copy() for node in pose_graph.nodes]
    save_poses(poses, POSES_CKPT)
    print(f"Poses saved. {len(poses)} total.")

# Center poses
positions = np.array([p[:3, 3] for p in poses])
center    = positions.mean(axis=0)
if not np.allclose(center, 0, atol=1e-3):
    for i in range(len(poses)):
        poses[i][:3, 3] -= center

traj_len = np.linalg.norm(np.diff(positions, axis=0), axis=1).sum()
bbox     = positions.max(axis=0) - positions.min(axis=0)
print(f"Trajectory: {traj_len:.2f}m  bbox {bbox[0]:.1f}x{bbox[1]:.1f}x{bbox[2]:.1f}m")


# =========================================================
# SECTION 5 — TSDF Fusion (strong-texture frames only)
# =========================================================
# Use all frames for TSDF — weak-texture frames still contribute
# geometry even if color is flat. NeRF will supplement later.
print(f"\nTSDF fusion ({N} frames)...")
volume = o3d.pipelines.integration.ScalableTSDFVolume(
    voxel_length=VOXEL_SIZE,
    sdf_trunc=VOXEL_SIZE * 4,
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
)

n_fused = min(N, len(poses))
for idx in range(0, n_fused, 2):
    rgbd = load_rgbd(idx)
    if rgbd is None:
        continue
    volume.integrate(rgbd, intrinsic, np.linalg.inv(poses[idx]))

tsdf_mesh = volume.extract_triangle_mesh()
tsdf_mesh.compute_vertex_normals()
tsdf_mesh.remove_degenerate_triangles()
tsdf_mesh.remove_duplicated_vertices()
tsdf_mesh.remove_non_manifold_edges()

# Keep largest component
tri_c, n_tri, _ = tsdf_mesh.cluster_connected_triangles()
tri_c  = np.asarray(tri_c)
n_tri  = np.asarray(n_tri)
if len(n_tri) > 0:
    tsdf_mesh.remove_triangles_by_mask(tri_c != n_tri.argmax())
    tsdf_mesh.remove_unreferenced_vertices()

print(f"TSDF mesh: {len(tsdf_mesh.vertices):,}v  {len(tsdf_mesh.triangles):,}t")
tsdf_ply = os.path.join(OUTPUT_DIR, "tsdf_mesh.ply")
o3d.io.write_triangle_mesh(tsdf_ply, tsdf_mesh)


# =========================================================
# SECTION 6 — Prepare NeRF Dataset (transforms.json)
# =========================================================
# Instant-NGP uses a transforms.json with camera poses in
# NeRF convention (OpenGL: Y-up, Z-backward, right-handed).
# Our poses are in OpenCV convention (Y-down, Z-forward).
# Conversion: flip Y and Z axes.

def opencv_to_nerf(pose):
    """Convert OpenCV camera pose to NeRF/OpenGL convention."""
    flip = np.diag([1, -1, -1, 1]).astype(np.float64)
    return pose @ flip

# Compute field of view from intrinsics
fov_x = 2 * np.arctan(W / (2 * fx))

# Select frames for NeRF — use both weak and strong texture
# but oversample weak-texture frames so NeRF focuses on them
nerf_frames = []
for i in range(0, N, USE_EVERY_N_NERF):
    if i >= len(poses):
        break
    img_src  = color_files[i]
    img_name = f"images/{i:05d}.png"
    img_dst  = os.path.join(NERF_SCENE_DIR, img_name)

    # Copy image
    img = cv2.imread(img_src)
    if img is None:
        continue
    cv2.imwrite(img_dst, img)

    # Convert pose
    c2w = opencv_to_nerf(poses[i])

    nerf_frames.append({
        "file_path": img_name,
        "transform_matrix": c2w.tolist()
    })

# Oversample weak-texture frames (2x weight for NeRF training)
for i in weak_texture_frames:
    if i % USE_EVERY_N_NERF == 0 or i >= len(poses):
        continue   # already included above
    img_src  = color_files[i]
    img_name = f"images/wt_{i:05d}.png"
    img_dst  = os.path.join(NERF_SCENE_DIR, img_name)
    img = cv2.imread(img_src)
    if img is None:
        continue
    cv2.imwrite(img_dst, img)
    c2w = opencv_to_nerf(poses[i])
    nerf_frames.append({"file_path": img_name, "transform_matrix": c2w.tolist()})

transforms = {
    "camera_angle_x": float(fov_x),
    "camera_angle_y": float(2 * np.arctan(H / (2 * fy))),
    "fl_x": float(fx),
    "fl_y": float(fy),
    "cx": float(cx),
    "cy": float(cy),
    "w": W,
    "h": H,
    "aabb_scale": 4,    # 4m scene scale — appropriate for apartment room
    "frames": nerf_frames
}

transforms_path = os.path.join(NERF_SCENE_DIR, "transforms.json")
with open(transforms_path, 'w') as f:
    json.dump(transforms, f, indent=2)
print(f"\nNeRF dataset: {len(nerf_frames)} frames → {transforms_path}")


# =========================================================
# SECTION 7 — Install & Train Instant-NGP
# =========================================================
print("\nInstalling Instant-NGP...")
if not os.path.exists(NERF_DIR):
    os.system("apt-get install -y build-essential cmake libglfw3-dev libglew-dev "
              "libboost-dev libssl-dev python3-dev > /dev/null 2>&1")
    os.system(f"git clone --recursive https://github.com/NVlabs/instant-ngp {NERF_DIR}")
    os.system(f"cd {NERF_DIR} && cmake . -B build -DCMAKE_BUILD_TYPE=Release > /dev/null 2>&1")
    os.system(f"cd {NERF_DIR} && cmake --build build --target testbed -- -j4 2>&1 | tail -5")
    print("Instant-NGP built.")
else:
    print("Instant-NGP already present.")

# Train NeRF
nerf_snapshot = os.path.join(OUTPUT_DIR, "nerf_snapshot.msgpack")
print(f"\nTraining NeRF ({NERF_STEPS} steps)...")
print("This will take 3-8 minutes on T4...")

train_cmd = (
    f"cd {NERF_DIR} && python scripts/run.py "
    f"--scene {NERF_SCENE_DIR} "
    f"--n_steps {NERF_STEPS} "
    f"--save_snapshot {nerf_snapshot} "
    f"--mode nerf"
)
ret = os.system(train_cmd)
if ret != 0:
    print("NeRF training failed — check Instant-NGP build above.")
    print("Falling back to TSDF-only output.")
    nerf_snapshot = None
else:
    print(f"NeRF trained. Snapshot → {nerf_snapshot}")


# =========================================================
# SECTION 8 — Extract NeRF Point Cloud at Weak-Texture Regions
# =========================================================
nerf_pcd = None

if nerf_snapshot and os.path.exists(nerf_snapshot):
    print("\nExtracting NeRF point cloud at weak-texture regions...")

    # Define the bounding volume from the TSDF mesh
    tsdf_bbox = tsdf_mesh.get_axis_aligned_bounding_box()
    bbox_min  = np.asarray(tsdf_bbox.min_bound)
    bbox_max  = np.asarray(tsdf_bbox.max_bound)

    # Extract marching cubes mesh from NeRF at TSDF resolution
    nerf_mesh_path = os.path.join(OUTPUT_DIR, "nerf_mesh_raw.ply")
    extract_cmd = (
        f"cd {NERF_DIR} && python scripts/run.py "
        f"--scene {NERF_SCENE_DIR} "
        f"--load_snapshot {nerf_snapshot} "
        f"--marching_cubes_res 256 "
        f"--save_mesh {nerf_mesh_path} "
        f"--mode nerf"
    )
    ret = os.system(extract_cmd)

    if ret == 0 and os.path.exists(nerf_mesh_path):
        nerf_mesh_raw = o3d.io.read_triangle_mesh(nerf_mesh_path)
        # Sample to point cloud
        nerf_pcd = nerf_mesh_raw.sample_points_poisson_disk(
            number_of_points=500_000
        )
        print(f"NeRF mesh → {len(nerf_pcd.points):,} points extracted")
    else:
        print("NeRF mesh extraction failed — using TSDF only")


# =========================================================
# SECTION 9 — Weak-Texture Region Detection on TSDF Mesh
# =========================================================
# Identify mesh vertices that are poorly supported by point cloud data.
# These are the weak-texture holes/flat regions NeRF should fill.

def find_weak_texture_vertices(mesh, pcd_support, radius=0.05):
    """
    Vertices in the mesh that have fewer than min_neighbors support
    points within `radius` are considered weak-texture regions.
    Returns a boolean mask: True = weak.
    """
    if pcd_support is None or len(pcd_support.points) == 0:
        return np.zeros(len(mesh.vertices), dtype=bool)

    verts = np.asarray(mesh.vertices)
    # Build KD-tree on support point cloud
    kd = o3d.geometry.KDTreeFlann(pcd_support)
    weak = np.zeros(len(verts), dtype=bool)
    for vi, v in enumerate(verts):
        k, _, _ = kd.search_radius_vector_3d(v, radius)
        if k < 3:
            weak[vi] = True
    return weak

# Build support point cloud from all TSDF frames
print("\nBuilding TSDF support point cloud...")
support_pcd = o3d.geometry.PointCloud()
for idx in range(0, min(N, len(poses)), 5):
    rgbd = load_rgbd(idx, preprocess=False)
    if rgbd is None:
        continue
    frame_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
    frame_pcd.transform(poses[idx])
    support_pcd += frame_pcd

support_pcd = support_pcd.voxel_down_sample(VOXEL_SIZE)
print(f"Support cloud: {len(support_pcd.points):,} points")

weak_mask = find_weak_texture_vertices(tsdf_mesh, support_pcd)
print(f"Weak-texture vertices: {weak_mask.sum():,} / {len(weak_mask):,} "
      f"({weak_mask.mean()*100:.1f}%)")


# =========================================================
# SECTION 10 — Merge TSDF + NeRF
# =========================================================
if nerf_pcd is not None and len(nerf_pcd.points) > 0:
    print("\nMerging TSDF mesh + NeRF point cloud...")

    # Strategy: keep TSDF mesh for strong-texture regions,
    # add NeRF points only where TSDF is weak.
    # Then run Poisson reconstruction on the combined point cloud.

    # Convert TSDF mesh to point cloud
    tsdf_pcd = tsdf_mesh.sample_points_poisson_disk(number_of_points=1_000_000)

    # Filter NeRF points to only those near weak-texture vertices
    if weak_mask.any():
        weak_verts = np.asarray(tsdf_mesh.vertices)[weak_mask]
        weak_cloud = o3d.geometry.PointCloud()
        weak_cloud.points = o3d.utility.Vector3dVector(weak_verts)
        kd_weak = o3d.geometry.KDTreeFlann(weak_cloud)

        nerf_pts    = np.asarray(nerf_pcd.points)
        keep_mask   = np.zeros(len(nerf_pts), dtype=bool)
        FILL_RADIUS = 0.15   # accept NeRF points within 15cm of weak vertices

        for ni, pt in enumerate(nerf_pts):
            k, _, _ = kd_weak.search_radius_vector_3d(pt, FILL_RADIUS)
            if k > 0:
                keep_mask[ni] = True

        nerf_filtered = nerf_pcd.select_by_index(np.where(keep_mask)[0])
        print(f"NeRF points in weak regions: {len(nerf_filtered.points):,}")
    else:
        nerf_filtered = nerf_pcd

    # Combine
    merged_pcd = tsdf_pcd + nerf_filtered
    merged_pcd = merged_pcd.voxel_down_sample(VOXEL_SIZE)
    merged_pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=VOXEL_SIZE * 4, max_nn=30)
    )
    merged_pcd.orient_normals_consistent_tangent_plane(k=15)
    print(f"Merged cloud: {len(merged_pcd.points):,} points")

    # Poisson reconstruction on merged cloud
    print("Running Poisson reconstruction on merged cloud...")
    t0 = time.time()
    final_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        merged_pcd, depth=9, width=0, scale=1.1, linear_fit=False, n_threads=-1
    )
    print(f"Poisson done in {time.time()-t0:.1f}s")

    # Remove low-density floaters
    densities = np.asarray(densities)
    final_mesh.remove_vertices_by_mask(densities < np.quantile(densities, 0.05))

else:
    print("\nNo NeRF output — using TSDF mesh directly.")
    final_mesh = tsdf_mesh


# =========================================================
# SECTION 11 — Final Cleanup & Export
# =========================================================
print("\nFinal mesh cleanup...")
final_mesh.remove_degenerate_triangles()
final_mesh.remove_duplicated_vertices()
final_mesh.remove_non_manifold_edges()

tri_c, n_tri, _ = final_mesh.cluster_connected_triangles()
tri_c  = np.asarray(tri_c)
n_tri  = np.asarray(n_tri)
if len(n_tri) > 0:
    final_mesh.remove_triangles_by_mask(tri_c != n_tri.argmax())
    final_mesh.remove_unreferenced_vertices()

final_mesh = final_mesh.filter_smooth_taubin(number_of_iterations=5)
final_mesh.compute_vertex_normals()

print(f"Final mesh: {len(final_mesh.vertices):,}v  {len(final_mesh.triangles):,}t")

mesh_path = os.path.join(OUTPUT_DIR, "final_mesh.ply")
obj_path  = os.path.join(OUTPUT_DIR, "final_mesh.obj")
o3d.io.write_triangle_mesh(mesh_path, final_mesh)
o3d.io.write_triangle_mesh(obj_path,  final_mesh)

# Save TSDF mesh separately for comparison
o3d.io.write_triangle_mesh(tsdf_ply, tsdf_mesh)

print(f"\nOutputs saved to {OUTPUT_DIR}:")
print(f"  final_mesh.ply   — TSDF + NeRF hybrid")
print(f"  tsdf_mesh.ply    — TSDF only (for comparison)")
print(f"  final_mesh.obj   — OBJ export for Blender/Unity")


# =========================================================
# SECTION 12 — Preview
# =========================================================
try:
    import matplotlib.pyplot as plt

    renderer = o3d.visualization.rendering.OffscreenRenderer(900, 700)
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = 'defaultLit'
    renderer.scene.add_geometry('mesh', final_mesh, mat)
    renderer.scene.set_background([0.08, 0.08, 0.08, 1.0])
    renderer.scene.scene.enable_sun_light(True)
    renderer.setup_camera(60.0, final_mesh.get_axis_aligned_bounding_box(), np.eye(4))
    img = np.asarray(renderer.render_to_image())

    plt.figure(figsize=(11, 8))
    plt.imshow(img)
    plt.axis('off')
    plt.tight_layout()
    preview_path = os.path.join(OUTPUT_DIR, 'preview.png')
    plt.savefig(preview_path, dpi=120, bbox_inches='tight')
    plt.show()
    print(f"Preview → {preview_path}")
except Exception as e:
    print(f"Preview skipped ({e})")
