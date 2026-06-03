import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_points(path):
    path = Path(path)

    if path.suffix == ".bin":
        # 일반적인 lidar bin: x, y, z, intensity
        points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    elif path.suffix == ".npy":
        points = np.load(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    return points


def hard_voxelize_numpy(
    points,
    voxel_size,
    point_cloud_range,
    max_points_per_voxel=10,
    max_voxels=120000,
):
    """
    points: [N, C], at least x, y, z
    voxel_size: [vx, vy, vz]
    point_cloud_range: [x_min, y_min, z_min, x_max, y_max, z_max]

    return:
        voxels: [M, max_points_per_voxel, C]
        coordinates: [M, 3], order = [z, y, x]
        num_points_per_voxel: [M]
    """

    voxel_size = np.array(voxel_size, dtype=np.float32)
    pc_range = np.array(point_cloud_range, dtype=np.float32)

    xyz = points[:, :3]

    # point cloud range filtering
    mask = (
        (xyz[:, 0] >= pc_range[0])
        & (xyz[:, 0] < pc_range[3])
        & (xyz[:, 1] >= pc_range[1])
        & (xyz[:, 1] < pc_range[4])
        & (xyz[:, 2] >= pc_range[2])
        & (xyz[:, 2] < pc_range[5])
    )
    points = points[mask]
    xyz = points[:, :3]

    grid_size = np.floor((pc_range[3:] - pc_range[:3]) / voxel_size).astype(np.int32)
    # grid_size = [x_size, y_size, z_size]

    voxel_indices_xyz = np.floor((xyz - pc_range[:3]) / voxel_size).astype(np.int32)

    voxel_dict = {}

    for point, voxel_xyz in zip(points, voxel_indices_xyz):
        x_idx, y_idx, z_idx = voxel_xyz

        # coordinate order를 일반적인 sparse conv 형식처럼 z, y, x로 저장
        coord = (z_idx, y_idx, x_idx)

        if coord not in voxel_dict:
            if len(voxel_dict) >= max_voxels:
                continue
            voxel_dict[coord] = []

        if len(voxel_dict[coord]) < max_points_per_voxel:
            voxel_dict[coord].append(point)

    num_voxels = len(voxel_dict)
    num_features = points.shape[1]

    voxels = np.zeros(
        (num_voxels, max_points_per_voxel, num_features),
        dtype=np.float32,
    )
    coordinates = np.zeros((num_voxels, 3), dtype=np.int32)
    num_points_per_voxel = np.zeros((num_voxels,), dtype=np.int32)

    for i, (coord, pts) in enumerate(voxel_dict.items()):
        pts = np.array(pts, dtype=np.float32)
        coordinates[i] = np.array(coord, dtype=np.int32)
        num_points_per_voxel[i] = len(pts)
        voxels[i, : len(pts)] = pts

    return voxels, coordinates, num_points_per_voxel, grid_size


def visualize_voxel_bev(
    coordinates,
    num_points_per_voxel,
    grid_size,
    title="Voxel BEV",
    save_path=None,
):
    """
    coordinates: [M, 3], order = z, y, x
    grid_size: [x_size, y_size, z_size]
    """

    x_size, y_size, z_size = grid_size

    bev = np.zeros((y_size, x_size), dtype=np.float32)

    # z, y, x
    ys = coordinates[:, 1]
    xs = coordinates[:, 2]

    # 같은 x,y 위치에 여러 z voxel이 있을 수 있으므로 max로 표현
    for x, y, count in zip(xs, ys, num_points_per_voxel):
        bev[y, x] = max(bev[y, x], count)

    plt.figure(figsize=(10, 10))
    plt.imshow(bev, origin="lower")
    plt.colorbar(label="num_points_per_voxel")
    plt.title(title)
    plt.xlabel("voxel x index")
    plt.ylabel("voxel y index")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200)

    plt.show()


def visualize_original_points_bev(
    points,
    point_cloud_range,
    title="Original Point Cloud BEV",
    save_path=None,
):
    pc_range = np.array(point_cloud_range, dtype=np.float32)

    mask = (
        (points[:, 0] >= pc_range[0])
        & (points[:, 0] < pc_range[3])
        & (points[:, 1] >= pc_range[1])
        & (points[:, 1] < pc_range[4])
    )
    points = points[mask]

    plt.figure(figsize=(10, 10))
    plt.scatter(points[:, 0], points[:, 1], s=0.2)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200)

    plt.show()


if __name__ == "__main__":
    point_path = "/mnt/e/nuScenes/v1.0-mini/samples/LIDAR_TOP/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151603547590.pcd.bin"

    voxel_size = [0.075, 0.075, 0.2]
    point_cloud_range = [-54, -54, -5, 54, 54, 3]

    points = load_points(point_path)

    visualize_original_points_bev(
        points,
        point_cloud_range,
        title="Original Points BEV",
        save_path="original_points_bev.png",
    )

    for max_points_per_voxel in [5, 10, 20]:
        for max_voxels in [30000, 60000, 120000]:
            voxels, coords, num_points, grid_size = hard_voxelize_numpy(
                points,
                voxel_size=voxel_size,
                point_cloud_range=point_cloud_range,
                max_points_per_voxel=max_points_per_voxel,
                max_voxels=max_voxels,
            )

            print("=" * 80)
            print(f"max_points_per_voxel = {max_points_per_voxel}")
            print(f"max_voxels            = {max_voxels}")
            print(f"created voxels        = {len(coords)}")
            print(f"grid_size             = {grid_size.tolist()}")
            print(f"mean points / voxel   = {num_points.mean():.3f}")
            print(f"max points / voxel    = {num_points.max()}")

            visualize_voxel_bev(
                coords,
                num_points,
                grid_size,
                title=(
                    f"Voxel BEV | "
                    f"max_points={max_points_per_voxel}, "
                    f"max_voxels={max_voxels}, "
                    f"created={len(coords)}"
                ),
                save_path=(
                    f"voxel_bev_points{max_points_per_voxel}" f"_voxels{max_voxels}.png"
                ),
            )
