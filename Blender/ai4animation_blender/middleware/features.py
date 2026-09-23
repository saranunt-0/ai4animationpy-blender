# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Model-facing features computed from AI4Animation-space data.

Everything here operates in AI4Animation world space (Y-up, +Z forward,
meters) and mirrors the formulas used during training, so that data
authored in Blender means the same thing to the network:

* compute_root        == RootModule.Compute (BIPED topology, GROUND reference)
* guidance_from_pose  == GuidanceModule.GetLegacyGuidance for a single frame
* velocities          == finite differences, as MotionModule/Actor use
"""

import numpy as np

from . import conventions as cv


def look_planar(forward):
    """Rotation with Z = normalized horizontal forward and Y = world up.

    A (near) vertical or zero forward has no horizontal facing; +Z is used
    instead of producing a singular matrix.
    """
    f = cv.as_matrix(forward).copy()
    f[..., 1] = 0.0
    norm = np.linalg.norm(f, axis=-1, keepdims=True)
    f = np.where(norm < 1e-6, np.array([0.0, 0.0, 1.0]), f)
    f = f / np.maximum(np.linalg.norm(f, axis=-1, keepdims=True), cv.EPS)
    up = np.zeros_like(f)
    up[..., 1] = 1.0
    x = np.cross(up, f)
    return np.stack((x, up, f), axis=-1)


def compute_root(transforms, profile):
    """Character root from a pose, identical to RootModule.Compute(BIPED, GROUND).

    transforms: (..., J, 4, 4) AI4Animation world transforms in profile bone order.
    Returns (..., 4, 4): position = hip projected to the ground (y = 0),
    Z axis = horizontal facing derived from hip and shoulder lines.
    """
    names = profile.bone_names
    rb = profile.root_bones
    p = cv.as_matrix(transforms)[..., :3, 3]

    def pos(key):
        return p[..., names.index(rb[key]), :]

    hip = pos("hip")
    up = np.zeros_like(hip)
    up[..., 1] = 1.0

    def horizontal_unit(v):
        v = v - np.sum(v * up, axis=-1, keepdims=True) * up
        return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), cv.EPS)

    across = horizontal_unit(pos("left_hip") - pos("right_hip")) + horizontal_unit(
        pos("left_shoulder") - pos("right_shoulder")
    )
    across = across / np.maximum(np.linalg.norm(across, axis=-1, keepdims=True), cv.EPS)
    forward = horizontal_unit(np.cross(across, up))

    root = np.zeros(hip.shape[:-1] + (4, 4))
    root[..., 3, 3] = 1.0
    root[..., :3, :3] = np.stack((np.cross(up, forward), up, forward), axis=-1)
    root[..., 0, 3] = hip[..., 0]
    root[..., 2, 3] = hip[..., 2]
    return root


def root_from_position_direction(position, direction):
    """Ground root at position facing direction (both AI4Animation space)."""
    root = np.eye(4)
    root[:3, :3] = look_planar(direction)
    root[:3, 3] = cv.as_matrix(position)
    root[1, 3] = 0.0
    return root


def reroot(transforms, from_root, to_root):
    """Rigidly move a pose so that from_root lands on to_root."""
    delta = cv.as_matrix(to_root) @ np.linalg.inv(cv.as_matrix(from_root))
    return np.einsum("ij,...jk->...ik", delta, cv.as_matrix(transforms))


def velocities(previous_transforms, current_transforms, dt):
    """Linear bone velocities (m/s) from two consecutive poses."""
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    prev = cv.as_matrix(previous_transforms)[..., :3, 3]
    cur = cv.as_matrix(current_transforms)[..., :3, 3]
    return (cur - prev) / dt


def guidance_from_pose(transforms, profile, root=None):
    """Guidance positions (J, 3) = bone positions expressed in the root frame.

    This is what the Guidances/*.npz files contain (GuidanceModule legacy
    guidance, with a zero-width smoothing window).
    """
    t = cv.as_matrix(transforms)
    if root is None:
        root = compute_root(t, profile)
    inv = np.linalg.inv(root)
    return np.einsum("ij,bj->bi", inv[:3, :3], t[:, :3, 3]) + inv[:3, 3]


def resample_polyline(points, spacing):
    """Resample a polyline at (approximately) equal arc-length spacing.

    The model's path follower advances the Catmull-Rom parameter uniformly per
    control point, so evenly spaced control points give a constant walk speed.
    """
    pts = cv.as_matrix(points).reshape(-1, 3)
    if pts.shape[0] < 2:
        return pts.copy()
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=-1)
    keep = np.concatenate(([True], seg > 1e-9))
    pts = pts[keep]
    if pts.shape[0] < 2:
        return pts[:1].copy()
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=-1)
    cumulative = np.concatenate(([0.0], np.cumsum(seg)))
    total = cumulative[-1]
    count = max(2, int(np.ceil(total / max(spacing, 1e-6))) + 1)
    samples = np.linspace(0.0, total, count)
    out = np.empty((count, 3))
    for axis in range(3):
        out[:, axis] = np.interp(samples, cumulative, pts[:, axis])
    return out


def blender_path_to_ai4a(points_blender_world, meters_per_unit=1.0, spacing=0.25, project_to_ground=True):
    """Blender world polyline -> AI4Animation path control points."""
    pts = cv.points_blender_to_ai4a(points_blender_world, meters_per_unit)
    if project_to_ground:
        pts = pts.copy()
        pts[:, 1] = 0.0
    return resample_polyline(pts, spacing)


def auto_planner_grid(points, obstacle_centers, obstacle_sizes, cell=0.8, margin=2.0, height=2.0):
    """Voxel grid (center, size, resolution) covering points and obstacles.

    Mirrors the demo layout (a single layer of cells, CENTER.y = height / 2)
    and sizes the XZ extent to the scene instead of a fixed 10 x 10 m.
    All inputs/outputs in AI4Animation space.
    """
    pts = [np.asarray(points, dtype=float).reshape(-1, 3)]
    centers = np.asarray(obstacle_centers, dtype=float).reshape(-1, 3)
    sizes = np.asarray(obstacle_sizes, dtype=float).reshape(-1, 3)
    if centers.shape[0]:
        pts += [centers - 0.5 * sizes, centers + 0.5 * sizes]
    allp = np.concatenate(pts, axis=0)
    lo = allp.min(axis=0) - margin
    hi = allp.max(axis=0) + margin
    size = np.array([hi[0] - lo[0], height, hi[2] - lo[2]])
    center = np.array([0.5 * (lo[0] + hi[0]), 0.5 * height, 0.5 * (lo[2] + hi[2])])
    resolution = [max(2, int(np.ceil(size[0] / cell))), 1, max(2, int(np.ceil(size[2] / cell)))]
    return center, size, resolution


def blender_boxes_to_ai4a(world_corners, meters_per_unit=1.0):
    """World-space bounding-box corners (N, 8, 3) of Blender objects -> AABB center/size.

    The path planner only supports axis-aligned boxes; rotated obstacles
    are conservatively replaced by their axis-aligned bounds.
    """
    c = cv.points_blender_to_ai4a(np.asarray(world_corners, dtype=float).reshape(-1, 8, 3), meters_per_unit)
    lo, hi = c.min(axis=1), c.max(axis=1)
    return 0.5 * (lo + hi), hi - lo
