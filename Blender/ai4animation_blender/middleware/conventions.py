# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Coordinate conventions shared by Blender and AI4Animation.

Pure NumPy - no bpy, no torch - so it runs inside Blender's bundled Python,
inside the AI4Animation environment, and in unit tests.

Conventions (verified against the code base, see Blender/README.md):

=====================  ===============================  ==========================
                       Blender                          AI4Animation
=====================  ===============================  ==========================
Up axis                +Z                               +Y
Character forward      -Y                               +Z (Transform.GetAxisZ)
Handedness             right-handed                     right-handed
Matrices               4x4, column vectors (M @ p)      4x4, column vectors
Units                  Blender units * scale_length     meters
Quaternion layout      (w, x, y, z)                     (x, y, z, w)
Bone transforms        armature space (pose_bone.matrix) world space (Actor.Transforms)
=====================  ===============================  ==========================

The basis change is a proper rotation (det = +1):
    ai4a = C @ blender,   (x, y, z)_blender -> (x, z, -y)_ai4a
It is the same mapping the Blender glTF importer/exporter uses.
"""

import numpy as np

# Blender world -> AI4Animation world (rotation only, det = +1).
BLENDER_TO_AI4A = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]], dtype=np.float64
)
AI4A_TO_BLENDER = BLENDER_TO_AI4A.T.copy()

BLENDER_TO_AI4A_4x4 = np.eye(4)
BLENDER_TO_AI4A_4x4[:3, :3] = BLENDER_TO_AI4A
AI4A_TO_BLENDER_4x4 = np.eye(4)
AI4A_TO_BLENDER_4x4[:3, :3] = AI4A_TO_BLENDER

EPS = 1e-8


def as_matrix(values):
    """Convert anything matrix-like (numpy, nested lists, mathutils.Matrix) to float64."""
    return np.array(values, dtype=np.float64)


def rigid(matrices, uniform_tolerance=1e-4):
    """Remove uniform scale from 4x4 transforms (keeps translation untouched).

    Blender armatures are often scaled (e.g. 0.01 after an FBX import), which
    leaks into world-space bone matrices. The model only understands rigid
    transforms, so the rotation columns are normalized. Non-uniform scale or
    shear cannot be represented and raises a ValueError instead of silently
    producing wrong axes.
    """
    m = as_matrix(matrices).copy()
    norms = np.linalg.norm(m[..., :3, :3], axis=-2)  # column norms, shape (..., 3)
    if np.any(norms < EPS):
        raise ValueError("Degenerate (zero-scale) transform cannot be converted.")
    spread = (norms.max(axis=-1) - norms.min(axis=-1)) / norms.max(axis=-1)
    if np.any(spread > uniform_tolerance):
        raise ValueError(
            "Non-uniform scale or shear found (relative spread %.3g). "
            "Apply scale on the armature (Ctrl+A > Scale) before converting."
            % float(spread.max())
        )
    m[..., :3, :3] = m[..., :3, :3] / norms[..., None, :]
    return m


def orthonormalize_zy(rotations):
    """Gram-Schmidt with Z as primary and Y as secondary axis.

    AI4Animation builds rotations with Rotation.Look(z, y), which normalizes
    z and y but does not make them perpendicular. The network reads the Z
    (forward) and Y (up) columns as separate features, so Z is kept exactly,
    Y is projected onto the plane orthogonal to Z and X = Y x Z.
    """
    r = as_matrix(rotations)
    z = r[..., :, 2]
    z = z / np.maximum(np.linalg.norm(z, axis=-1, keepdims=True), EPS)
    y = r[..., :, 1]
    y = y - np.sum(y * z, axis=-1, keepdims=True) * z
    y_norm = np.linalg.norm(y, axis=-1, keepdims=True)
    if np.any(y_norm < 1e-6):
        # Y parallel to Z: fall back to the X column to recover a valid frame.
        x = r[..., :, 0]
        y_alt = np.cross(z, x)
        y = np.where(y_norm < 1e-6, y_alt, y)
        y_norm = np.linalg.norm(y, axis=-1, keepdims=True)
    y = y / np.maximum(y_norm, EPS)
    x = np.cross(y, z)
    return np.stack((x, y, z), axis=-1)


def orthonormalize_transforms(transforms):
    t = as_matrix(transforms).copy()
    t[..., :3, :3] = orthonormalize_zy(t[..., :3, :3])
    return t


def points_blender_to_ai4a(points, meters_per_unit=1.0):
    p = as_matrix(points)
    return np.einsum("ij,...j->...i", BLENDER_TO_AI4A, p) * meters_per_unit


def points_ai4a_to_blender(points, meters_per_unit=1.0):
    p = as_matrix(points)
    return np.einsum("ij,...j->...i", AI4A_TO_BLENDER, p) / meters_per_unit


def directions_blender_to_ai4a(vectors, meters_per_unit=1.0):
    """Directions and velocities rotate like points but ignore translation."""
    return points_blender_to_ai4a(vectors, meters_per_unit)


def directions_ai4a_to_blender(vectors, meters_per_unit=1.0):
    return points_ai4a_to_blender(vectors, meters_per_unit)


def transforms_blender_to_ai4a(world_matrices, meters_per_unit=1.0):
    """Blender world 4x4 -> AI4Animation world 4x4 WITHOUT any bone-frame fix.

    Only the axes/units change here: C @ rigid(M). Bone-local axis
    conventions are handled by rig.Calibration on top of this.
    """
    m = rigid(world_matrices)
    out = np.einsum("ij,...jk->...ik", BLENDER_TO_AI4A_4x4, m)
    out[..., :3, 3] *= meters_per_unit
    return out


def transforms_ai4a_to_blender(world_matrices, meters_per_unit=1.0):
    m = as_matrix(world_matrices).copy()
    m[..., :3, 3] /= meters_per_unit
    return np.einsum("ij,...jk->...ik", AI4A_TO_BLENDER_4x4, m)


def object_frames_blender_to_ai4a(world_matrices, meters_per_unit=1.0):
    """Scene objects (targets, empties): Blender object frame -> AI4Animation frame.

    A Blender object "faces" its local -Y with +Z up; an AI4Animation root or
    goal faces its local +Z with +Y up. Converting the frame as well as the
    axes gives C @ M @ C^T, so an unrotated Blender empty becomes an identity
    goal (facing +Z). Using transforms_blender_to_ai4a here would make the
    object's up axis the model's forward axis.
    """
    m = transforms_blender_to_ai4a(world_matrices, meters_per_unit)
    return np.einsum("...ij,jk->...ik", m, AI4A_TO_BLENDER_4x4)


def object_frames_ai4a_to_blender(world_matrices, meters_per_unit=1.0):
    """Inverse of object_frames_blender_to_ai4a (roots/goals -> Blender empties)."""
    m = transforms_ai4a_to_blender(world_matrices, meters_per_unit)
    return np.einsum("...ij,jk->...ik", m, BLENDER_TO_AI4A_4x4)


# ----------------------------------------------------------------------------
# Quaternions (Blender order: w, x, y, z)
# ----------------------------------------------------------------------------


def quaternions_from_matrices(rotations):
    """Rotation matrices (..., 3, 3) -> unit quaternions (..., 4) as (w, x, y, z).

    Shepperd's method, stable for 180 degree rotations.
    """
    r = as_matrix(rotations)
    shape = r.shape[:-2]
    r = r.reshape(-1, 3, 3)
    m00, m01, m02 = r[:, 0, 0], r[:, 0, 1], r[:, 0, 2]
    m10, m11, m12 = r[:, 1, 0], r[:, 1, 1], r[:, 1, 2]
    m20, m21, m22 = r[:, 2, 0], r[:, 2, 1], r[:, 2, 2]
    trace = m00 + m11 + m22
    q = np.empty((r.shape[0], 4))

    case0 = trace > 0.0
    case1 = (~case0) & (m00 >= m11) & (m00 >= m22)
    case2 = (~case0) & (~case1) & (m11 >= m22)
    case3 = ~(case0 | case1 | case2)

    s = np.sqrt(np.maximum(trace[case0] + 1.0, 0.0)) * 2.0
    q[case0] = np.stack(
        (0.25 * s, (m21 - m12)[case0] / s, (m02 - m20)[case0] / s, (m10 - m01)[case0] / s),
        axis=-1,
    )
    s = np.sqrt(np.maximum(1.0 + m00 - m11 - m22, 0.0)[case1]) * 2.0
    q[case1] = np.stack(
        ((m21 - m12)[case1] / s, 0.25 * s, (m01 + m10)[case1] / s, (m02 + m20)[case1] / s),
        axis=-1,
    )
    s = np.sqrt(np.maximum(1.0 + m11 - m00 - m22, 0.0)[case2]) * 2.0
    q[case2] = np.stack(
        ((m02 - m20)[case2] / s, (m01 + m10)[case2] / s, 0.25 * s, (m12 + m21)[case2] / s),
        axis=-1,
    )
    s = np.sqrt(np.maximum(1.0 + m22 - m00 - m11, 0.0)[case3]) * 2.0
    q[case3] = np.stack(
        ((m10 - m01)[case3] / s, (m02 + m20)[case3] / s, (m12 + m21)[case3] / s, 0.25 * s),
        axis=-1,
    )
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    return q.reshape(shape + (4,))


def matrices_from_quaternions(quaternions):
    """Unit quaternions (..., 4) as (w, x, y, z) -> rotation matrices (..., 3, 3)."""
    q = as_matrix(quaternions)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    r = np.empty(q.shape[:-1] + (3, 3))
    r[..., 0, 0] = 1 - 2 * (y * y + z * z)
    r[..., 0, 1] = 2 * (x * y - z * w)
    r[..., 0, 2] = 2 * (x * z + y * w)
    r[..., 1, 0] = 2 * (x * y + z * w)
    r[..., 1, 1] = 1 - 2 * (x * x + z * z)
    r[..., 1, 2] = 2 * (y * z - x * w)
    r[..., 2, 0] = 2 * (x * z - y * w)
    r[..., 2, 1] = 2 * (y * z + x * w)
    r[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return r


def make_quaternions_continuous(quaternions, axis=0):
    """Flip signs so consecutive keys stay in the same hemisphere.

    q and -q are the same rotation, but Blender interpolates quaternion
    F-curves component-wise, so a sign flip between two keys makes the bone
    spin the long way around. Works along `axis` (the frame axis).
    """
    q = np.moveaxis(as_matrix(quaternions).copy(), axis, 0)
    for i in range(1, q.shape[0]):
        flip = np.sum(q[i] * q[i - 1], axis=-1) < 0.0
        q[i][flip] *= -1.0
    return np.moveaxis(q, 0, axis)


def rotation_angle_deg(a, b):
    """Angle (degrees) between rotation matrices a and b, elementwise.

    Uses ||A - B||_F = 2 sqrt(2) sin(theta / 2), which stays precise for
    small angles (arccos of the trace loses ~1e-3 degrees on float32 input).
    """
    diff = np.linalg.norm(as_matrix(a) - as_matrix(b), axis=(-2, -1))
    return np.degrees(2.0 * np.arcsin(np.clip(diff / (2.0 * np.sqrt(2.0)), 0.0, 1.0)))
