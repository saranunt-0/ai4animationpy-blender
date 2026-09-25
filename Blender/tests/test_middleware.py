# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Middleware tests: pure NumPy, no Blender and no torch required."""

import numpy as np
import pytest
from synthetic import SyntheticRig, random_model_poses, random_rotation, rotation_about, tr

from ai4animation_blender.middleware import conventions as cv
from ai4animation_blender.middleware import exchange, features, rig

POS_TOL = 1e-8
DEG_TOL = 1e-6


# ----------------------------------------------------------------------------
# Conventions
# ----------------------------------------------------------------------------


def test_basis_change_is_proper_rotation():
    c = cv.BLENDER_TO_AI4A
    assert np.allclose(c @ c.T, np.eye(3))
    assert np.isclose(np.linalg.det(c), 1.0)


def test_axes_map_up_and_forward():
    # Blender up (+Z) -> AI4A up (+Y); Blender character forward (-Y) -> AI4A +Z
    assert np.allclose(cv.points_blender_to_ai4a([0, 0, 1]), [0, 1, 0])
    assert np.allclose(cv.points_blender_to_ai4a([0, -1, 0]), [0, 0, 1])
    assert np.allclose(cv.points_blender_to_ai4a([1, 0, 0]), [1, 0, 0])


def test_units_scale_positions_not_rotations():
    m = tr(rotation_about([0, 0, 1], 30), [100.0, 0, 0])
    a = cv.transforms_blender_to_ai4a(m, meters_per_unit=0.01)
    assert np.allclose(a[:3, 3], [1.0, 0, 0])
    assert np.allclose(a[:3, :3] @ a[:3, :3].T, np.eye(3))
    back = cv.transforms_ai4a_to_blender(a, meters_per_unit=0.01)
    assert np.allclose(back, m)


def test_object_frames_use_character_convention():
    # An unrotated Blender empty faces -Y with +Z up == an identity AI4A goal.
    assert np.allclose(cv.object_frames_blender_to_ai4a(np.eye(4)), np.eye(4))
    # Yaw +90 deg about Blender Z: the empty's -Y now points to Blender +X.
    m = tr(rotation_about([0, 0, 1], 90.0), [1.0, 2.0, 0.0])
    a = cv.object_frames_blender_to_ai4a(m)
    assert np.allclose(a[:3, 2], [1.0, 0.0, 0.0])  # forward
    assert np.allclose(a[:3, 1], [0.0, 1.0, 0.0])  # up
    assert np.allclose(a[:3, 3], [1.0, 0.0, -2.0])
    assert np.allclose(cv.object_frames_ai4a_to_blender(a), m)


def test_rigid_removes_uniform_scale_and_rejects_shear():
    m = tr(random_rotation(np.random.default_rng(3)) * 0.01, [1, 2, 3])
    r = cv.rigid(m)
    assert np.allclose(r[:3, :3] @ r[:3, :3].T, np.eye(3))
    assert np.allclose(r[:3, 3], [1, 2, 3])
    bad = m.copy()
    bad[:3, 0] *= 2.0
    with pytest.raises(ValueError):
        cv.rigid(bad)


def test_orthonormalize_keeps_forward_axis():
    poses = random_model_poses(
        rig.RigProfile(["a"], [None], np.eye(4)[None], {}), 20, orthonormal=False
    )
    r = poses[..., :3, :3]
    o = cv.orthonormalize_zy(r)
    assert np.allclose(np.einsum("...ji,...jk->...ik", o, o), np.eye(3), atol=1e-12)
    assert np.allclose(np.linalg.det(o), 1.0)
    assert np.allclose(o[..., :, 2], r[..., :, 2] / np.linalg.norm(r[..., :, 2], axis=-1, keepdims=True))


def test_quaternion_roundtrip_and_continuity():
    rng = np.random.default_rng(0)
    rots = np.stack([random_rotation(rng) for _ in range(200)])
    q = cv.quaternions_from_matrices(rots)
    assert np.allclose(cv.matrices_from_quaternions(q), rots, atol=1e-12)
    # 180 degree rotations (Shepperd edge cases)
    for axis in np.eye(3):
        m = rotation_about(axis, 180.0)
        assert np.allclose(cv.matrices_from_quaternions(cv.quaternions_from_matrices(m)), m, atol=1e-12)
    flipped = q.copy()
    flipped[1::2] *= -1
    cont = cv.make_quaternions_continuous(flipped[:, None, :], axis=0)[:, 0]
    assert np.all(np.sum(cont[1:] * cont[:-1], axis=-1) >= 0)
    assert np.allclose(cv.matrices_from_quaternions(cont), rots, atol=1e-12)


# ----------------------------------------------------------------------------
# Bone mapping
# ----------------------------------------------------------------------------


def test_auto_bone_map_handles_prefixes(profile):
    blender = ["mixamorig:" + n for n in profile.bone_names] + ["mixamorig:Neck1"]
    mapping = rig.auto_bone_map(profile.bone_names, blender)
    assert mapping["Hips"] == "mixamorig:Hips"
    with pytest.raises(ValueError, match="Missing"):
        rig.auto_bone_map(profile.bone_names, blender[1:])


# ----------------------------------------------------------------------------
# Calibration and round trips on synthetic armatures
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("rest_rotated", [True, False])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_calibration_recovers_offsets(profile, seed, rest_rotated):
    s = SyntheticRig(profile, seed=seed, rest_rotated=rest_rotated)
    cal = rig.calibrate(s.snapshot(), profile)
    assert cal.max_residual < POS_TOL
    for k, name in enumerate(profile.bone_names):
        assert np.allclose(cal.offsets[k], s.offsets[name], atol=1e-8)
    a = cal.to_ai4a(s.snapshot())
    ref = cv.orthonormalize_transforms(profile.reference_transforms)
    assert np.abs(a[:, :3, 3] - ref[:, :3, 3]).max() < POS_TOL
    assert cv.rotation_angle_deg(a[:, :3, :3], ref[:, :3, :3]).max() < DEG_TOL


def test_calibration_is_placement_invariant(profile):
    s = SyntheticRig(profile, seed=4)
    moved = tr(rotation_about([0, 0, 1], 73.0), [3.0, -2.0, 0.5])  # Blender world yaw + move
    s.armature_world = moved @ s.armature_world
    cal = rig.calibrate(s.snapshot(), profile)
    assert cal.max_residual < 1e-7
    for k, name in enumerate(profile.bone_names):
        assert np.allclose(cal.offsets[k], s.offsets[name], atol=1e-7)


def test_calibration_rejects_wrong_pose(profile):
    s = SyntheticRig(profile, seed=5)
    snap = s.snapshot()
    i = snap.index("LeftHand")
    snap.pose_matrices[i, :3, 3] += 5.0  # 5 cm in armature units (scale 0.01)
    with pytest.raises(ValueError, match="does not match the model reference pose"):
        rig.calibrate(snap, profile)


def test_calibration_reports_scale_mismatch(profile):
    s = SyntheticRig(profile, seed=6)
    s.meters_per_unit = 1.0
    snap = s.snapshot()
    snap.meters_per_unit = 0.5  # wrong unit assumption halves every length
    with pytest.raises(ValueError, match="scale mismatch"):
        rig.calibrate(snap, profile)


def test_calibration_json_roundtrip(profile):
    s = SyntheticRig(profile, seed=7)
    cal = rig.calibrate(s.snapshot(), profile)
    again = rig.Calibration.from_json(cal.to_json())
    assert np.allclose(again.offsets, cal.offsets)
    assert again.blender_bone_names == cal.blender_bone_names
    assert set(again.reference_basis) == set(cal.reference_basis)


@pytest.mark.parametrize("orthonormal", [True, False])
def test_model_to_blender_to_model_roundtrip(profile, orthonormal):
    s = SyntheticRig(profile, seed=8)
    cal = rig.calibrate(s.snapshot(), profile)
    poses = random_model_poses(profile, 12, seed=9, orthonormal=orthonormal)
    basis, pose = cal.to_blender_basis(s.snapshot(), poses, rig.LOCATION_ALL)
    # Evaluate the basis with Blender's own FK rule, independently of the middleware.
    for f in range(poses.shape[0]):
        fk = s.fk(basis[f])
        assert np.allclose(fk, pose[f], atol=1e-9)
        back = cal.to_ai4a(s.snapshot(), fk)
        expected = cv.orthonormalize_transforms(poses[f])
        assert np.abs(back[:, :3, 3] - expected[:, :3, 3]).max() < 1e-8
        assert cv.rotation_angle_deg(back[:, :3, :3], expected[:, :3, :3]).max() < 1e-5


def test_blender_to_model_to_blender_roundtrip(profile):
    s = SyntheticRig(profile, seed=10)
    cal = rig.calibrate(s.snapshot(), profile)
    rng = np.random.default_rng(11)
    basis = s.reference_basis.copy()
    for i, name in enumerate(s.bone_names):
        if name in profile.bone_names:
            basis[i] = basis[i] @ tr(rotation_about(rng.normal(size=3), rng.uniform(-60, 60)), rng.normal(scale=2, size=3))
    pose = s.fk(basis)
    a = cal.to_ai4a(s.snapshot(), pose)
    basis_back, pose_back = cal.to_blender_basis(s.snapshot(), a, rig.LOCATION_ALL)
    assert np.allclose(pose_back, pose, atol=1e-9)
    assert np.allclose(basis_back, basis, atol=1e-9)


def test_unmapped_bones_keep_reference_basis(profile):
    s = SyntheticRig(profile, seed=12)
    cal = rig.calibrate(s.snapshot(), profile)
    poses = random_model_poses(profile, 3, seed=13)
    basis, _ = cal.to_blender_basis(s.snapshot(), poses)
    for name in ("Root", "Neck1", "LeftHandIndex1", "HeadTop_End"):
        i = s.bone_names.index(name)
        assert np.allclose(basis[:, i], s.reference_basis[i])


def test_rotation_only_mode(profile):
    s = SyntheticRig(profile, seed=14)
    cal = rig.calibrate(s.snapshot(), profile)
    poses = random_model_poses(profile, 4, seed=15)
    basis, pose = cal.to_blender_basis(s.snapshot(), poses, rig.LOCATION_ROOT)
    for f in range(poses.shape[0]):
        fk = s.fk(basis[f])
        back = cal.to_ai4a(s.snapshot(), fk)
        expected = cv.orthonormalize_transforms(poses[f])
        assert cv.rotation_angle_deg(back[:, :3, :3], expected[:, :3, :3]).max() < 1e-5
        hips = profile.bone_names.index("Hips")
        assert np.allclose(back[hips, :3, 3], expected[hips, :3, 3], atol=1e-8)
        for name in profile.bone_names:
            i = s.bone_names.index(name)
            if name != "Hips":
                assert np.allclose(basis[f, i, :3, 3], s.reference_basis[i][:3, 3])


def test_single_frame_input_shape(profile):
    s = SyntheticRig(profile, seed=16)
    cal = rig.calibrate(s.snapshot(), profile)
    basis, pose = cal.to_blender_basis(s.snapshot(), cv.orthonormalize_transforms(profile.reference_transforms))
    assert basis.shape == (len(s.bone_names), 4, 4)
    assert np.allclose(pose, s.reference_pose, atol=1e-8)
    assert np.allclose(basis, s.reference_basis, atol=1e-8)


# ----------------------------------------------------------------------------
# Features
# ----------------------------------------------------------------------------


def test_root_of_reference_pose_faces_forward(profile):
    ref = profile.reference_transforms
    root = features.compute_root(ref, profile)
    hips = ref[profile.bone_names.index("Hips")]
    assert np.allclose(root[:3, 3], [hips[0, 3], 0.0, hips[2, 3]])
    assert root[:3, 2] @ np.array([0, 0, 1.0]) > 0.99  # Geno faces +Z
    assert np.allclose(root[:3, 1], [0, 1, 0])


def test_root_follows_rigid_motion(profile):
    ref = profile.reference_transforms
    move = tr(rotation_about([0, 1, 0], 120.0), [2.0, 0.0, -1.0])
    moved = np.einsum("ij,bjk->bik", move, ref)
    assert np.allclose(features.compute_root(moved, profile), move @ features.compute_root(ref, profile), atol=1e-9)


def test_guidance_is_root_relative(profile):
    ref = profile.reference_transforms
    move = tr(rotation_about([0, 1, 0], -35.0), [4.0, 0.0, 1.0])
    g0 = features.guidance_from_pose(ref, profile)
    g1 = features.guidance_from_pose(np.einsum("ij,bjk->bik", move, ref), profile)
    assert g0.shape == (profile.bone_count, 3)
    assert np.allclose(g0, g1, atol=1e-9)
    assert np.isclose(g0[profile.bone_names.index("Hips"), 0], 0.0, atol=1e-9)


def test_look_planar_handles_vertical_direction():
    r = features.look_planar([0.0, 1.0, 0.0])
    assert np.allclose(r, np.eye(3))
    root = features.root_from_position_direction([1.0, 0.5, 2.0], [0.0, 0.0, 0.0])
    assert np.isclose(np.linalg.det(root[:3, :3]), 1.0)


def test_reroot_moves_pose_to_start(profile):
    ref = profile.reference_transforms
    start = features.root_from_position_direction([-4.0, 0.0, -4.0], [1.0, 0.0, 1.0])
    moved = features.reroot(ref, features.compute_root(ref, profile), start)
    assert np.allclose(features.compute_root(moved, profile), start, atol=1e-9)


def test_blender_curve_conversion():
    pts = np.array([[0.0, 0.0, 0.0], [0.0, -10.0, 0.0]])  # 10 BU along Blender forward (-Y)
    out = features.blender_path_to_ai4a(pts, meters_per_unit=0.1, spacing=0.25)
    assert np.allclose(out[0], 0.0)
    assert np.allclose(out[-1], [0.0, 0.0, 1.0])
    assert np.allclose(np.linalg.norm(np.diff(out, axis=0), axis=-1), 0.25)


def test_boxes_to_axis_aligned(profile):
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-2, 2) for z in (0, 1)], float)
    center, size = features.blender_boxes_to_ai4a(corners[None])
    assert np.allclose(center, [[0, 0.5, 0]])
    assert np.allclose(size, [[2, 1, 4]])


def test_velocities():
    a = np.repeat(np.eye(4)[None], 2, 0)
    b = a.copy()
    b[:, :3, 3] = [1.0, 0.0, 0.0]
    assert np.allclose(features.velocities(a, b, 0.5), [[2.0, 0, 0]] * 2)


# ----------------------------------------------------------------------------
# Exchange files
# ----------------------------------------------------------------------------


def _meta(profile, frames=5):
    return {
        "fps": 24.0,
        "frame_count": frames,
        "bone_names": profile.bone_names,
        "path_mode": exchange.PATH_CURVE,
        "style_names": ["Neutral", "custom:0"],
        "idle_style": "Idle",
    }


def test_exchange_request_roundtrip(tmp_path, profile):
    path = tmp_path / "req.npz"
    guid = np.zeros((1, profile.bone_count, 3))
    exchange.save_request(
        path,
        _meta(profile),
        speeds=np.ones(5),
        style_indices=np.array([0, 0, 1, 1, 1]),
        path_points=np.zeros((3, 3)),
        custom_guidances=guid,
    )
    meta, arrays = exchange.load_request(path)
    assert meta["version"] == exchange.VERSION
    assert arrays["style_indices"].tolist() == [0, 0, 1, 1, 1]


def test_exchange_rejects_bad_requests(tmp_path, profile):
    with pytest.raises(ValueError, match="custom_guidances"):
        exchange.save_request(
            tmp_path / "a.npz", _meta(profile), speeds=np.ones(5), style_indices=np.zeros(5, int),
            path_points=np.zeros((3, 3)),
        )
    with pytest.raises(ValueError, match="NaN"):
        exchange.save_request(
            tmp_path / "b.npz", dict(_meta(profile), style_names=["Neutral"]),
            speeds=np.r_[np.ones(4), np.nan], style_indices=np.zeros(5, int), path_points=np.zeros((3, 3)),
        )
    with pytest.raises(ValueError, match="TARGET"):
        exchange.save_request(
            tmp_path / "c.npz", dict(_meta(profile), style_names=["Neutral"], path_mode="TARGET"),
            speeds=np.ones(5), style_indices=np.zeros(5, int),
        )


# ----------------------------------------------------------------------------
# Virtual joystick
# ----------------------------------------------------------------------------


def test_stick_up_matches_gamepad_demo_convention():
    # Gamepad demo: stick up -> velocity AI4A -Z, stick right -> AI4A +X.
    # Default gate (identity) in Blender: stick up = knob at +Y.
    gate = np.eye(4)[None]
    up = features.stick_from_knob(gate, [[0.0, 1.0, 0.0]])
    right = features.stick_from_knob(gate, [[1.0, 0.0, 0.0]])
    assert np.allclose(up, [[0.0, 0.0, -1.0]])
    assert np.allclose(right, [[1.0, 0.0, 0.0]])


def test_stick_clamps_uses_gate_frame_and_ignores_height():
    gate = tr(rotation_about([0, 0, 1], 90.0), [5.0, 5.0, 2.0])
    gate[:3, :3] *= 2.0  # gate scaled x2: radius 2 BU == full stick
    knob = [[5.0 - 1.0, 5.0, 2.3]]  # 1 BU along gate +Y (rotated to world -X), 0.3 up
    s = features.stick_from_knob(gate[None], knob)
    assert np.allclose(np.linalg.norm(s), 0.5)
    assert np.allclose(s[0] / 0.5, cv.directions_blender_to_ai4a([-1.0, 0.0, 0.0]))
    far = features.stick_from_knob(np.eye(4)[None], [[3.0, 4.0, 0.0]])
    assert np.isclose(np.linalg.norm(far), 1.0)
    assert np.allclose(features.stick_from_knob(np.eye(4)[None], [[0.0, 0.0, 0.7]]), 0.0)


def test_facing_from_objects_uses_minus_y():
    m = tr(rotation_about([0, 0, 1], 90.0), [1.0, 2.0, 3.0])  # -Y rotated to Blender +X
    d = features.facing_from_objects(m[None])
    assert np.allclose(d, [[1.0, 0.0, 0.0]])
    assert np.allclose(features.facing_from_objects(np.eye(4)[None]), [[0.0, 0.0, 1.0]])


def test_exchange_stick_requests(tmp_path, profile):
    base = dict(_meta(profile), style_names=["Neutral"], controller=exchange.CONTROLLER_STICK)
    common = dict(speeds=np.ones(5), style_indices=np.zeros(5, int))
    exchange.save_request(
        tmp_path / "ok.npz", dict(base, path_mode=exchange.PATH_STICK, facing_mode=exchange.FACING_DIRECTION),
        move_sticks=np.zeros((5, 3)), facing_directions=np.zeros((5, 3)), start_transform=np.eye(4), **common,
    )
    with pytest.raises(ValueError, match="length <= 1"):
        exchange.save_request(
            tmp_path / "a.npz", dict(base, path_mode=exchange.PATH_STICK),
            move_sticks=np.full((5, 3), 2.0), **common,
        )
    with pytest.raises(ValueError, match="needs controller STICK"):
        exchange.save_request(
            tmp_path / "b.npz", dict(base, controller=exchange.CONTROLLER_GOAL, path_mode=exchange.PATH_STICK),
            move_sticks=np.zeros((5, 3)), **common,
        )
    with pytest.raises(ValueError, match="facing_points"):
        exchange.save_request(
            tmp_path / "c.npz", dict(base, facing_mode=exchange.FACING_LOOK_AT),
            path_points=np.zeros((3, 3)), **common,
        )
