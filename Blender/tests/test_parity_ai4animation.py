# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Parity tests: middleware formulas vs. the real AI4Animation code.

Run in the AI4Animation environment (torch + ai4animation importable):
    python -m pytest Blender/tests/test_parity_ai4animation.py
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
ai4a = pytest.importorskip("ai4animation")

from ai4animation_blender.middleware import conventions as cv  # noqa: E402
from ai4animation_blender.middleware import exchange, features  # noqa: E402
from ai4animation_blender.middleware.rig import RigProfile  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
RUNNER = REPO / "Blender" / "ai4animation_blender" / "runner" / "ai4a_runner.py"
MOTION = REPO / "Demos" / "_ASSETS_" / "Geno" / "Motions" / "walk3_subject3.npz"
sys.path.insert(0, str(REPO / "Demos" / "_ASSETS_" / "Geno"))


@pytest.fixture(scope="module")
def motion(profile):
    from ai4animation import GuidanceModule, Motion, MotionModule, RootModule

    import Definitions

    m = Motion.LoadFromNPZ(str(MOTION))
    m.AddModules(
        [
            lambda x: RootModule(
                x,
                Definitions.HipName,
                Definitions.LeftHipName,
                Definitions.RightHipName,
                Definitions.LeftShoulderName,
                Definitions.RightShoulderName,
                Definitions.NeckName,
            ),
            lambda x: MotionModule(x),
            lambda x: GuidanceModule(x),
        ]
    )
    return m


def test_root_matches_rootmodule(motion, profile):
    from ai4animation import RootModule

    timestamps = np.linspace(0.0, motion.TotalTime, 40)
    expected = motion.GetModule(RootModule).GetTransforms(timestamps, False)
    pose = motion.GetBoneTransformations(timestamps, profile.bone_names)
    ours = features.compute_root(pose, profile)
    assert np.abs(ours[..., :3, 3] - expected[..., :3, 3]).max() < 1e-5
    assert cv.rotation_angle_deg(ours[..., :3, :3], expected[..., :3, :3]).max() < 1e-2


def test_guidance_matches_guidancemodule(motion, profile):
    from ai4animation import GuidanceModule, TimeSeries

    t = np.array([1.0, 2.5])
    # A 2-sample window of ~zero width averages the same frame twice.
    smoothing = TimeSeries(0.0, 1e-6, 2)
    expected = motion.GetModule(GuidanceModule).GetLegacyGuidance(t, False, smoothing, profile.bone_names)
    pose = motion.GetBoneTransformations(t, profile.bone_names)
    for k in range(len(t)):
        ours = features.guidance_from_pose(pose[k], profile)
        assert np.abs(ours - expected[k]).max() < 1e-5


def test_quaternion_layout_matches_ai4animation():
    from ai4animation import Quaternion

    rng = np.random.default_rng(0)
    q = rng.normal(size=(50, 4))
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    m = cv.matrices_from_quaternions(q)  # (w, x, y, z)
    xyzw = Quaternion.FromMatrix(m.astype(np.float64))
    ours = cv.quaternions_from_matrices(m)
    same = np.abs(np.abs(np.sum(ours[:, [1, 2, 3, 0]] * xyzw, axis=-1)) - 1.0)
    assert same.max() < 1e-9
    assert np.allclose(Quaternion.ToMatrix(xyzw), m, atol=1e-9)


def test_shipped_guidances_use_profile_bone_order(profile):
    for path in sorted((REPO / "Demos" / "Authoring" / "Guidances").glob("*.npz")):
        with np.load(path, allow_pickle=True) as data:
            assert list(data["Names"]) == profile.bone_names, path.name
            assert data["Positions"].shape == (profile.bone_count, 3)


def test_shipped_profile_matches_runner(tmp_path, profile):
    out = tmp_path / "profile.json"
    subprocess.run(
        [sys.executable, str(RUNNER), "profile", "--repo", str(REPO), "--out", str(out)],
        check=True,
        capture_output=True,
    )
    fresh = RigProfile.from_json(out.read_text())
    assert fresh.bone_names == profile.bone_names
    assert fresh.parent_names == profile.parent_names
    assert fresh.guidance_names == profile.guidance_names
    assert np.allclose(fresh.reference_transforms, profile.reference_transforms, atol=1e-6)


def _request(path, profile, frames, **overrides):
    meta = {
        "fps": 24.0,
        "frame_count": frames,
        "bone_names": profile.bone_names,
        "path_mode": exchange.PATH_CURVE,
        "control_strength": 2.0,
        "prediction_fps": 10.0,
        "end_behavior": exchange.END_STOP,
        "style_names": ["Neutral"],
        "idle_style": "Idle",
    }
    arrays = {
        "speeds": np.full(frames, 1.0),
        "style_indices": np.zeros(frames, int),
        "path_points": np.array([[-1.0, 0.0, -1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 1.0]]),
    }
    meta.update(overrides.pop("meta", {}))
    arrays.update(overrides)
    exchange.save_request(path, meta, **arrays)


def _run(tmp_path, request):
    result = tmp_path / "result.npz"
    subprocess.run(
        [sys.executable, str(RUNNER), "run", "--repo", str(REPO), "--request", str(request), "--result", str(result)],
        check=True,
        capture_output=True,
    )
    return exchange.load_result(result)


def test_runner_starts_at_path_start_without_teleport(tmp_path, profile):
    req = tmp_path / "req.npz"
    _request(req, profile, 72)
    meta, res = _run(tmp_path, req)
    roots = res["roots"]
    assert np.allclose(roots[0, :3, 3], [-1.0, 0.0, -1.0], atol=1e-5)
    assert roots[0, :3, 2] @ (np.array([1.0, 0, 1.0]) / np.sqrt(2)) > 0.99
    hips = res["transforms"][:, 0, :3, 3]
    steps = np.linalg.norm(np.diff(hips, axis=0), axis=-1)
    assert steps.max() < 0.15  # < 3.6 m/s at 24 fps, no teleport
    assert np.all((res["contacts"] >= 0) & (res["contacts"] <= 1))
    final = roots[-1, :3, 3]
    assert np.linalg.norm(final - np.array([1.0, 0.0, 1.0])) < 0.35  # stopped near the goal


def test_runner_uses_initial_pose(tmp_path, profile):
    req = tmp_path / "req.npz"
    ref = np.asarray(profile.reference_transforms, float)
    start = features.root_from_position_direction([2.0, 0.0, 0.0], [0.0, 0.0, -1.0])
    pose = features.reroot(ref, features.compute_root(ref, profile), start)
    _request(
        req, profile, 10,
        initial_transforms=pose,
        initial_velocities=np.zeros((profile.bone_count, 3)),
        initial_root=start,
    )
    meta, res = _run(tmp_path, req)
    assert np.allclose(res["transforms"][0], pose, atol=1e-5)
    assert np.allclose(res["roots"][0], start, atol=1e-5)


def test_runner_target_mode(tmp_path, profile):
    frames = 48
    goals = np.repeat(np.eye(4)[None], frames, 0)
    goals[:, 0, 3] = np.linspace(0.0, 2.0, frames)
    req = tmp_path / "req.npz"
    _request(req, profile, frames, meta={"path_mode": exchange.PATH_TARGET}, goals=goals)
    meta, res = _run(tmp_path, req)
    assert res["transforms"].shape == (frames, profile.bone_count, 4, 4)
    assert res["roots"][-1, 0, 3] > 0.5  # moved towards +X
