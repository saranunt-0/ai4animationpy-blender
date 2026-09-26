# Copyright (c) Meta Platforms, Inc. and affiliates.
"""End-to-end tests inside Blender (bpy) with the real model runner.

Needs the `bpy` module (pip install bpy==4.5.*, Python 3.11) and the model
environment's python in AI4A_PYTHON:

    AI4A_PYTHON=/path/to/model/python python -m pytest Blender/tests/test_blender_e2e.py
"""

import sys
from pathlib import Path

import numpy as np
import pytest

bpy = pytest.importorskip("bpy")

from conftest import model_python  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
ADDON = "ai4animation_blender"

pytestmark = pytest.mark.skipif(not model_python(), reason="AI4A_PYTHON not set")


@pytest.fixture()
def scene():
    import addon_utils

    bpy.ops.wm.read_factory_settings(use_empty=True)
    sys.path.insert(0, str(REPO / "Blender"))
    addon_utils.enable(ADDON, default_set=True)
    prefs = bpy.context.preferences.addons[ADDON].preferences
    prefs.repo_path = str(REPO)
    prefs.python_path = model_python()
    scene = bpy.context.scene
    scene.render.fps = 24
    scene.frame_start, scene.frame_end = 1, 73
    yield scene
    addon_utils.disable(ADDON)


def _check_against_model(scene, report):
    assert report == {"FINISHED"}
    status = scene.ai4a.status
    assert "Baked 73 frames" in status, status


def _import(file_format="GLB"):
    assert bpy.ops.ai4a.import_rig(file_format=file_format) == {"FINISHED"}
    arm = bpy.context.scene.ai4a.armature
    assert arm is not None and arm.get("ai4a_calibration")
    return arm


def _hips_world(arm, frame):
    bpy.context.scene.frame_set(frame)
    return np.array(arm.matrix_world @ arm.pose.bones["Hips"].matrix.translation)


@pytest.mark.parametrize("file_format", ["GLB", "FBX"])
def test_curve_mode(scene, file_format):
    arm = _import(file_format)
    s = scene.ai4a
    s.path_mode = "CURVE"
    assert bpy.ops.ai4a.setup_helpers() == {"FINISHED"}
    s.style = "Neutral"
    s.idle_style = "Idle"
    s.walk_speed = 1.5
    _check_against_model(scene, bpy.ops.ai4a.generate())
    status = s.status
    err_mm = float(status.split("Check vs model: ")[1].split(" mm")[0])
    assert err_mm < 0.1, status
    start, end = _hips_world(arm, 1), _hips_world(arm, 73)
    assert np.linalg.norm(start[:2] - np.array([0.0, 0.0])) < 0.05  # curve starts at the origin
    assert np.linalg.norm(end[:2] - start[:2]) > 1.5  # walked at least 1.5 m in 3 s
    assert bpy.data.objects.get("AI4A_Path") is not None
    assert "ai4a_contact_LeftFoot" in arm.keys()


def test_planner_mode_with_style_keys_and_custom_style(scene):
    _import("GLB")
    s = scene.ai4a
    s.path_mode = "PLANNER"
    assert bpy.ops.ai4a.setup_helpers() == {"FINISHED"}
    s.custom_style_name = "APose"
    assert bpy.ops.ai4a.capture_style() == {"FINISHED"}
    s.style = "Zombie"
    scene.frame_set(30)
    assert bpy.ops.ai4a.style_key_add() == {"FINISHED"}
    s.style_keys[0].style = "custom:APose"
    s.location_mode = "ROOT"
    _check_against_model(scene, bpy.ops.ai4a.generate())


@pytest.mark.parametrize("start_from_pose", [False, True])
def test_target_mode(scene, start_from_pose):
    arm = _import("GLB")
    s = scene.ai4a
    s.path_mode = "TARGET"
    assert bpy.ops.ai4a.setup_helpers() == {"FINISHED"}
    target = s.target_object
    target.location = (0.0, 0.0, 0.0)
    target.keyframe_insert("location", frame=1)
    target.location = (0.0, -3.0, 0.0)  # Blender forward
    target.keyframe_insert("location", frame=73)
    arm.location = (1.0, 0.0, 0.0)  # armature placement must not matter
    s.start_from_pose = start_from_pose
    _check_against_model(scene, bpy.ops.ai4a.generate())
    start, end = _hips_world(arm, 1), _hips_world(arm, 73)
    assert end[1] < start[1] - 1.0  # moved towards -Y (Blender forward)
    if not start_from_pose:
        # starts at the target, facing the target's -Y (Blender forward)
        assert np.linalg.norm(start[:2]) < 0.05
        root = bpy.data.objects["AI4A_Root"]
        scene.frame_set(1)
        forward = -np.array(root.matrix_world.col[1][:3])
        assert forward @ np.array([0.0, -1.0, 0.0]) > 0.99


def _key_knob(knob, frames, xy):
    for frame in frames:
        knob.location = (xy[0], xy[1], 0.0)
        knob.keyframe_insert("location", frame=frame)


def test_virtual_joystick_walks_backwards(scene):
    arm = _import("GLB")
    s = scene.ai4a
    s.path_mode = "STICK"
    s.facing_mode = "STICK"
    assert bpy.ops.ai4a.setup_helpers() == {"FINISHED"}
    assert s.left_stick.parent is not None and s.right_stick.parent is not None
    _key_knob(s.left_stick, (1, 73), (0.0, 1.0))  # stick up = Blender +Y
    _key_knob(s.right_stick, (1, 73), (0.0, -1.0))  # face Blender -Y
    _check_against_model(scene, bpy.ops.ai4a.generate())
    start, end = _hips_world(arm, 25), _hips_world(arm, 73)
    assert end[1] - start[1] > 1.2  # walked towards +Y ...
    root = bpy.data.objects["AI4A_Root"]
    scene.frame_set(60)
    forward = -np.array(root.matrix_world.col[1][:3])
    assert forward @ np.array([0.0, -1.0, 0.0]) > 0.9  # ... while facing -Y: backwards


def test_curve_with_look_at_uses_joystick_controller(scene):
    from ai4animation_blender import pipeline

    _import("GLB")
    s = scene.ai4a
    s.path_mode = "CURVE"
    s.facing_mode = "LOOK_AT"
    assert bpy.ops.ai4a.setup_helpers() == {"FINISHED"}
    assert pipeline.resolve_controller(s) == "STICK"
    s.facing_object.location = (3.0, 3.0, 0.0)
    _check_against_model(scene, bpy.ops.ai4a.generate())
    s.controller = "GOAL"
    with pytest.raises(pipeline.PipelineError):
        pipeline.resolve_controller(s)
    s.controller = "AUTO"
    s.facing_mode = "MOVE"
    assert pipeline.resolve_controller(s) == "GOAL"


def test_default_style_is_neutral(scene):
    assert scene.ai4a.style == "Neutral"
    assert scene.ai4a.idle_style == "Idle"


# ----------------------------------------------------------------------------
# Quadruped (Dog / Wolf)
# ----------------------------------------------------------------------------


def _import_quadruped(scene, character):
    from ai4animation_blender import blender_io as bio

    scene.ai4a.model = "QUADRUPED"
    scene.ai4a.character = character
    arm = _import("GLB")
    calibration = bio.load_calibration(arm)
    assert calibration.character == character
    assert calibration.max_residual < 1e-4
    return arm, calibration


@pytest.mark.parametrize("character", ["dog", "wolf"])
def test_quadruped_curve(scene, character):
    arm, calibration = _import_quadruped(scene, character)
    assert calibration.missing_bones == (["HeadSite"] if character == "dog" else [])
    s = scene.ai4a
    s.path_mode = "CURVE"
    assert bpy.ops.ai4a.setup_helpers() == {"FINISHED"}
    s.walk_speed = 2.0  # trot
    _check_against_model(scene, bpy.ops.ai4a.generate())
    err_mm = float(s.status.split("Check vs model: ")[1].split(" mm")[0])
    assert err_mm < 0.1, s.status
    start, end = _hips_world(arm, 1), _hips_world(arm, 73)
    assert np.linalg.norm(start[:2]) < 0.1  # curve starts at the origin
    assert np.linalg.norm(end[:2] - start[:2]) > 2.5  # trotted in 3 s
    assert "ai4a_contact_LeftFootSite" in arm.keys()


def test_quadruped_sits_on_a_keyed_action(scene):
    arm, _ = _import_quadruped(scene, "dog")
    s = scene.ai4a
    s.path_mode = "STICK"
    assert bpy.ops.ai4a.setup_helpers() == {"FINISHED"}  # knob centered: stand still
    s.style = "Auto"
    scene.frame_set(13)
    assert bpy.ops.ai4a.style_key_add() == {"FINISHED"}
    s.style_keys[0].style = "Sit"
    _check_against_model(scene, bpy.ops.ai4a.generate())
    standing, sitting = _hips_world(arm, 12)[2], _hips_world(arm, 73)[2]
    assert standing - sitting > 0.15, (standing, sitting)


def test_character_pairing_is_checked(scene):
    from ai4animation_blender import pipeline

    _import("GLB")  # Geno
    scene.ai4a.model = "QUADRUPED"
    with pytest.raises(pipeline.PipelineError, match="paired with Geno, but the selected character is Dog"):
        pipeline.require_calibration(scene.ai4a)


def test_refresh_profile_for_the_selected_character(scene):
    from ai4animation_blender.properties import get_profile

    s = scene.ai4a
    s.model = "QUADRUPED"
    s.character = "wolf"
    assert bpy.ops.ai4a.refresh_profile() == {"FINISHED"}
    assert get_profile(scene).character == "wolf"
    s.character = "dog"
    assert get_profile(scene).character == "dog"  # shipped profile, not the Wolf's
