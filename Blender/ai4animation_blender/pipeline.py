# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Blender scene -> request, runner process, result -> Blender.

Glue between blender_io (bpy access) and the middleware (maths). Kept apart
from operators so scripts and tests can drive the whole pipeline.
"""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from . import blender_io as bio
from . import preferences
from .middleware import conventions as cv
from .middleware import exchange, features, rig
from .properties import get_profile

RUNNER = Path(__file__).resolve().parent / "runner" / "ai4a_runner.py"


class PipelineError(RuntimeError):
    pass


# ----------------------------------------------------------------------------
# Runner process
# ----------------------------------------------------------------------------


def runner_environment():
    env = dict(os.environ)
    # Blender's own Python settings must not leak into the model environment.
    for key in ("PYTHONHOME", "PYTHONPATH", "PYTHONNOUSERSITE"):
        env.pop(key, None)
    env["PYTHONUNBUFFERED"] = "1"
    return env


def runner_command(context, *args):
    prefs = preferences.get(context)
    python = bpy_path(prefs.python_path)
    repo = bpy_path(prefs.repo_path)
    if not python or not Path(python).is_file():
        raise PipelineError("Set 'Model Python' in the add-on preferences (the Python with torch + ai4animation).")
    if not repo or not (Path(repo) / "ai4animation").is_dir():
        raise PipelineError("Set 'AI4AnimationPy Repository' in the add-on preferences.")
    return [python, str(RUNNER), args[0], "--repo", repo, *args[1:]]


def bpy_path(path):
    if not path:
        return ""
    import bpy

    return os.path.normpath(bpy.path.abspath(path))


def start_process(command):
    flags = 0
    if sys.platform.startswith("win"):
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=runner_environment(),
        creationflags=flags,
    )


def run_blocking(command, timeout):
    proc = subprocess.run(
        command, capture_output=True, text=True, env=runner_environment(), timeout=timeout
    )
    if proc.returncode != 0:
        raise PipelineError("Runner failed:\n" + tail(proc.stdout + proc.stderr))
    return proc.stdout


def tail(text, lines=15):
    return "\n".join(text.strip().splitlines()[-lines:])


def refresh_profile(context, timeout=600):
    work = tempfile.mkdtemp(prefix="ai4a_")
    try:
        out = Path(work) / "profile.json"
        run_blocking(runner_command(context, "profile", "--out", str(out)), timeout)
        context.scene.ai4a.profile_json = out.read_text()
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return get_profile(context.scene)


# ----------------------------------------------------------------------------
# Request
# ----------------------------------------------------------------------------


def frame_numbers(settings, scene):
    if settings.use_scene_range:
        start, end = scene.frame_start, scene.frame_end
    else:
        start, end = settings.frame_start, settings.frame_end
    if end <= start:
        raise PipelineError("Frame range must contain at least 2 frames.")
    return np.arange(start, end + 1)


def require_calibration(settings):
    arm = settings.armature
    if arm is None:
        raise PipelineError("Pick the armature (or use 'Import Geno Rig').")
    calibration = bio.load_calibration(arm)
    if calibration is None:
        raise PipelineError("Armature is not calibrated. Use 'Calibrate' right after importing the rig.")
    return arm, calibration


def style_timeline(settings, frames):
    """Style names used + per-frame index, from the base style and style keys."""
    keys = sorted(((k.frame, k.style) for k in settings.style_keys), key=lambda x: x[0])
    names = [settings.style]
    for _, style in keys:
        if style not in names:
            names.append(style)
    idle = settings.idle_style or "Idle"
    if idle.startswith(exchange.CUSTOM_PREFIX) and idle not in names:
        names.append(idle)
    indices = np.zeros(len(frames), dtype=np.int64)
    for frame, style in keys:
        indices[frames >= frame] = names.index(style)
    # Replace "custom:<name>" by "custom:<i>" into the custom_guidances array.
    customs, remap = [], {}
    by_name = {c.name: c for c in settings.custom_styles}
    for name in names + [idle]:
        if name.startswith(exchange.CUSTOM_PREFIX) and name not in remap:
            label = name[len(exchange.CUSTOM_PREFIX):]
            if label not in by_name:
                raise PipelineError("Custom style %r no longer exists." % label)
            remap[name] = exchange.CUSTOM_PREFIX + str(len(customs))
            customs.append(np.array(by_name[label].positions(), dtype=float))
    names = [remap.get(n, n) for n in names]
    idle = remap.get(idle, idle)
    return names, idle, indices, (np.stack(customs) if customs else None)


def build_request(context):
    scene = context.scene
    settings = scene.ai4a
    profile = get_profile(scene)
    arm, calibration = require_calibration(settings)
    if calibration.model_bone_names != profile.bone_names:
        raise PipelineError("Calibration was made for a different rig profile. Calibrate again.")
    frames = frame_numbers(settings, scene)
    fps = scene.render.fps / scene.render.fps_base
    mpu = bio.meters_per_unit(scene)

    style_names, idle, style_indices, customs = style_timeline(settings, frames)
    meta = {
        "fps": fps,
        "frame_count": int(len(frames)),
        "bone_names": profile.bone_names,
        "path_mode": settings.path_mode,
        "control_strength": settings.control_strength,
        "prediction_fps": settings.prediction_fps,
        "end_behavior": settings.end_behavior,
        "style_names": style_names,
        "idle_style": idle,
        "network_iterations": settings.network_iterations,
    }
    arrays = {
        "speeds": bio.sample_property(settings, "walk_speed", frames, id_data=scene),
        "style_indices": style_indices,
        "custom_guidances": customs,
    }

    mode = settings.path_mode
    if mode == exchange.PATH_CURVE:
        if settings.curve is None:
            raise PipelineError("Pick a curve object for the path.")
        pts = bio.curve_world_points(settings.curve)
        arrays["path_points"] = features.blender_path_to_ai4a(pts, mpu, spacing=settings.path_spacing)
    elif mode == exchange.PATH_PLANNER:
        if settings.start_object is None or settings.goal_object is None:
            raise PipelineError("Pick Start and Goal objects for the planner.")
        start = cv.points_blender_to_ai4a(np.array(settings.start_object.matrix_world.translation), mpu)
        goal = cv.points_blender_to_ai4a(np.array(settings.goal_object.matrix_world.translation), mpu)
        boxes = [
            bio.object_world_corners(o)
            for o in (settings.obstacles.all_objects if settings.obstacles else [])
            if o.type == "MESH"
        ]
        if boxes:
            centers, sizes = features.blender_boxes_to_ai4a(np.stack(boxes), mpu)
        else:
            centers, sizes = np.zeros((0, 3)), np.zeros((0, 3))
        center, size, resolution = features.auto_planner_grid(
            np.stack([start, goal]), centers, sizes,
            cell=settings.planner_cell * mpu, margin=settings.planner_margin * mpu,
            height=settings.planner_height * mpu,
        )
        meta["planner"] = {
            "center": center.tolist(), "size": size.tolist(), "resolution": resolution,
            "max_depth": settings.planner_max_depth,
        }
        arrays.update(start=start, goal=goal, obstacle_centers=centers, obstacle_sizes=sizes)
    else:
        if settings.target_object is None:
            raise PipelineError("Pick a target object to follow.")
        mats = bio.sample_world_matrices(scene, settings.target_object, frames)
        arrays["goals"] = cv.object_frames_blender_to_ai4a(mats, mpu)

    if settings.start_from_pose:
        current = scene.frame_current
        try:
            scene.frame_set(int(frames[0]) - 1)
            before = calibration.to_ai4a(bio.snapshot_armature(arm, scene))
            scene.frame_set(int(frames[0]))
            pose = calibration.to_ai4a(bio.snapshot_armature(arm, scene))
        finally:
            scene.frame_set(current)
        arrays["initial_transforms"] = pose
        arrays["initial_velocities"] = features.velocities(before, pose, 1.0 / fps)
        arrays["initial_root"] = features.compute_root(pose, profile)

    return meta, arrays, frames


def write_request(context, work_dir):
    meta, arrays, frames = build_request(context)
    path = Path(work_dir) / "request.npz"
    exchange.save_request(path, meta, **arrays)
    return path, frames


# ----------------------------------------------------------------------------
# Result
# ----------------------------------------------------------------------------


def apply_result(context, result_path, frames):
    scene = context.scene
    settings = scene.ai4a
    arm, calibration = require_calibration(settings)
    meta, res = exchange.load_result(result_path)
    if len(frames) != meta["frame_count"]:
        raise PipelineError("Result has %d frames, expected %d." % (meta["frame_count"], len(frames)))
    mpu = bio.meters_per_unit(scene)

    current = scene.frame_current
    scene.frame_set(int(frames[0]))
    snapshot = bio.snapshot_armature(arm, scene)  # rest + armature placement
    scene.frame_set(current)

    basis, _ = calibration.to_blender_basis(snapshot, res["transforms"], settings.location_mode)
    mapped = set(calibration.blender_bone_names)
    constant = [n for n in snapshot.bone_names if n not in mapped]
    action = bio.bake_basis(arm, basis, frames, snapshot.bone_names, settings.action_name, constant_bones=constant)
    if settings.store_contacts and "contacts" in res:
        bio.store_contacts(arm, meta.get("contact_bones", []), res["contacts"], frames)

    if settings.create_helpers:
        coll = bio.helper_collection(scene)
        bio.create_polyline(scene, "AI4A_Path", cv.points_ai4a_to_blender(res["path_points"], mpu), coll)
        bio.animate_empty(scene, "AI4A_Goal", cv.object_frames_ai4a_to_blender(res["goals"], mpu), frames, coll, "SINGLE_ARROW")
        bio.animate_empty(scene, "AI4A_Root", cv.object_frames_ai4a_to_blender(res["roots"], mpu), frames, coll, "ARROWS")

    warnings = list(meta.get("warnings", []))
    warnings += bio.check_bone_setup(arm, calibration)
    if settings.location_mode == rig.LOCATION_ALL:
        connected = bio.connected_bones(arm, calibration)
        if connected:
            warnings.append("Connected bones ignore location keys: %s" % ", ".join(connected[:5]))
    pos_err, rot_err = bio.verify_bake(scene, arm, calibration, res["transforms"], frames)
    return {
        "action": action.name,
        "frames": len(frames),
        "position_error": pos_err,
        "rotation_error": rot_err,
        "warnings": warnings,
        "seconds": meta.get("seconds", 0.0),
    }


def summary(report, location_mode):
    tolerance = 1e-3 if location_mode == rig.LOCATION_ALL else float("inf")
    ok = report["position_error"] < tolerance and report["rotation_error"] < 0.01
    text = "Baked %d frames into '%s' (model %.1fs). Check vs model: %.2f mm, %.4f deg" % (
        report["frames"], report["action"], report["seconds"],
        1000 * report["position_error"], report["rotation_error"],
    )
    return ok, text


def generate_blocking(context):
    prefs = preferences.get(context)
    work = tempfile.mkdtemp(prefix="ai4a_")
    try:
        request, frames = write_request(context, work)
        result = Path(work) / "result.npz"
        run_blocking(
            runner_command(context, "run", "--request", str(request), "--result", str(result)), prefs.timeout
        )
        return apply_result(context, result, frames)
    finally:
        if not prefs.keep_files:
            shutil.rmtree(work, ignore_errors=True)

