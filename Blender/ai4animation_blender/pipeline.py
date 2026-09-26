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
from .middleware import exchange, features, models, rig
from .properties import get_profile, is_quadruped, model_spec, selected_character

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


def check_model_python(path, platform=sys.platform):
    """Error message if `path` is not a Python executable, else None.

    A common mistake is picking another file of the model environment (e.g.
    Network.pt), which Windows reports only as "[WinError 193] %1 is not a
    valid Win32 application".
    """
    windows = platform.startswith("win")
    example = r"<venv>\Scripts\python.exe" if windows else "<venv>/bin/python"
    hint = (
        " 'Model Python' must be the Python executable of the environment with torch and "
        "ai4animation, e.g. %s. The network is found through the repository path." % example
    )
    if not path:
        return "Set 'Model Python' in the add-on preferences." + hint
    file = Path(path)
    if not file.is_file():
        return "Model Python not found: %s." % path + hint
    if not file.name.lower().startswith("python") or (windows and file.suffix.lower() != ".exe"):
        return "Model Python is '%s', which is not a Python executable." % file.name + hint
    if not windows and not os.access(file, os.X_OK):
        return "Model Python '%s' is not executable." % path + hint
    return None


def runner_command(context, *args):
    prefs = preferences.get(context)
    python = bpy_path(prefs.python_path)
    repo = bpy_path(prefs.repo_path)
    error = check_model_python(python)
    if error:
        raise PipelineError(error)
    if not repo or not (Path(repo) / "ai4animation").is_dir():
        raise PipelineError("Set 'AI4AnimationPy Repository' in the add-on preferences.")
    return [python, str(RUNNER), args[0], "--repo", repo, *args[1:]]


def bpy_path(path):
    if not path:
        return ""
    import bpy

    return os.path.normpath(bpy.path.abspath(path))


def _creation_flags():
    # No console window flashing up on Windows.
    if sys.platform.startswith("win"):
        return getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return 0


def _start_error(command, error):
    return PipelineError("Could not start Model Python '%s': %s" % (command[0], error))


def start_process(command):
    try:
        return subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=runner_environment(),
            creationflags=_creation_flags(),
        )
    except OSError as error:
        raise _start_error(command, error) from error


def run_blocking(command, timeout):
    try:
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            env=runner_environment(),
            timeout=timeout,
            creationflags=_creation_flags(),
        )
    except OSError as error:
        raise _start_error(command, error) from error
    if proc.returncode != 0:
        raise PipelineError("Runner failed:\n" + tail(proc.stdout + proc.stderr))
    return proc.stdout


def tail(text, lines=15):
    return "\n".join(text.strip().splitlines()[-lines:])


def model_info(settings):
    """meta.model: which network animates which character (overrides as absolute paths)."""
    info = {
        "type": settings.model,
        "character": selected_character(settings).key,
        "network": bpy_path(settings.network_path),
        "postprocessor": bpy_path(settings.postprocessor_path),
    }
    for key in ("network", "postprocessor"):
        if info[key] and not Path(info[key]).is_file():
            raise PipelineError("%s file not found: %s" % (key.capitalize(), info[key]))
    return info


def refresh_profile(context, timeout=600):
    info = model_info(context.scene.ai4a)
    work = tempfile.mkdtemp(prefix="ai4a_")
    try:
        out = Path(work) / "profile.json"
        args = ["profile", "--out", str(out), "--model", info["type"], "--character", info["character"]]
        for key in ("network", "postprocessor"):
            if info[key]:
                args += ["--" + key, info[key]]
        run_blocking(runner_command(context, *args), timeout)
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
    """The armature and its calibration, which must pair it with the selected character."""
    character = selected_character(settings)
    arm = settings.armature
    if arm is None:
        raise PipelineError("Pick the armature (or use 'Import %s')." % character.label)
    calibration = bio.load_calibration(arm)
    if calibration is None:
        raise PipelineError("Armature is not calibrated. Use 'Calibrate' right after importing the rig.")
    if calibration.character != character.key:
        paired = models.character_label(calibration.character)
        raise PipelineError(
            "'%s' is paired with %s, but the selected character is %s. Select %s, or import / "
            "calibrate a %s rig." % (arm.name, paired, character.label, paired, character.label)
        )
    return arm, calibration


def style_timeline(settings, frames):
    """Style names used + per-frame index, from the base style and style keys.

    Quadruped "styles" are the demo's actions (Auto = gait by speed).
    """
    keys = sorted(((k.frame, k.style) for k in settings.style_keys), key=lambda x: x[0])
    names = [settings.style]
    for _, style in keys:
        if style not in names:
            names.append(style)
    if "" in names:  # an enum index left over from another model
        raise PipelineError("A style key has no valid style for this model. Pick one or remove the key.")
    idle = models.QUADRUPED_AUTO if is_quadruped(settings) else (settings.idle_style or "Idle")
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


# Blender facing options -> exchange facing modes
FACING_TO_EXCHANGE = {
    "MOVE": exchange.FACING_MOVE,
    "LOOK_AT": exchange.FACING_LOOK_AT,
    "OBJECT": exchange.FACING_DIRECTION,
    "STICK": exchange.FACING_DIRECTION,
}


def resolve_controller(settings):
    """Pick the controller that serves the request best.

    Measured on S-curve / corner / hairpin paths: with the character facing
    where it walks, the Authoring goal controller follows paths tighter and
    smoother. Facing control (look-at, object, right stick) and keyed stick
    input need the gamepad controller (Demos/Locomotion/Biped).
    """
    if not model_spec(settings).goal_controller:  # the quadruped demo only has stick control
        return exchange.CONTROLLER_STICK
    needs_stick = settings.path_mode == exchange.PATH_STICK or settings.facing_mode != "MOVE"
    if settings.controller == "AUTO":
        return exchange.CONTROLLER_STICK if needs_stick else exchange.CONTROLLER_GOAL
    if settings.controller == exchange.CONTROLLER_GOAL and needs_stick:
        raise PipelineError(
            "The goal controller always faces where it walks and has no joystick input. "
            "Use Controller 'Auto' or 'Joystick'."
        )
    return settings.controller


def facing_mode(settings):
    """Blender facing option in effect (a model without facing control faces its movement)."""
    return settings.facing_mode if model_spec(settings).facing_control else "MOVE"


def knob_and_gate(obj, label):
    if obj is None:
        raise PipelineError("Pick the %s knob (use 'Create Helpers')." % label)
    return obj, obj.parent  # no parent: the world origin is the gate


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
    controller = resolve_controller(settings)
    facing = facing_mode(settings)
    meta = {
        "model": model_info(settings),
        "fps": fps,
        "frame_count": int(len(frames)),
        "bone_names": profile.bone_names,
        "path_mode": settings.path_mode,
        "controller": controller,
        "facing_mode": FACING_TO_EXCHANGE[facing],
        "tracking": {
            "gain": settings.tracking_gain,
            "leash": settings.tracking_leash * mpu,
            "assist": settings.stick_assist,
        },
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
    # Everything animated is sampled in one pass over the frames.
    sampled = {}
    if mode == exchange.PATH_TARGET:
        if settings.target_object is None:
            raise PipelineError("Pick a target object to follow.")
        sampled["target"] = settings.target_object
    if mode == exchange.PATH_STICK:
        sampled["left_knob"], sampled["left_gate"] = knob_and_gate(settings.left_stick, "left stick")
    if facing == "STICK":
        sampled["right_knob"], sampled["right_gate"] = knob_and_gate(settings.right_stick, "right stick")
    elif facing in ("LOOK_AT", "OBJECT"):
        if settings.facing_object is None:
            raise PipelineError("Pick the facing object.")
        sampled["facing"] = settings.facing_object
    keys = list(sampled)
    mats = dict(zip(keys, bio.sample_world_matrices_many(scene, [sampled[k] for k in keys], frames)))

    if mode == exchange.PATH_TARGET:
        arrays["goals"] = cv.object_frames_blender_to_ai4a(mats["target"], mpu)
    if mode == exchange.PATH_STICK:
        arrays["move_sticks"] = features.stick_from_knob(mats["left_gate"], mats["left_knob"][:, :3, 3])
        arrays["start_transform"] = start_transform(scene, settings, arm, calibration, profile, frames[0], mpu)
    if facing == "STICK":
        arrays["facing_directions"] = features.stick_from_knob(mats["right_gate"], mats["right_knob"][:, :3, 3])
    elif facing == "LOOK_AT":
        arrays["facing_points"] = cv.points_blender_to_ai4a(mats["facing"][:, :3, 3], mpu)
    elif facing == "OBJECT":
        arrays["facing_directions"] = features.facing_from_objects(mats["facing"], mpu)

    if settings.start_from_pose:
        current = scene.frame_current
        try:
            scene.frame_set(int(frames[0]) - 1)
            before = armature_pose(arm, scene, calibration, profile)
            scene.frame_set(int(frames[0]))
            pose = armature_pose(arm, scene, calibration, profile)
        finally:
            scene.frame_set(current)
        arrays["initial_transforms"] = pose
        arrays["initial_velocities"] = features.velocities(before, pose, 1.0 / fps)
        arrays["initial_root"] = features.compute_root(pose, profile)

    return meta, arrays, frames


def armature_pose(arm, scene, calibration, profile):
    """Current armature pose as model transforms; bones the armature lacks
    (end sites) are filled from the reference pose."""
    return features.complete_pose(calibration.to_ai4a(bio.snapshot_armature(arm, scene)), profile)


def start_transform(scene, settings, arm, calibration, profile, frame, mpu):
    """Where a joystick-driven character starts: the Start object, else where it stands."""
    if settings.start_object is not None:
        m = bio.sample_world_matrices(scene, settings.start_object, [frame])[0]
        return cv.object_frames_blender_to_ai4a(m, mpu)
    current = scene.frame_current
    try:
        scene.frame_set(int(frame))
        pose = armature_pose(arm, scene, calibration, profile)
    finally:
        scene.frame_set(current)
    return features.compute_root(pose, profile)


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
    mapped = set(calibration.mapped_blender_bones)
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

