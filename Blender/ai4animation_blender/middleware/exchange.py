# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Request/Result files exchanged between Blender and the AI4Animation runner.

Both files are .npz archives. Every array is in AI4Animation world space
(Y-up, +Z forward, meters, seconds); Blender-specific conventions never cross
this boundary. A JSON "meta" entry carries scalars and names.

Request (Blender -> runner)
    meta.version, fps, frame_count, bone_names, model {type, character,
    network, postprocessor}, path_mode, controller,
    control_strength, prediction_fps, end_behavior, style_names, idle_style,
    network_iterations, facing_mode, tracking {gain, leash, assist},
    planner {center, size, resolution, max_depth}
    speeds              (F,)        walk speed per frame, m/s (stick: speed at full deflection)
    style_indices       (F,)        index into style_names per frame (QUADRUPED: Auto/Sit/Stand/Lie)
    path_points         (N, 3)      path_mode == "CURVE"
    start, goal         (3,)        path_mode == "PLANNER"
    obstacle_centers    (M, 3)      path_mode == "PLANNER"
    obstacle_sizes      (M, 3)
    goals               (F, 4, 4)   path_mode == "TARGET"
    move_sticks         (F, 3)      path_mode == "STICK": left stick in world space, |v| <= 1
    start_transform     (4, 4)      path_mode == "STICK": where the character starts
    facing_points       (F, 3)      facing_mode == "LOOK_AT"
    facing_directions   (F, 3)      facing_mode == "DIRECTION" (right stick; zero = movement)
    initial_transforms  (J, 4, 4)   optional start pose
    initial_velocities  (J, 3)      optional
    initial_root        (4, 4)      optional
    custom_guidances    (K, J, 3)   optional, referenced as "custom:<i>" in style_names

Models (middleware.models)
    BIPED      Geno, Demos/Authoring network. Both controllers.
    QUADRUPED  Dog / Wolf, Demos/Locomotion/Quadruped network. STICK only
               (the demo's gamepad control): speed-chosen gait, facing
               follows movement, actions Sit/Stand/Lie.
    network / postprocessor: optional .pt overrides ("" = the demo's files).

Controllers
    GOAL   Demos/Authoring: SimulationObject.ControlFromTarget(goal). Faces
           the movement direction.
    STICK  Demos/Locomotion/Biped (the original gamepad control):
           SimulationObject.Control(position, direction, velocity) with
           velocity = left stick and direction = right stick. A virtual
           player produces the left stick by tracking a reference that moves
           at the keyed speed along a path, follows a target, or integrates
           keyed stick objects; the right stick comes from facing_mode.

Result (runner -> Blender)
    meta.version, fps, frame_count, bone_names, controller, warnings
    transforms          (F, J, 4, 4)
    velocities          (F, J, 3)
    roots               (F, 4, 4)
    goals               (F, 4, 4)   goal (GOAL) or end of the simulated trajectory (STICK)
    contacts            (F, C)      contact bones in meta.contact_bones
    path_points         (N, 3)      the path actually followed
    commands, facings   (F, 3)      velocity command (m/s) and facing used (STICK)
"""

import json

import numpy as np

from . import models

VERSION = 3

PATH_CURVE = "CURVE"
PATH_PLANNER = "PLANNER"
PATH_TARGET = "TARGET"
PATH_STICK = "STICK"
PATH_MODES = (PATH_CURVE, PATH_PLANNER, PATH_TARGET, PATH_STICK)

CONTROLLER_GOAL = "GOAL"
CONTROLLER_STICK = "STICK"
CONTROLLERS = (CONTROLLER_GOAL, CONTROLLER_STICK)

FACING_MOVE = "MOVE"
FACING_LOOK_AT = "LOOK_AT"
FACING_DIRECTION = "DIRECTION"
FACING_MODES = (FACING_MOVE, FACING_LOOK_AT, FACING_DIRECTION)

# Joystick controller's virtual player: position gain (1/s), max distance of the
# reference ahead of the character (m), assist = track a moving reference
# (exact speeds) instead of feeding raw stick velocities.
DEFAULT_TRACKING = {"gain": 2.0, "leash": 1.0, "assist": True}

END_STOP = "STOP"
END_PINGPONG = "PINGPONG"
END_BEHAVIORS = (END_STOP, END_PINGPONG)

CUSTOM_PREFIX = "custom:"

REQUEST_ARRAYS = {
    "speeds": 1,
    "style_indices": 1,
    "path_points": 2,
    "start": 1,
    "goal": 1,
    "obstacle_centers": 2,
    "obstacle_sizes": 2,
    "goals": 3,
    "move_sticks": 2,
    "start_transform": 2,
    "facing_points": 2,
    "facing_directions": 2,
    "initial_transforms": 3,
    "initial_velocities": 2,
    "initial_root": 2,
    "custom_guidances": 3,
}

RESULT_ARRAYS = {
    "transforms": 4,
    "velocities": 3,
    "roots": 3,
    "goals": 3,
    "contacts": 2,
    "path_points": 2,
    "commands": 2,
    "facings": 2,
}


def _save(path, meta, arrays, spec):
    payload = {"meta": np.array(json.dumps(meta))}
    for key, value in arrays.items():
        if value is None:
            continue
        if key not in spec:
            raise KeyError("Unknown exchange array %r" % key)
        value = np.asarray(value)
        if value.ndim != spec[key]:
            raise ValueError("%s must have %d dims, got shape %s" % (key, spec[key], value.shape))
        payload[key] = value
    np.savez_compressed(path, **payload)


def _load(path, spec):
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta"]))
        arrays = {k: data[k].copy() for k in data.files if k != "meta"}
    if meta.get("version") != VERSION:
        raise ValueError("Exchange version %r != %r" % (meta.get("version"), VERSION))
    unknown = set(arrays) - set(spec)
    if unknown:
        raise KeyError("Unknown exchange arrays %s" % sorted(unknown))
    return meta, arrays


def model_meta(meta):
    """meta.model with defaults (requests without it drive the biped)."""
    info = dict(meta.get("model") or {})
    info.setdefault("type", models.BIPED)
    info.setdefault("character", "geno" if info["type"] == models.BIPED else "")
    info.setdefault("network", "")
    info.setdefault("postprocessor", "")
    return info


def validate_request(meta, arrays):
    errors = []
    frames = int(meta.get("frame_count", 0))
    joints = len(meta.get("bone_names", []))
    model = model_meta(meta)
    spec = models.MODELS.get(model["type"])
    if spec is None:
        errors.append("model.type must be one of %s" % sorted(models.MODELS))
    elif model["character"] not in spec.character_keys():
        errors.append("model.character must be one of %s for %s" % (spec.character_keys(), spec.key))
    if frames < 2:
        errors.append("frame_count must be >= 2")
    if meta.get("fps", 0) <= 0:
        errors.append("fps must be > 0")
    mode = meta.get("path_mode")
    if mode not in PATH_MODES:
        errors.append("path_mode must be one of %s" % (PATH_MODES,))
    control = meta.get("controller", CONTROLLER_GOAL)
    if control not in CONTROLLERS:
        errors.append("controller must be one of %s" % (CONTROLLERS,))
    if control == CONTROLLER_GOAL and mode == PATH_STICK:
        errors.append("path_mode STICK needs controller STICK")
    facing = meta.get("facing_mode", FACING_MOVE)
    if facing not in FACING_MODES:
        errors.append("facing_mode must be one of %s" % (FACING_MODES,))
    if facing != FACING_MOVE and control != CONTROLLER_STICK:
        errors.append("facing_mode %s needs controller STICK" % facing)
    for key, needed in (("facing_points", FACING_LOOK_AT), ("facing_directions", FACING_DIRECTION)):
        if facing == needed and (key not in arrays or arrays[key].shape != (frames, 3)):
            errors.append("facing_mode %s needs %s with shape (%d, 3)" % (needed, key, frames))
    if spec is not None and not spec.goal_controller and control != CONTROLLER_STICK:
        errors.append("model %s needs controller STICK" % spec.key)
    if spec is not None and not spec.facing_control and facing != FACING_MOVE:
        errors.append("model %s only faces its movement (facing_mode MOVE)" % spec.key)
    if spec is not None and spec.key == models.QUADRUPED:
        unknown = [n for n in meta.get("style_names", []) if n not in models.QUADRUPED_STYLES]
        if unknown:
            errors.append("QUADRUPED styles must be in %s, got %s" % (models.QUADRUPED_STYLES, unknown))
    tracking = meta.get("tracking", {})
    if float(tracking.get("gain", 1.0)) < 0.0 or float(tracking.get("leash", 1.0)) <= 0.0:
        errors.append("tracking gain must be >= 0 and leash > 0")
    if meta.get("end_behavior", END_STOP) not in END_BEHAVIORS:
        errors.append("end_behavior must be one of %s" % (END_BEHAVIORS,))
    for key in ("speeds", "style_indices"):
        if key not in arrays or arrays[key].shape != (frames,):
            errors.append("%s must have shape (%d,)" % (key, frames))
    styles = meta.get("style_names", [])
    if not styles:
        errors.append("style_names must not be empty")
    elif "style_indices" in arrays and len(arrays["style_indices"]):
        idx = arrays["style_indices"]
        if idx.min() < 0 or idx.max() >= len(styles):
            errors.append("style_indices out of range")
    if mode == PATH_CURVE and ("path_points" not in arrays or arrays["path_points"].shape[0] < 2):
        errors.append("CURVE mode needs path_points with >= 2 points")
    if mode == PATH_PLANNER:
        for key in ("start", "goal"):
            if key not in arrays or arrays[key].shape != (3,):
                errors.append("PLANNER mode needs %s with shape (3,)" % key)
        if "planner" not in meta:
            errors.append("PLANNER mode needs meta.planner")
    if mode == PATH_TARGET and ("goals" not in arrays or arrays["goals"].shape != (frames, 4, 4)):
        errors.append("TARGET mode needs goals with shape (%d, 4, 4)" % frames)
    if mode == PATH_STICK:
        sticks = arrays.get("move_sticks")
        if sticks is None or sticks.shape != (frames, 3):
            errors.append("STICK mode needs move_sticks with shape (%d, 3)" % frames)
        elif np.linalg.norm(sticks, axis=-1).max(initial=0.0) > 1.0 + 1e-6:
            errors.append("move_sticks must have length <= 1")
        if "start_transform" in arrays and arrays["start_transform"].shape != (4, 4):
            errors.append("start_transform must have shape (4, 4)")
    if "initial_transforms" in arrays and arrays["initial_transforms"].shape != (joints, 4, 4):
        errors.append("initial_transforms must have shape (%d, 4, 4)" % joints)
    if "initial_velocities" in arrays and arrays["initial_velocities"].shape != (joints, 3):
        errors.append("initial_velocities must have shape (%d, 3)" % joints)
    for key, value in arrays.items():
        if np.issubdtype(value.dtype, np.floating) and not np.all(np.isfinite(value)):
            errors.append("%s contains NaN/Inf" % key)
    custom = arrays.get("custom_guidances")
    for name in styles:
        if name.startswith(CUSTOM_PREFIX):
            i = int(name[len(CUSTOM_PREFIX):])
            if custom is None or i >= custom.shape[0]:
                errors.append("style %r has no matching custom_guidances entry" % name)
    if custom is not None and custom.shape[1:] != (joints, 3):
        errors.append("custom_guidances must have shape (K, %d, 3)" % joints)
    if errors:
        raise ValueError("Invalid request: " + "; ".join(errors))


def save_request(path, meta, **arrays):
    meta = dict(meta, version=VERSION)
    validate_request(meta, {k: np.asarray(v) for k, v in arrays.items() if v is not None})
    _save(path, meta, arrays, REQUEST_ARRAYS)


def load_request(path):
    meta, arrays = _load(path, REQUEST_ARRAYS)
    validate_request(meta, arrays)
    return meta, arrays


def save_result(path, meta, **arrays):
    meta = dict(meta, version=VERSION)
    _save(path, meta, arrays, RESULT_ARRAYS)


def load_result(path):
    meta, arrays = _load(path, RESULT_ARRAYS)
    frames = int(meta["frame_count"])
    joints = len(meta["bone_names"])
    if arrays["transforms"].shape != (frames, joints, 4, 4):
        raise ValueError("Result transforms shape %s != %s" % (arrays["transforms"].shape, (frames, joints, 4, 4)))
    return meta, arrays
