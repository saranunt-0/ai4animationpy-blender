# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Request/Result files exchanged between Blender and the AI4Animation runner.

Both files are .npz archives. Every array is in AI4Animation world space
(Y-up, +Z forward, meters, seconds); Blender-specific conventions never cross
this boundary. A JSON "meta" entry carries scalars and names.

Request (Blender -> runner)
    meta.version, fps, frame_count, bone_names, path_mode, control_strength,
    prediction_fps, end_behavior, style_names, idle_style, network_iterations,
    planner {center, size, resolution, max_depth}
    speeds              (F,)        walk speed per frame, m/s
    style_indices       (F,)        index into style_names per frame
    path_points         (N, 3)      path_mode == "CURVE"
    start, goal         (3,)        path_mode == "PLANNER"
    obstacle_centers    (M, 3)      path_mode == "PLANNER"
    obstacle_sizes      (M, 3)
    goals               (F, 4, 4)   path_mode == "TARGET"
    initial_transforms  (J, 4, 4)   optional start pose
    initial_velocities  (J, 3)      optional
    initial_root        (4, 4)      optional
    custom_guidances    (K, J, 3)   optional, referenced as "custom:<i>" in style_names

Result (runner -> Blender)
    meta.version, fps, frame_count, bone_names, warnings
    transforms          (F, J, 4, 4)
    velocities          (F, J, 3)
    roots               (F, 4, 4)
    goals               (F, 4, 4)
    contacts            (F, C)      contact bones in meta.contact_bones
    path_points         (N, 3)      the path actually followed
"""

import json

import numpy as np

VERSION = 1

PATH_CURVE = "CURVE"
PATH_PLANNER = "PLANNER"
PATH_TARGET = "TARGET"
PATH_MODES = (PATH_CURVE, PATH_PLANNER, PATH_TARGET)

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


def validate_request(meta, arrays):
    errors = []
    frames = int(meta.get("frame_count", 0))
    joints = len(meta.get("bone_names", []))
    if frames < 2:
        errors.append("frame_count must be >= 2")
    if meta.get("fps", 0) <= 0:
        errors.append("fps must be > 0")
    mode = meta.get("path_mode")
    if mode not in PATH_MODES:
        errors.append("path_mode must be one of %s" % (PATH_MODES,))
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
