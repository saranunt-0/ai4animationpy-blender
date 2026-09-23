# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Headless AI4Animation runner used by the Blender add-on.

Runs in the AI4Animation Python environment (torch + ai4animation), NOT in
Blender. Blender talks to it through files written by
ai4animation_blender.middleware.exchange, all in AI4Animation world space.

    python ai4a_runner.py profile --repo <ai4animationpy> --out profile.json
    python ai4a_runner.py run     --repo <ai4animationpy> --request req.npz --result res.npz

The simulation reuses Demos/Authoring (MotionController, PathPlanner3D) as-is,
driven in AI4Animation MANUAL mode with a fixed time step, so Blender gets the
same controller behavior as the standalone demo.
"""

import argparse
import contextlib
import functools
import math
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ADDONS_DIR = Path(__file__).resolve().parents[2]
if str(ADDONS_DIR) not in sys.path:
    sys.path.insert(0, str(ADDONS_DIR))

from ai4animation_blender.middleware import exchange, features  # noqa: E402
from ai4animation_blender.middleware.rig import RigProfile  # noqa: E402

AUTHORING_DIR = Path("Demos") / "Authoring"
ASSETS_DIR = Path("Demos") / "_ASSETS_" / "Geno"

# Must match Demos/Authoring/Program.py
SPLINE_RESOLUTION = 80
IDLE_STYLE = "Idle"
# The demo runs at display refresh rate; sub-stepping keeps the controller in
# the regime it was tuned for when Blender scenes run at 24/25/30 fps.
TARGET_SIMULATION_RATE = 60.0


def log(*args):
    print("[ai4a-runner]", *args, flush=True)


@contextlib.contextmanager
def working_directory(path):
    previous = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


@contextlib.contextmanager
def device_safe_torch_load():
    """Networks in Demos/Authoring/Models were pickled on CUDA.

    torch.load(...) without map_location fails on CPU-only machines (e.g.
    macOS). Patch the default for the duration of MotionController.__init__.
    """
    import torch

    original = torch.load
    device = "cuda" if torch.cuda.is_available() else "cpu"

    @functools.wraps(original)
    def patched(*args, **kwargs):
        kwargs.setdefault("map_location", device)
        return original(*args, **kwargs)

    torch.load = patched
    try:
        yield
    finally:
        torch.load = original


def setup_paths(repo):
    repo = Path(repo).resolve()
    for sub in (repo / ASSETS_DIR, repo / AUTHORING_DIR, repo):
        if not sub.is_dir():
            raise FileNotFoundError("Expected directory not found: %s" % sub)
        if str(sub) not in sys.path:
            sys.path.insert(0, str(sub))
    return repo


def create_controller(repo):
    """Boot AI4Animation in MANUAL mode and build the Authoring MotionController."""
    from ai4animation import AI4Animation

    holder = {}

    class _Boot:
        def Start(self):
            from MotionController import MotionController

            with working_directory(repo / AUTHORING_DIR), device_safe_torch_load():
                holder["controller"] = MotionController()

    AI4Animation(_Boot(), mode=AI4Animation.Mode.MANUAL)
    return holder["controller"]


def build_profile(controller, repo):
    import Definitions

    actor = controller.Actor
    return RigProfile(
        name="Geno",
        model_file=str(ASSETS_DIR / "Model.glb"),
        bone_names=list(actor.GetBoneNames()),
        parent_names=list(actor.GetParentNames()),
        reference_transforms=np.array(actor.Transforms, dtype=float),
        root_bones={
            "hip": Definitions.HipName,
            "left_hip": Definitions.LeftHipName,
            "right_hip": Definitions.RightHipName,
            "left_shoulder": Definitions.LeftShoulderName,
            "right_shoulder": Definitions.RightShoulderName,
            "neck": Definitions.NeckName,
        },
        contact_bones=list(controller.ContactBones),
        guidance_names=list(controller.GuidanceNames),
        sequence={
            "window": float(controller.SequenceWindow),
            "length": int(controller.SequenceLength),
            "fps": int(controller.SequenceFPS),
            "network_input_dim": int(controller.Model.input_dim()),
            "postprocessor_input_dim": int(controller.PostProcessor.input_dim()),
        },
    )


# ----------------------------------------------------------------------------
# Path following
# ----------------------------------------------------------------------------


class PathFollower:
    """Goal transform along a path as a function of walked distance.

    Same spline and tangent as AuthoringProgram.GetPivotBySpeed, with two
    differences: STOP end behavior (demo only ping-pongs) and a backward
    tangent at the very end so the goal keeps facing along the path instead
    of snapping to +Z.
    """

    def __init__(self, path, end_behavior):
        self.Path = path
        self.EndBehavior = end_behavior
        self.Length = max(float(path.GetPathLength()), 1e-5)

    def percentage(self, walked):
        if self.EndBehavior == exchange.END_PINGPONG:
            p = math.fmod(max(walked, 0.0), 2.0 * self.Length) / self.Length
            return 2.0 - p if p > 1.0 else p
        return min(max(walked / self.Length, 0.0), 1.0)

    def goal(self, walked):
        from ai4animation import Rotation, Spline, Transform

        p = self.percentage(walked)
        step = 1.0 / (SPLINE_RESOLUTION - 1)
        points = self.Path.Points
        position = Spline.GetPointOnSpline(points, p)
        if p + step <= 1.0:
            tangent = Spline.GetPointOnSpline(points, p + step) - position
        else:
            tangent = position - Spline.GetPointOnSpline(points, max(p - step, 0.0))
        if float(np.linalg.norm(tangent)) < 1e-6:
            rotation = Rotation.Identity()
        else:
            rotation = Rotation.LookPlanar(tangent)
        return Transform.TR(position, rotation)

    def sampled_points(self):
        return np.asarray(self.Path.GetPathPoints(SPLINE_RESOLUTION), dtype=float)


def build_follower(meta, arrays, warnings):
    from PathPlanner3D import Path as SplinePath
    from PathPlanner3D import PathPlanner3D

    mode = meta["path_mode"]
    end = meta.get("end_behavior", exchange.END_STOP)
    if mode == exchange.PATH_CURVE:
        return PathFollower(SplinePath(arrays["path_points"]), end)
    if mode == exchange.PATH_PLANNER:
        cfg = meta["planner"]
        centers = arrays.get("obstacle_centers", np.zeros((0, 3)))
        sizes = arrays.get("obstacle_sizes", np.zeros((0, 3)))
        planner = PathPlanner3D(
            center=tuple(cfg["center"]),
            size=tuple(cfg["size"]),
            resolution=tuple(cfg["resolution"]),
            obstacles=list(zip(centers.tolist(), sizes.tolist())),
            project_zero=True,
        )
        path = planner.Search(arrays["start"], arrays["goal"], int(cfg["max_depth"]))
        points = np.asarray(path.Points)
        if points.shape[0] >= 2:
            last_voxel = points[-2]
            goal = np.asarray(arrays["goal"], dtype=float).copy()
            goal[1] = 0.0
            if np.linalg.norm(last_voxel - goal) > 1.5 * float(np.max(planner.Volume)):
                warnings.append(
                    "Planner could not reach the goal within max depth; the last "
                    "segment ignores obstacles. Increase Max Depth or the grid size."
                )
        return PathFollower(path, end)
    return None


# ----------------------------------------------------------------------------
# Simulation
# ----------------------------------------------------------------------------


def resolve_guidances(controller, meta, arrays):
    from ai4animation import GuidanceModule

    custom = arrays.get("custom_guidances")
    positions = []
    for name in meta["style_names"]:
        if name.startswith(exchange.CUSTOM_PREFIX):
            index = int(name[len(exchange.CUSTOM_PREFIX):])
            values = np.asarray(custom[index], dtype=np.float32)
        elif name in controller.GuidanceTemplates:
            values = controller.GuidanceTemplates[name].Positions
        else:
            raise KeyError(
                "Unknown guidance style %r. Available: %s" % (name, controller.GuidanceNames)
            )
        positions.append(np.array(values, dtype=np.float32))
    idle = meta.get("idle_style", IDLE_STYLE)
    if idle.startswith(exchange.CUSTOM_PREFIX):
        values = np.asarray(custom[int(idle[len(exchange.CUSTOM_PREFIX):])], dtype=np.float32)
        controller.GuidanceTemplates[IDLE_STYLE] = GuidanceModule.Guidance(
            IDLE_STYLE, controller.Actor.GetBoneNames(), values
        )
    elif idle != IDLE_STYLE:
        controller.GuidanceTemplates[IDLE_STYLE] = controller.GuidanceTemplates[idle]
    return positions


def initialize_state(controller, profile, meta, arrays, first_goal):
    """Place the actor at the start. Arrays are written IN PLACE: LegIK keeps
    views into Actor.Transforms from construction time."""
    from ai4animation import Transform, Vector3

    actor = controller.Actor
    if "initial_transforms" in arrays:
        pose = np.asarray(arrays["initial_transforms"], dtype=actor.Transforms.dtype)
        root = np.asarray(
            arrays["initial_root"] if "initial_root" in arrays else features.compute_root(pose, profile),
            dtype=np.float32,
        )
        velocities = arrays.get("initial_velocities", np.zeros((pose.shape[0], 3)))
    else:
        reference = np.array(actor.Transforms, dtype=float)
        reference_root = features.compute_root(reference, profile)
        root = features.root_from_position_direction(
            Transform.GetPosition(first_goal), Transform.GetAxisZ(first_goal)
        )
        pose = features.reroot(reference, reference_root, root)
        velocities = np.zeros((pose.shape[0], 3))

    actor.Transforms[...] = pose
    actor.Velocities[...] = velocities
    actor.SetRoot(np.array(root, dtype=np.float32))
    for leg in (controller.LeftLegIK, controller.RightLegIK):
        leg.AnkleTargetPosition = leg.AnkleIK.LastBone().GetPosition().copy()
        leg.AnkleTargetRotation = leg.AnkleIK.LastBone().GetRotation().copy()
        leg.BallTargetPosition = leg.BallIK.LastBone().GetPosition().copy()
        leg.BallTargetRotation = leg.BallIK.LastBone().GetRotation().copy()

    position = Transform.GetPosition(actor.Root).copy()
    position[1] = 0.0
    direction = Transform.GetAxisZ(actor.Root)
    for series in (controller.SimulationObject, controller.RootControl):
        for i in range(series.SampleCount):
            series.SetPosition(position, i)
            series.SetDirection(direction, i)
            series.SetVelocity(Vector3.Zero(), i)
    actor.SyncToScene()


def current_contacts(controller, prediction_fps):
    from ai4animation import Tensor, Time

    if controller.Sequence is None or controller.Sequence.Contacts is None:
        return np.zeros(len(controller.ContactBones))
    # Same blend as MotionController.Animate. The weight can exceed 1 between
    # predictions (extrapolation), so clamp for export.
    blend = (Time.TotalTime - controller.Timestamp) * prediction_fps
    contacts = Tensor.Interpolate(
        controller.Previous.SampleContacts(0.0), controller.Sequence.SampleContacts(0.0), blend
    )
    return np.clip(np.asarray(contacts, dtype=float), 0.0, 1.0)


def run(repo, request_path, result_path):
    started = time.time()
    meta, arrays = exchange.load_request(request_path)
    repo = setup_paths(repo)
    controller = create_controller(repo)
    profile = build_profile(controller, repo)

    if list(meta["bone_names"]) != profile.bone_names:
        raise ValueError(
            "Request bone order %s does not match the model %s" % (meta["bone_names"], profile.bone_names)
        )
    controller.NetworkIterations = int(meta.get("network_iterations", controller.NetworkIterations))

    from ai4animation import AI4Animation, Time

    warnings = []
    fps = float(meta["fps"])
    frames = int(meta["frame_count"])
    substeps = int(meta.get("substeps") or max(1, math.ceil(TARGET_SIMULATION_RATE / fps)))
    dt = 1.0 / (fps * substeps)
    prediction_fps = float(meta.get("prediction_fps", 10.0))
    strength = float(meta.get("control_strength", 2.0))
    speeds = np.asarray(arrays["speeds"], dtype=float)
    style_indices = np.asarray(arrays["style_indices"], dtype=int)
    styles = resolve_guidances(controller, meta, arrays)

    follower = build_follower(meta, arrays, warnings)
    walked = 0.0

    def goal_at(frame, walked_distance):
        if follower is None:
            return np.asarray(arrays["goals"][frame], dtype=np.float32)
        return np.asarray(follower.goal(walked_distance), dtype=np.float32)

    initialize_state(controller, profile, meta, arrays, goal_at(0, 0.0))

    joints = profile.bone_count
    out_transforms = np.zeros((frames, joints, 4, 4))
    out_velocities = np.zeros((frames, joints, 3))
    out_roots = np.zeros((frames, 4, 4))
    out_goals = np.zeros((frames, 4, 4))
    out_contacts = np.zeros((frames, len(controller.ContactBones)))

    def record(frame, goal):
        out_transforms[frame] = controller.Actor.Transforms
        out_velocities[frame] = controller.Actor.Velocities
        out_roots[frame] = controller.Actor.Root
        out_goals[frame] = goal
        out_contacts[frame] = current_contacts(controller, prediction_fps)

    class _Stepper:
        # AI4Animation.Update advances Time *before* calling Program.Update,
        # exactly like the standalone demo loop.
        goal = None
        guidance = None

        def Update(self):
            controller.Update(self.goal, strength, self.guidance, Time.DeltaTime, prediction_fps)

    stepper = _Stepper()
    AI4Animation.Program = stepper

    record(0, goal_at(0, 0.0))
    for frame in range(1, frames):
        speed = float(speeds[frame])
        stepper.guidance = styles[style_indices[frame]]
        for _ in range(substeps):
            walked += speed * dt
            stepper.goal = goal_at(frame, walked)
            AI4Animation.Update(dt)
        record(frame, stepper.goal)
        if frame % max(1, frames // 10) == 0:
            log("frame %d / %d" % (frame, frames))

    path_points = follower.sampled_points() if follower is not None else out_goals[:, :3, 3]
    exchange.save_result(
        result_path,
        {
            "fps": fps,
            "frame_count": frames,
            "substeps": substeps,
            "bone_names": profile.bone_names,
            "contact_bones": profile.contact_bones,
            "warnings": warnings,
            "seconds": time.time() - started,
        },
        transforms=out_transforms,
        velocities=out_velocities,
        roots=out_roots,
        goals=out_goals,
        contacts=out_contacts,
        path_points=path_points,
    )
    log("done: %d frames in %.1fs (%d substeps)" % (frames, time.time() - started, substeps))
    for w in warnings:
        log("WARNING:", w)


def profile_command(repo, out_path):
    repo = setup_paths(repo)
    controller = create_controller(repo)
    profile = build_profile(controller, repo)
    Path(out_path).write_text(profile.to_json())
    log("profile written to", out_path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("profile", help="Write the rig profile (bones, reference pose, styles).")
    p.add_argument("--repo", required=True)
    p.add_argument("--out", required=True)
    r = sub.add_parser("run", help="Simulate a request and write the result.")
    r.add_argument("--repo", required=True)
    r.add_argument("--request", required=True)
    r.add_argument("--result", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "profile":
            profile_command(args.repo, args.out)
        else:
            run(args.repo, args.request, args.result)
    except Exception as error:  # report cleanly to Blender, which shows stderr
        traceback.print_exc()
        print("[ai4a-runner] ERROR: %s" % error, file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
