# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Headless AI4Animation runner used by the Blender add-on.

Runs in the AI4Animation Python environment (torch + ai4animation), NOT in
Blender. Blender talks to it through files written by
ai4animation_blender.middleware.exchange, all in AI4Animation world space.

    python ai4a_runner.py profile --repo <ai4animationpy> --out profile.json
    python ai4a_runner.py run     --repo <ai4animationpy> --request req.npz --result res.npz

The simulation reuses the demos' controller code as-is, driven in
AI4Animation MANUAL mode with a fixed time step, so Blender gets the same
behavior as the standalone demos:

    BIPED      Demos/Authoring MotionController (Geno)
    QUADRUPED  Demos/Locomotion/Quadruped Program (Dog / Wolf): Predict and
               Animate unchanged, Control fed by the virtual player

    python ai4a_runner.py profile --repo <repo> --model QUADRUPED --character dog --out dog.json
"""

import argparse
import contextlib
import functools
import importlib.util
import math
import os
import sys
import time
import traceback
import types
from pathlib import Path

import numpy as np

ADDONS_DIR = Path(__file__).resolve().parents[2]
if str(ADDONS_DIR) not in sys.path:
    sys.path.insert(0, str(ADDONS_DIR))

from ai4animation_blender.middleware import exchange, features, models  # noqa: E402
from ai4animation_blender.middleware.rig import RigProfile  # noqa: E402

AUTHORING_DIR = Path(models.MODELS[models.BIPED].demo_dir)
PATH_PLANNER = AUTHORING_DIR / "PathPlanner3D.py"  # spline + planner, used by every model

# Must match Demos/Authoring/Program.py
SPLINE_RESOLUTION = 80
IDLE_STYLE = "Idle"
# The demo runs at display refresh rate; sub-stepping keeps the controller in
# the regime it was tuned for when Blender scenes run at 24/25/30 fps.
TARGET_SIMULATION_RATE = 60.0
# Virtual player of the joystick controller (meters, m/s): release the stick
# within STOP_RADIUS of a stopped reference, push again beyond RESUME_RADIUS,
# never command more than MAX_COMMAND.
STOP_RADIUS = 0.15
RESUME_RADIUS = 0.5
MAX_COMMAND = 3.0
BRAKE = 1.0  # m/s^2, deceleration of the path reference before its end


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


def setup_paths(repo, spec=None):
    """Put the model's assets (Definitions.py) and demo folder on sys.path.

    Only one model per process: both demos ship LegIK.py, Sequence.py and
    Program.py under the same names.
    """
    spec = spec or models.MODELS[models.BIPED]
    repo = Path(repo).resolve()
    for sub in (repo / spec.assets_dir, repo / spec.demo_dir, repo):
        if not sub.is_dir():
            raise FileNotFoundError("Expected directory not found: %s" % sub)
        if str(sub) not in sys.path:
            sys.path.insert(0, str(sub))
    return repo


def load_module(path, name):
    """Import a demo file under a unique module name (no clash with other demos)."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def path_planner(repo):
    return load_module(Path(repo) / PATH_PLANNER, "ai4a_path_planner3d")


def resolve_file(path):
    """`path`, or a file in the same folder whose name differs only in case.

    The quadruped demo loads "PostProcessor.pt" but ships "Postprocessor.pt",
    which only works on case-insensitive file systems.
    """
    path = Path(path)
    if path.is_file():
        return path
    if path.parent.is_dir():
        for candidate in path.parent.iterdir():
            if candidate.name.lower() == path.name.lower() and candidate.is_file():
                return candidate
    raise FileNotFoundError("Network file not found: %s" % path)


def network_files(repo, spec, info):
    """(network, postprocessor): the user's overrides or the demo's files."""
    demo = Path(repo) / spec.demo_dir
    network = resolve_file(info.get("network") or demo / spec.network)
    postprocessor = resolve_file(info.get("postprocessor") or demo / spec.postprocessor)
    return network, postprocessor


def load_network(path):
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.load(str(path), weights_only=False, map_location=device)
    model.eval()
    return model


def check_networks(controller, spec, network, postprocessor):
    """The networks must read exactly what this model's controller feeds them."""
    bones = controller.Actor.GetBoneCount()
    contacts = len(controller.ContactBones)
    expected = spec.network_input_dim(bones)
    got = int(controller.Model.input_dim())
    if got != expected:
        raise ValueError(
            "Network %s expects %d inputs, but the %s controller feeds %d (%d bones). "
            "Pick a network trained for this model." % (network, got, spec.label, expected, bones)
        )
    expected = spec.postprocessor_input_dim(bones, contacts)
    got = int(controller.PostProcessor.input_dim())
    if got != expected:
        raise ValueError(
            "PostProcessor %s expects %d inputs, but the %s controller feeds %d. "
            "Pick the contact network trained with this model." % (postprocessor, got, spec.label, expected)
        )


def create_controller(repo, spec=None, info=None):
    """Boot AI4Animation in MANUAL mode and build the model's controller."""
    spec = spec or models.MODELS[models.BIPED]
    info = exchange.model_meta({"model": dict(info or {}, type=spec.key)})
    network, postprocessor = network_files(repo, spec, info)
    if spec.key == models.QUADRUPED:
        controller = create_quadruped(repo, spec, info["character"], network, postprocessor)
    else:
        controller = create_biped(repo, network, postprocessor)
    check_networks(controller, spec, network, postprocessor)
    return controller


def create_biped(repo, network, postprocessor):
    """The Authoring MotionController; custom networks replace the demo's after loading."""
    from ai4animation import AI4Animation

    holder = {}

    class _Boot:
        def Start(self):
            from MotionController import MotionController

            with working_directory(repo / AUTHORING_DIR), device_safe_torch_load():
                holder["controller"] = MotionController()

    AI4Animation(_Boot(), mode=AI4Animation.Mode.MANUAL)
    controller = holder["controller"]
    demo = repo / AUTHORING_DIR / "Models"
    if network.resolve() != resolve_file(demo / "Network.pt").resolve():
        controller.Model = load_network(network)
    if postprocessor.resolve() != resolve_file(demo / "PostProcessor.pt").resolve():
        controller.PostProcessor = load_network(postprocessor)
    return controller


class _Headless:
    """Stands in for AI4Animation.Standalone inside the quadruped Program only:
    its Start() sets a camera target and checks the gamepad."""

    class _NoOp:
        def __getattr__(self, name):
            return lambda *args, **kwargs: None

    Camera = _NoOp()
    IO = _NoOp()


def create_quadruped(repo, spec, character, network, postprocessor):
    """Demos/Locomotion/Quadruped Program, started headless.

    The demo's own Start() runs with three shims: one character only (the
    demo creates hidden copies that need a renderer), a module-local
    AI4Animation whose Standalone is a no-op camera/gamepad, and torch.load
    redirected to the chosen network files (on CPU if needed). Predict and
    Animate are the demo's; Control is replaced by quadruped_control.
    """
    import torch
    from ai4animation import AI4Animation

    program = load_module(repo / spec.demo_dir / "Program.py", "ai4a_quadruped_program")
    if character not in program.CHARACTER_MODELS:
        raise ValueError("Unknown quadruped character %r (%s)" % (character, sorted(program.CHARACTER_MODELS)))

    class _Boot:
        def Start(self):
            pass

    AI4Animation(_Boot(), mode=AI4Animation.Mode.MANUAL)

    class _ModuleAI4Animation:
        Standalone = _Headless()

        def __getattr__(self, name):
            return getattr(AI4Animation, name)

    program.AI4Animation = _ModuleAI4Animation()
    program.CHARACTER_MODELS = {character: program.CHARACTER_MODELS[character]}

    files = {"network.pt": network, "postprocessor.pt": postprocessor}
    original = torch.load
    device = "cuda" if torch.cuda.is_available() else "cpu"

    @functools.wraps(original)
    def load(path, *args, **kwargs):
        kwargs.setdefault("map_location", device)
        return original(str(files.get(Path(path).name.lower(), path)), *args, **kwargs)

    controller = program.Program()
    controller.Character = character
    torch.load = load
    try:
        with working_directory(repo / spec.demo_dir):
            controller.Start()
    finally:
        torch.load = original
    controller.Module = program
    controller.Command = np.zeros(3)
    controller.Action = None
    controller.Control = types.MethodType(quadruped_control, controller)
    return controller


def build_profile(controller, repo, spec=None, character=None):
    import Definitions

    spec = spec or models.MODELS[models.BIPED]
    character = spec.character(character or spec.characters[0].key)
    actor = controller.Actor
    if spec.key == models.QUADRUPED:
        module = controller.Module
        guidance_names = sorted(controller.GuidanceTemplates)
        window, length, fps = module.SEQUENCE_WINDOW, module.SEQUENCE_LENGTH, module.SEQUENCE_FPS
    else:
        guidance_names = list(controller.GuidanceNames)
        window, length, fps = controller.SequenceWindow, controller.SequenceLength, controller.SequenceFPS
    return RigProfile(
        name=character.label,
        model_file=character.model_file,
        model=spec.key,
        character=character.key,
        root_topology=spec.root_topology,
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
        guidance_names=guidance_names,
        sequence={
            "window": float(window),
            "length": int(length),
            "fps": int(fps),
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


class PathReference:
    """Reference point moving along the path at the keyed speed (time-accurate).

    Uses the same Catmull-Rom spline as the goal controller. The reference is
    kept at most `leash` meters ahead of the character's projection onto the
    path, so a lagging character is pulled along instead of cutting corners.
    """

    def __init__(self, path, end_behavior, leash, gain, spacing=0.05, max_command=None):
        from ai4animation import Spline

        length = max(float(path.GetPathLength()), 1e-5)
        count = max(2, int(math.ceil(length / spacing)) + 1)
        dense = np.asarray(Spline.GetPointsOnSpline(path.Points, count), dtype=float)
        dense[:, 1] = 0.0
        segment = np.linalg.norm(np.diff(dense, axis=0), axis=-1)
        self.Points = dense
        self.Arc = np.concatenate(([0.0], np.cumsum(segment)))
        self.Length = float(self.Arc[-1])
        self.EndBehavior = end_behavior
        self.Leash = float(leash)
        self.Tracker = ReferenceTracker(gain, max_command)
        self.S = 0.0
        self.Projected = 0.0
        self.Sign = 1.0

    def point_at(self, s):
        s = min(max(s, 0.0), self.Length)
        return np.array([np.interp(s, self.Arc, self.Points[:, k]) for k in range(3)])

    def _project(self, position):
        # Search near the current progress so self-crossing paths are followed in order.
        lo, hi = self.Projected - 0.5 - self.Leash, self.Projected + 0.5 + self.Leash
        candidates = np.nonzero((self.Arc >= lo) & (self.Arc <= hi))[0]
        delta = self.Points[candidates] - position
        delta[:, 1] = 0.0
        return float(self.Arc[candidates[int(np.argmin(np.sum(delta * delta, axis=-1)))]])

    def tangent_at(self, s, sign=1.0):
        ahead = self.point_at(s + sign * 0.05) - self.point_at(s - sign * 0.05)
        norm = float(np.linalg.norm(ahead))
        return ahead / norm if norm > 1e-9 else np.zeros(3)

    def command(self, position, speed, dt):
        """Velocity command (m/s) of the virtual left stick.

        The reference advances along the path at the keyed speed (time
        accurate), at most `leash` ahead of the character's projection, and
        with STOP brakes into the end at BRAKE m/s^2 (stopping it abruptly
        from full speed makes the character overshoot by its momentum). The
        stick tracks that point: feed-forward velocity + gain * offset.
        Measured on S-curve / 90 degree corner / hairpin paths: tracking the
        reference point is more stable under the network's reaction delay
        than cross-track (path-normal) feedback, and gain 2/s beats 3 and 4
        on tight turns. Returns (velocity, reference point).
        """
        position = np.array(position, dtype=float).reshape(3)
        position[1] = 0.0
        self.Projected = self._project(position)
        stop = self.EndBehavior != exchange.END_PINGPONG
        if stop:
            speed = min(speed, math.sqrt(2.0 * BRAKE * max(self.Length - self.S, 0.0)))
        s = self.S + self.Sign * speed * dt
        s = min(s, self.Projected + self.Leash) if self.Sign > 0 else max(s, self.Projected - self.Leash)
        if not stop and (s >= self.Length or s <= 0.0):
            self.Sign = -self.Sign
        self.S = min(max(s, 0.0), self.Length)
        reference = self.point_at(self.S)
        parked = stop and self.S >= self.Length - 1e-9
        feedforward = np.zeros(3) if parked else self.tangent_at(self.S, self.Sign) * speed
        return self.Tracker.command(reference, feedforward, position), reference

    def sampled_points(self):
        return self.Points


class ReferenceTracker:
    """Virtual player: turns a moving reference into a stick velocity command.

    velocity = reference velocity (feed-forward) + gain * position error,
    clamped to MAX_COMMAND. Pure velocity commands (a real gamepad) cannot
    hold speeds below ~0.6 m/s or start walking below ~1 m/s from standing
    with this network; tracking a position that moves at the keyed speed
    makes both exact, like the goal controller, while keeping the joystick
    controller's independent facing. When the reference stops, the stick is
    released within STOP_RADIUS and re-engaged beyond RESUME_RADIUS.
    """

    def __init__(self, gain, max_command=None):
        self.Gain = float(gain)
        self.MaxCommand = MAX_COMMAND if max_command is None else float(max_command)
        self.Engaged = True

    def command(self, reference, reference_velocity, position):
        error = np.asarray(reference, dtype=float).reshape(3) - np.asarray(position, dtype=float).reshape(3)
        error[1] = 0.0
        feedforward = np.asarray(reference_velocity, dtype=float).reshape(3).copy()
        feedforward[1] = 0.0
        distance = float(np.linalg.norm(error))
        if float(np.linalg.norm(feedforward)) > 1e-6:
            self.Engaged = True
        elif self.Engaged and distance <= STOP_RADIUS:
            self.Engaged = False
        elif not self.Engaged and distance > RESUME_RADIUS:
            self.Engaged = True
        if not self.Engaged:
            return np.zeros(3)
        velocity = feedforward + self.Gain * error
        speed = float(np.linalg.norm(velocity))
        if speed > self.MaxCommand:
            velocity *= self.MaxCommand / speed
        return velocity


def stick_control(self, command, control_strength, guidance_pose):
    """Joystick controller of Demos/Locomotion/Biped/Program.py (Control).

    Bound onto the Authoring MotionController in STICK mode, so Update,
    PredictSequence and Animate stay exactly as in the demo. `command` is a
    (velocity, direction) pair in world space: velocity = left stick * speed,
    direction = right stick (zero = face the movement direction).
    """
    from ai4animation import Tensor, Time, Transform, Vector3

    velocity, direction = command
    position = Vector3.Lerp(
        self.SimulationObject.GetPosition(0), self.Actor.GetRootPosition(), self.Synchronization
    )
    self.SimulationObject.Control(position, direction, velocity, Time.DeltaTime)

    speed = float(np.linalg.norm(velocity))
    template = self.GuidanceTemplates["Idle"].Positions if speed < 0.1 else guidance_pose
    self.GuidanceControl.Positions = np.array(template, copy=True)

    # Correction (identical in the Biped and Authoring controllers)
    if self.Sequence is not None:
        self.RootControl.Transforms = Transform.Interpolate(
            self.SimulationObject.Transforms,
            self.Sequence.Trajectory.Transforms,
            self.TrajectoryCorrection,
        )
        for i in range(self.RootControl.SampleCount):
            target = Transform.GetPosition(self.RootControl.Transforms)[i:]
            current = self.Actor.GetRootPosition().reshape(-1, 3)
            time = self.RootControl.Timestamps[i:].reshape(-1, 1)
            self.RootControl.Velocities[i] = Tensor.Sum(
                target - current, axis=0, keepDim=False
            ) / Tensor.Sum(time, axis=0, keepDim=False)
        self.RootControl.Velocities = Vector3.Lerp(
            self.RootControl.Velocities,
            self.Sequence.Trajectory.Velocities,
            self.TrajectoryCorrection,
        )


def quadruped_control(self):
    """Control() of Demos/Locomotion/Quadruped/Program.py without the gamepad.

    self.Command is the virtual left stick as a world velocity (m/s) and
    self.Action one of Sit / Stand / Lie or None. Everything after reading the
    input is the demo's: PID speed smoothing, facing = movement, gait guidance
    by speed, action poses only below ACTION_TRIGGER_SPEED_MAX, trajectory
    correction. One difference: an action also releases the stick (the demo
    needs the player to stop first), so a keyed "Sit" slows down, then sits.
    """
    from ai4animation import Tensor, Time, Transform, Vector3

    demo = self.Module
    command = np.array(self.Command, dtype=float).reshape(3)
    command[1] = 0.0
    action = self.Action
    current_speed = self.GetCurrentSpeed()

    desired_speed = float(np.linalg.norm(command))
    move_direction = command / desired_speed if desired_speed > 1e-6 else Vector3.Zero()
    if action is not None:
        desired_speed = 0.0

    can_trigger_action_pose = current_speed < demo.ACTION_TRIGGER_SPEED_MAX
    sit_active = can_trigger_action_pose and action == "Sit"
    stand_active = can_trigger_action_pose and action == "Stand"
    lie_active = can_trigger_action_pose and action == "Lie"

    action_pose_active = sit_active or lie_active or stand_active
    target_speed = 0.0 if action_pose_active else desired_speed

    speed = current_speed + self.PID(current_speed, Time.DeltaTime, setpoint=target_speed)
    speed = max(speed, 0.0)

    if action_pose_active:
        speed = 0.0
        velocity = Vector3.Zero()
        direction = self.Actor.GetRootDirection()
    else:
        velocity = speed * move_direction
        direction = velocity

    self._UpdatePIDSpeedHistory(current_speed, target_speed, speed)

    position = Vector3.Lerp(
        self.SimulationObject.GetPosition(0),
        self.Actor.GetRootPosition(),
        self.Synchronization,
    )
    self.SimulationObject.Control(position, direction, velocity, Time.DeltaTime)

    speed = Vector3.Length(velocity)
    modes = demo.LOCOMOTION_MODES
    if sit_active:
        guidance_state = "Sit"
    elif lie_active:
        guidance_state = "Lie"
    elif stand_active:
        guidance_state = "Stand"
    elif speed < 0.1:
        guidance_state = "Sit" if self.Sequence is None else "Idle"
    elif speed < modes["pace"]:
        guidance_state = "Walk"
    elif speed < modes["trot"]:
        guidance_state = "Pace"
    elif speed < modes["canter"]:
        guidance_state = "Trot"
    else:
        guidance_state = "Canter"

    self.CurrentGuidanceState = guidance_state
    self.GuidanceControl.Positions = self.GuidanceTemplates[guidance_state].Positions.copy()

    self.RootControl.Transforms = self.SimulationObject.Transforms.copy()
    self.RootControl.Velocities = self.SimulationObject.Velocities.copy()

    # Correction (as in the demo)
    if self.Sequence is not None:
        self.RootControl.Transforms = Transform.Interpolate(
            self.SimulationObject.Transforms,
            self.Sequence.Trajectory.Transforms,
            self.TrajectoryCorrection,
        )
        for i in range(self.RootControl.SampleCount):
            target = Transform.GetPosition(self.RootControl.Transforms)[i:]
            current = self.Actor.GetRootPosition().reshape(-1, 3)
            time = self.RootControl.Timestamps[i:].reshape(-1, 1)
            self.RootControl.Velocities[i] = Tensor.Sum(
                target - current, axis=0, keepDim=False
            ) / Tensor.Sum(time, axis=0, keepDim=False)
        self.RootControl.Velocities = Vector3.Lerp(
            self.RootControl.Velocities,
            self.Sequence.Trajectory.Velocities,
            self.TrajectoryCorrection,
        )


def build_path(repo, meta, arrays, warnings):
    """The demo Path object for CURVE / PLANNER modes (None otherwise)."""
    planner_module = path_planner(repo)
    SplinePath = planner_module.Path
    PathPlanner3D = planner_module.PathPlanner3D

    mode = meta["path_mode"]
    if mode == exchange.PATH_CURVE:
        return SplinePath(arrays["path_points"])
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
        return path
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
    reset_leg_ik(controller)

    position = Transform.GetPosition(actor.Root).copy()
    position[1] = 0.0
    direction = Transform.GetAxisZ(actor.Root)
    for series in (controller.SimulationObject, controller.RootControl):
        for i in range(series.SampleCount):
            series.SetPosition(position, i)
            series.SetDirection(direction, i)
            series.SetVelocity(Vector3.Zero(), i)
    actor.SyncToScene()


def reset_leg_ik(controller):
    """Point the leg IK targets at the new pose (they remember the last contact)."""
    for name in ("LeftLegIK", "RightLegIK"):  # biped: ankle + ball chains
        leg = getattr(controller, name, None)
        if leg is not None:
            leg.AnkleTargetPosition = leg.AnkleIK.LastBone().GetPosition().copy()
            leg.AnkleTargetRotation = leg.AnkleIK.LastBone().GetRotation().copy()
            leg.BallTargetPosition = leg.BallIK.LastBone().GetPosition().copy()
            leg.BallTargetRotation = leg.BallIK.LastBone().GetRotation().copy()
    for name in ("LeftHandIK", "RightHandIK", "LeftFootIK", "RightFootIK"):  # quadruped: one chain per leg
        leg = getattr(controller, name, None)
        if leg is not None:
            leg.TargetPosition = leg.IK.LastBone().GetPosition().copy()
            leg.TargetRotation = leg.IK.LastBone().GetRotation().copy()


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
    info = exchange.model_meta(meta)
    spec = models.get(info["type"])
    quadruped = spec.key == models.QUADRUPED
    repo = setup_paths(repo, spec)
    controller = create_controller(repo, spec, info)
    profile = build_profile(controller, repo, spec, info["character"])

    if list(meta["bone_names"]) != profile.bone_names:
        raise ValueError(
            "Request bone order %s does not match the model %s" % (meta["bone_names"], profile.bone_names)
        )
    controller.NetworkIterations = int(meta.get("network_iterations", spec.network_iterations))

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
    if quadruped:
        # the demo reads its prediction rate from a module constant
        controller.Module.PREDICTION_FPS = prediction_fps
        styles = [None if name == models.QUADRUPED_AUTO else name for name in meta["style_names"]]
    else:
        styles = resolve_guidances(controller, meta, arrays)

    def goal_at(frame, walked_distance):
        if follower is None:
            return np.asarray(arrays["goals"][frame], dtype=np.float32)
        return np.asarray(follower.goal(walked_distance), dtype=np.float32)

    mode = meta["path_mode"]
    control = meta.get("controller", exchange.CONTROLLER_GOAL)
    end_behavior = meta.get("end_behavior", exchange.END_STOP)
    path = build_path(repo, meta, arrays, warnings)
    follower = PathFollower(path, end_behavior) if path is not None else None
    if mode == exchange.PATH_STICK:
        first_goal = np.asarray(arrays.get("start_transform", np.eye(4)), dtype=np.float32)
    elif follower is not None:
        first_goal = np.asarray(follower.goal(0.0), dtype=np.float32)
    else:
        first_goal = np.asarray(arrays["goals"][0], dtype=np.float32)
    initialize_state(controller, profile, meta, arrays, first_goal)

    reference = None
    tracker = None
    if control == exchange.CONTROLLER_STICK:
        if not quadruped:  # the quadruped controller already has its stick Control
            controller.Control = types.MethodType(stick_control, controller)
        cfg = dict(exchange.DEFAULT_TRACKING, **meta.get("tracking", {}))
        assist = bool(cfg["assist"])
        max_command = max(MAX_COMMAND, spec.max_speed + 1.0)
        tracker = ReferenceTracker(cfg["gain"], max_command)
        leash = float(cfg["leash"])
        if path is not None:
            reference = PathReference(path, end_behavior, leash, cfg["gain"], max_command=max_command)
        # Start the stick's ghost where the actor really starts (initial pose may differ).
        ghost = np.array(controller.Actor.GetRootPosition(), dtype=float).reshape(3)
    facing_mode = meta.get("facing_mode", exchange.FACING_MOVE)

    joints = profile.bone_count
    out_transforms = np.zeros((frames, joints, 4, 4))
    out_velocities = np.zeros((frames, joints, 3))
    out_roots = np.zeros((frames, 4, 4))
    out_goals = np.zeros((frames, 4, 4))
    out_contacts = np.zeros((frames, len(controller.ContactBones)))
    out_commands = np.zeros((frames, 3))
    out_facings = np.zeros((frames, 3))

    def record(frame, goal, command=None, facing=None):
        out_transforms[frame] = controller.Actor.Transforms
        out_velocities[frame] = controller.Actor.Velocities
        out_roots[frame] = controller.Actor.Root
        out_goals[frame] = goal
        out_contacts[frame] = current_contacts(controller, prediction_fps)
        if command is not None:
            out_commands[frame] = command
            out_facings[frame] = facing

    def stick_command(frame, speed):
        """(velocity, facing) of the virtual left/right sticks for one sub-step."""
        nonlocal ghost
        root = np.array(controller.Actor.GetRootPosition(), dtype=float).reshape(3)
        if reference is not None:
            velocity, _ = reference.command(root, speed, dt)
        elif mode == exchange.PATH_TARGET:
            target = np.asarray(arrays["goals"][frame][:3, 3], dtype=float)
            previous = np.asarray(arrays["goals"][max(frame - 1, 0)][:3, 3], dtype=float)
            velocity = tracker.command(target, (target - previous) * fps, root)
        else:
            stick = np.array(arrays["move_sticks"][frame], dtype=float)
            stick[1] = 0.0
            length = float(np.linalg.norm(stick))
            stick = stick / length if length > 1.0 else stick
            if assist:
                # Integrate the stick into a ghost position on a leash and track it.
                ghost = ghost + speed * stick * dt
                offset = ghost - root
                offset[1] = 0.0
                if np.linalg.norm(offset) > leash:
                    ghost = root + offset / np.linalg.norm(offset) * leash
                velocity = tracker.command(ghost, speed * stick, root)
            else:
                velocity = speed * stick  # exactly the gamepad demo
        if facing_mode == exchange.FACING_LOOK_AT:
            facing = np.array(arrays["facing_points"][frame], dtype=float) - root
            facing[1] = 0.0
            if np.linalg.norm(facing) < 0.05:
                facing = np.zeros(3)
        elif facing_mode == exchange.FACING_DIRECTION:
            facing = np.array(arrays["facing_directions"][frame], dtype=float)
            facing[1] = 0.0
        else:
            facing = np.zeros(3)
        return velocity, facing, (velocity.astype(np.float32), facing.astype(np.float32))

    class _Stepper:
        # AI4Animation.Update advances Time *before* calling Program.Update,
        # exactly like the standalone demo loop.
        command = None
        guidance = None

        def Update(self):
            if quadruped:
                controller.Command = self.command[0]
                controller.Action = self.guidance
                controller.Update()
            else:
                controller.Update(self.command, strength, self.guidance, Time.DeltaTime, prediction_fps)

    stepper = _Stepper()
    AI4Animation.Program = stepper

    walked = 0.0
    record(0, first_goal, np.zeros(3), np.zeros(3))
    for frame in range(1, frames):
        speed = float(speeds[frame])
        stepper.guidance = styles[style_indices[frame]]
        command = facing = None
        for _ in range(substeps):
            if control == exchange.CONTROLLER_STICK:
                command, facing, stepper.command = stick_command(frame, speed)
            else:
                walked += speed * dt
                stepper.command = goal_at(frame, walked)
            AI4Animation.Update(dt)
        if control == exchange.CONTROLLER_STICK:
            # "Goal" of a stick controller: where its simulated trajectory ends (0.5 s ahead)
            goal = np.array(controller.SimulationObject.Transforms[-1], dtype=float)
        else:
            goal = stepper.command
        record(frame, goal, command, facing)
        if frame % max(1, frames // 10) == 0:
            log("frame %d / %d" % (frame, frames))

    if reference is not None:
        path_points = reference.sampled_points()
    elif follower is not None:
        path_points = follower.sampled_points()
    else:
        path_points = out_goals[:, :3, 3]
    exchange.save_result(
        result_path,
        {
            "fps": fps,
            "frame_count": frames,
            "substeps": substeps,
            "controller": control,
            "model": info,
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
        commands=out_commands,
        facings=out_facings,
    )
    log("done: %d frames in %.1fs (%d substeps)" % (frames, time.time() - started, substeps))
    for w in warnings:
        log("WARNING:", w)


def profile_command(repo, out_path, model=models.BIPED, character=None, network="", postprocessor=""):
    spec = models.get(model)
    info = {"character": character or spec.characters[0].key, "network": network, "postprocessor": postprocessor}
    repo = setup_paths(repo, spec)
    controller = create_controller(repo, spec, info)
    profile = build_profile(controller, repo, spec, info["character"])
    Path(out_path).write_text(profile.to_json())
    log("profile written to", out_path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("profile", help="Write the rig profile (bones, reference pose, styles).")
    p.add_argument("--repo", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model", default=models.BIPED, choices=sorted(models.MODELS))
    p.add_argument("--character", default=None)
    p.add_argument("--network", default="", help="Network .pt to use instead of the demo's")
    p.add_argument("--postprocessor", default="", help="Contact network .pt to use instead of the demo's")
    r = sub.add_parser("run", help="Simulate a request and write the result.")
    r.add_argument("--repo", required=True)
    r.add_argument("--request", required=True)
    r.add_argument("--result", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "profile":
            profile_command(args.repo, args.out, args.model, args.character, args.network, args.postprocessor)
        else:
            run(args.repo, args.request, args.result)
    except Exception as error:  # report cleanly to Blender, which shows stderr
        traceback.print_exc()
        print("[ai4a-runner] ERROR: %s" % error, file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
