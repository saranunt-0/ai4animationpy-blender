# Copyright (c) Meta Platforms, Inc. and affiliates.
import json
from pathlib import Path

import bpy

from .middleware import exchange, rig

SHIPPED_PROFILE = Path(__file__).resolve().parent / "profiles" / "geno_profile.json"

_enum_cache = []  # Blender requires dynamic enum item strings to stay referenced
_main_cache = []
_idle_cache = []
DEFAULT_STYLE = "Neutral"


def get_profile(scene):
    text = scene.ai4a.profile_json or SHIPPED_PROFILE.read_text()
    return rig.RigProfile.from_json(text)


def style_items(self, context):
    items = []
    scene = context.scene if context else None
    try:
        names = get_profile(scene).guidance_names if scene else []
    except Exception:  # corrupted profile text: fall back to shipped one
        names = rig.RigProfile.from_json(SHIPPED_PROFILE.read_text()).guidance_names
    for name in names:
        items.append((name, name, "Guidance style from Demos/Authoring/Guidances/%s.npz" % name))
    if scene is not None:
        for custom in scene.ai4a.custom_styles:
            items.append(
                (exchange.CUSTOM_PREFIX + custom.name, "* " + custom.name, "Captured from a Blender pose")
            )
    if not items:
        items.append(("Neutral", "Neutral", ""))
    _enum_cache[:] = items
    return _enum_cache


def _first(items, name):
    """Dynamic enums default to their first item: move `name` to the front."""
    return sorted(items, key=lambda item: item[0] != name)


def main_style_items(self, context):
    _main_cache[:] = _first(style_items(self, context), DEFAULT_STYLE)
    return _main_cache


def idle_items(self, context):
    _idle_cache[:] = _first(style_items(self, context), "Idle")
    return _idle_cache


def _poll_armature(self, obj):
    return obj.type == "ARMATURE"


def _poll_curve(self, obj):
    return obj.type == "CURVE"


class AI4A_CustomStyle(bpy.types.PropertyGroup):
    name: bpy.props.StringProperty(name="Name", default="Custom")
    data: bpy.props.StringProperty(name="Positions", description="JSON (J, 3) root-space positions")

    def positions(self):
        return json.loads(self.data)


class AI4A_StyleKey(bpy.types.PropertyGroup):
    frame: bpy.props.IntProperty(name="Frame", default=1)
    style: bpy.props.EnumProperty(name="Style", items=style_items)


class AI4A_Settings(bpy.types.PropertyGroup):
    profile_json: bpy.props.StringProperty(options={"HIDDEN"})

    armature: bpy.props.PointerProperty(name="Armature", type=bpy.types.Object, poll=_poll_armature)

    path_mode: bpy.props.EnumProperty(
        name="Movement",
        items=(
            (exchange.PATH_CURVE, "Curve", "Walk along a curve object (e.g. a Bezier path)"),
            (exchange.PATH_PLANNER, "Planner", "Plan a path from Start to Goal around obstacle boxes"),
            (exchange.PATH_TARGET, "Target", "Follow an animated object"),
            (exchange.PATH_STICK, "Joystick", "Keyframe a virtual left stick: a knob moving inside its gate circle"),
        ),
        default=exchange.PATH_CURVE,
    )
    facing_mode: bpy.props.EnumProperty(
        name="Facing",
        items=(
            ("MOVE", "Movement", "Face where the character walks (right stick released)"),
            ("LOOK_AT", "Look At", "Keep facing an object while walking (strafe / walk backwards)"),
            ("OBJECT", "Object Axis", "Face along an object's -Y axis (Blender character forward)"),
            ("STICK", "Right Stick", "Keyframe a virtual right stick knob; centered = face movement"),
        ),
        default="MOVE",
    )
    facing_object: bpy.props.PointerProperty(name="Facing Object", type=bpy.types.Object)
    left_stick: bpy.props.PointerProperty(
        name="Left Stick", type=bpy.types.Object,
        description="Knob object; its offset inside its parent (gate, radius 1) is the stick deflection",
    )
    right_stick: bpy.props.PointerProperty(
        name="Right Stick", type=bpy.types.Object,
        description="Knob object; its offset inside its parent (gate) is the facing direction",
    )
    controller: bpy.props.EnumProperty(
        name="Controller",
        items=(
            ("AUTO", "Auto", "Goal controller for paths facing the movement, joystick controller otherwise"),
            (exchange.CONTROLLER_GOAL, "Goal", "Demos/Authoring: follow a goal moving along the path"),
            (exchange.CONTROLLER_STICK, "Joystick", "Demos/Locomotion/Biped gamepad control (velocity + facing)"),
        ),
        default="AUTO",
    )
    stick_assist: bpy.props.BoolProperty(
        name="Speed Assist", default=True,
        description=(
            "Track the position the stick implies so speeds are exact. Off = raw gamepad input "
            "(below ~1 m/s the model may not start walking, as in the demo)"
        ),
    )
    tracking_gain: bpy.props.FloatProperty(
        name="Tracking Gain", default=2.0, min=0.0, soft_max=6.0,
        description="Joystick controller: how strongly (1/s) the virtual player corrects position errors",
    )
    tracking_leash: bpy.props.FloatProperty(
        name="Leash", default=1.0, min=0.1, soft_max=3.0, unit="LENGTH",
        description="Joystick controller: how far the reference may run ahead of the character",
    )
    curve: bpy.props.PointerProperty(name="Curve", type=bpy.types.Object, poll=_poll_curve)
    path_spacing: bpy.props.FloatProperty(
        name="Spacing", default=0.25, min=0.02, soft_max=2.0, unit="LENGTH",
        description="Resample the curve with this point spacing (constant walking speed)",
    )
    start_object: bpy.props.PointerProperty(name="Start", type=bpy.types.Object)
    goal_object: bpy.props.PointerProperty(name="Goal", type=bpy.types.Object)
    obstacles: bpy.props.PointerProperty(
        name="Obstacles", type=bpy.types.Collection,
        description="Mesh objects in this collection are obstacles (axis-aligned bounds)",
    )
    planner_cell: bpy.props.FloatProperty(name="Cell Size", default=0.8, min=0.1, unit="LENGTH")
    planner_margin: bpy.props.FloatProperty(name="Margin", default=2.0, min=0.0, unit="LENGTH")
    planner_height: bpy.props.FloatProperty(name="Height", default=2.0, min=0.1, unit="LENGTH")
    planner_max_depth: bpy.props.IntProperty(name="Max Depth", default=80, min=1)
    target_object: bpy.props.PointerProperty(name="Target", type=bpy.types.Object)

    walk_speed: bpy.props.FloatProperty(
        name="Walk Speed", default=1.0, min=0.0, soft_max=3.0, unit="VELOCITY",
        description="Meters per second along the path, or at full stick deflection (can be keyframed)",
    )
    control_strength: bpy.props.FloatProperty(name="Control Strength", default=2.0, min=0.0, soft_max=5.0)
    end_behavior: bpy.props.EnumProperty(
        name="At Path End",
        items=((exchange.END_STOP, "Stop", "Stop at the end"), (exchange.END_PINGPONG, "Ping-Pong", "Walk back and forth (demo)")),
        default=exchange.END_STOP,
    )

    style: bpy.props.EnumProperty(name="Style", items=main_style_items)
    idle_style: bpy.props.EnumProperty(name="Idle Style", items=idle_items, description="Used when nearly standing still")
    style_keys: bpy.props.CollectionProperty(type=AI4A_StyleKey)
    style_key_index: bpy.props.IntProperty()
    custom_styles: bpy.props.CollectionProperty(type=AI4A_CustomStyle)
    custom_style_name: bpy.props.StringProperty(name="Name", default="MyPose")

    use_scene_range: bpy.props.BoolProperty(name="Scene Frame Range", default=True)
    frame_start: bpy.props.IntProperty(name="Start", default=1)
    frame_end: bpy.props.IntProperty(name="End", default=250)
    start_from_pose: bpy.props.BoolProperty(
        name="Start From Current Pose", default=False,
        description="Initialize from the armature pose at the start frame instead of the rest pose at the path start",
    )
    location_mode: bpy.props.EnumProperty(
        name="Bone Locations",
        items=(
            (rig.LOCATION_ALL, "All Bones (exact)", "Key location on every model bone: reproduces model positions exactly"),
            (rig.LOCATION_ROOT, "Root Only", "Rotation-only bones below the root: clean rig, ~1 cm drift"),
        ),
        default=rig.LOCATION_ALL,
    )
    prediction_fps: bpy.props.FloatProperty(name="Prediction Rate", default=10.0, min=1.0, max=60.0)
    network_iterations: bpy.props.IntProperty(name="Network Iterations", default=3, min=1, max=10)
    action_name: bpy.props.StringProperty(name="Action", default="AI4A_Motion")
    create_helpers: bpy.props.BoolProperty(name="Create Path/Goal Helpers", default=True)
    store_contacts: bpy.props.BoolProperty(name="Store Foot Contacts", default=True)

    status: bpy.props.StringProperty(options={"HIDDEN"})


CLASSES = (AI4A_CustomStyle, AI4A_StyleKey, AI4A_Settings)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.ai4a = bpy.props.PointerProperty(type=AI4A_Settings)


def unregister():
    del bpy.types.Scene.ai4a
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
