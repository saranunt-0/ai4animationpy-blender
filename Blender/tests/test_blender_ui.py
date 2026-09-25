# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Headless UI smoke test: every panel draw() only references real properties/operators."""

import sys
import types
from pathlib import Path

import pytest

bpy = pytest.importorskip("bpy")

REPO = Path(__file__).resolve().parents[2]
ADDON = "ai4animation_blender"


class OperatorProps:
    def __init__(self, idname):
        module, name = idname.split(".")
        op = getattr(getattr(bpy.ops, module), name)
        self.__dict__["_rna"] = op.get_rna_type()

    def __setattr__(self, key, value):
        assert key in self._rna.properties, "operator has no property %r" % key


class Layout:
    calls = []

    def _child(self, *args, **kwargs):
        return Layout()

    row = column = box = split = _child

    def prop(self, data, name, **kwargs):
        assert name in data.bl_rna.properties, "%s has no property %r" % (data, name)
        Layout.calls.append(name)

    def operator(self, idname, **kwargs):
        Layout.calls.append(idname)
        return OperatorProps(idname)

    def label(self, **kwargs):
        pass

    def separator(self, **kwargs):
        pass

    def template_list(self, list_type, list_id, data, prop, active_data, active_prop, **kwargs):
        assert prop in data.bl_rna.properties and active_prop in active_data.bl_rna.properties
        assert hasattr(bpy.types, list_type)


@pytest.fixture()
def addon():
    import addon_utils

    bpy.ops.wm.read_factory_settings(use_empty=True)
    sys.path.insert(0, str(REPO / "Blender"))
    addon_utils.enable(ADDON, default_set=True)
    yield sys.modules[ADDON]
    addon_utils.disable(ADDON)


@pytest.mark.parametrize("facing", ["MOVE", "LOOK_AT", "OBJECT", "STICK"])
@pytest.mark.parametrize("mode", ["CURVE", "PLANNER", "TARGET", "STICK"])
def test_panels_draw(addon, mode, facing):
    from ai4animation_blender import ui

    scene = bpy.context.scene
    scene.ai4a.path_mode = mode
    scene.ai4a.facing_mode = facing
    scene.ai4a.use_scene_range = False
    scene.ai4a.status = "status"
    arm = bpy.data.objects.new("Arm", bpy.data.armatures.new("Arm"))
    scene.collection.objects.link(arm)
    scene.ai4a.armature = arm
    scene.ai4a.custom_styles.add().name = "Mine"
    scene.ai4a.style_keys.add()
    Layout.calls = []
    for panel in (ui.AI4A_PT_main, ui.AI4A_PT_path, ui.AI4A_PT_style, ui.AI4A_PT_generate):
        panel.draw(types.SimpleNamespace(layout=Layout()), bpy.context)
    assert "ai4a.generate" in Layout.calls


def test_style_enum_lists_shipped_and_custom_styles(addon):
    from ai4animation_blender.properties import style_items

    scene = bpy.context.scene
    scene.ai4a.custom_styles.add().name = "Mine"
    names = [i[0] for i in style_items(scene.ai4a, bpy.context)]
    assert "Neutral" in names and "Idle" in names and "custom:Mine" in names
    # dynamic enums default to their first item: idle must default to "Idle"
    assert scene.ai4a.idle_style == "Idle"


class _PrefsProxy:
    """AddonPreferences.draw reads self.layout; forward everything else."""

    def __init__(self, prefs):
        self.layout = Layout()
        self._prefs = prefs

    def __getattr__(self, name):
        return getattr(self._prefs, name)


def test_preferences_draw(addon):
    prefs = bpy.context.preferences.addons[ADDON].preferences
    type(prefs).draw(_PrefsProxy(prefs), bpy.context)


def test_keyframed_walk_speed_is_sampled_per_frame(addon):
    from ai4animation_blender import blender_io as bio

    scene = bpy.context.scene
    s = scene.ai4a
    s.walk_speed = 0.0
    s.keyframe_insert("walk_speed", frame=1)
    s.walk_speed = 2.0
    s.keyframe_insert("walk_speed", frame=11)
    for fc in bio._action_fcurves(scene, scene.animation_data.action):
        for kp in fc.keyframe_points:
            kp.interpolation = "LINEAR"
    speeds = bio.sample_property(s, "walk_speed", list(range(1, 12)), id_data=scene)
    assert abs(speeds[0]) < 1e-6 and abs(speeds[5] - 1.0) < 1e-6 and abs(speeds[-1] - 2.0) < 1e-6
