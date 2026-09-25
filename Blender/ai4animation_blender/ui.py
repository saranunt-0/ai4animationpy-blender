# Copyright (c) Meta Platforms, Inc. and affiliates.
from pathlib import Path

import bpy

from . import blender_io as bio
from . import pipeline, preferences


class AI4A_UL_style_keys(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row(align=True)
        row.prop(item, "frame", text="", emboss=False)
        row.prop(item, "style", text="")


class _Panel:
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "AI4A"


class AI4A_PT_main(_Panel, bpy.types.Panel):
    bl_label = "AI4Animation"

    def draw(self, context):
        layout = self.layout
        s = context.scene.ai4a
        prefs = preferences.get(context)

        box = layout.box()
        repo_ok = bool(prefs.repo_path) and (Path(bpy.path.abspath(prefs.repo_path)) / "ai4animation").is_dir()
        py_ok = bool(prefs.python_path) and Path(bpy.path.abspath(prefs.python_path)).is_file()
        box.label(text="Repository", icon="CHECKMARK" if repo_ok else "ERROR")
        box.label(text="Model Python", icon="CHECKMARK" if py_ok else "ERROR")
        if not (repo_ok and py_ok):
            box.label(text="Set both in Preferences > Add-ons > AI4Animation")
        box.operator("ai4a.refresh_profile", icon="FILE_REFRESH")

        col = layout.column(align=True)
        col.prop(s, "armature")
        row = col.row(align=True)
        row.operator("ai4a.import_rig", text="Import Geno (glb)", icon="IMPORT").file_format = "GLB"
        row.operator("ai4a.import_rig", text="(fbx)").file_format = "FBX"
        row = col.row(align=True)
        row.operator("ai4a.calibrate", icon="ARMATURE_DATA")
        calibration = None
        if s.armature is not None:
            try:
                calibration = bio.load_calibration(s.armature)
            except ValueError:
                calibration = None
        if calibration is not None:
            col.label(text="Calibrated (%.2f mm)" % (1000 * calibration.max_residual), icon="CHECKMARK")
        elif s.armature is not None:
            col.label(text="Not calibrated", icon="ERROR")


class AI4A_PT_path(_Panel, bpy.types.Panel):
    bl_label = "Control"
    bl_parent_id = "AI4A_PT_main"

    def draw(self, context):
        layout = self.layout
        s = context.scene.ai4a
        layout.label(text="Movement")
        layout.prop(s, "path_mode", expand=True)
        col = layout.column()
        if s.path_mode == "CURVE":
            col.prop(s, "curve")
            col.prop(s, "path_spacing")
        elif s.path_mode == "PLANNER":
            col.prop(s, "start_object")
            col.prop(s, "goal_object")
            col.prop(s, "obstacles")
            sub = col.column(align=True)
            sub.prop(s, "planner_cell")
            sub.prop(s, "planner_margin")
            sub.prop(s, "planner_height")
            sub.prop(s, "planner_max_depth")
        elif s.path_mode == "TARGET":
            col.prop(s, "target_object")
        else:
            col.prop(s, "left_stick")
            col.prop(s, "start_object", text="Start (optional)")
            col.prop(s, "stick_assist")
            col.label(text="Key the knob inside its circle; circle radius = full stick", icon="INFO")
        if s.path_mode in ("CURVE", "PLANNER"):
            col.prop(s, "end_behavior")

        layout.label(text="Facing")
        layout.prop(s, "facing_mode", text="")
        if s.facing_mode == "STICK":
            layout.prop(s, "right_stick")
        elif s.facing_mode in ("LOOK_AT", "OBJECT"):
            layout.prop(s, "facing_object")
            if s.facing_mode == "OBJECT":
                layout.label(text="The object's -Y axis is the facing", icon="INFO")

        box = layout.box()
        box.prop(s, "controller")
        try:
            resolved = pipeline.resolve_controller(s)
            box.label(text="Using: %s" % ("Joystick (Biped demo)" if resolved == "STICK" else "Goal (Authoring demo)"))
        except pipeline.PipelineError as error:
            resolved = None
            box.label(text=str(error)[:60], icon="ERROR")
        if resolved == "STICK":
            row = box.row(align=True)
            row.prop(s, "tracking_gain")
            row.prop(s, "tracking_leash")
        elif resolved == "GOAL":
            box.prop(s, "control_strength")
        layout.operator("ai4a.setup_helpers", icon="ADD")


class AI4A_PT_style(_Panel, bpy.types.Panel):
    bl_label = "Style"
    bl_parent_id = "AI4A_PT_main"

    def draw(self, context):
        layout = self.layout
        s = context.scene.ai4a
        col = layout.column()
        col.prop(s, "style")
        col.prop(s, "idle_style")
        layout.label(text="Style changes over time:")
        row = layout.row()
        row.template_list("AI4A_UL_style_keys", "", s, "style_keys", s, "style_key_index", rows=2)
        sub = row.column(align=True)
        sub.operator("ai4a.style_key_add", icon="ADD", text="")
        sub.operator("ai4a.style_key_remove", icon="REMOVE", text="")

        box = layout.box()
        box.label(text="Custom style from the current pose")
        row = box.row(align=True)
        row.prop(s, "custom_style_name", text="")
        row.operator("ai4a.capture_style", text="Capture", icon="POSE_HLT")
        for custom in s.custom_styles:
            r = box.row(align=True)
            r.label(text=custom.name, icon="ARMATURE_DATA")
            r.operator("ai4a.remove_custom_style", text="", icon="X").name = custom.name


class AI4A_PT_generate(_Panel, bpy.types.Panel):
    bl_label = "Generate"
    bl_parent_id = "AI4A_PT_main"

    def draw(self, context):
        layout = self.layout
        s = context.scene.ai4a
        col = layout.column()
        col.prop(s, "use_scene_range")
        if not s.use_scene_range:
            row = col.row(align=True)
            row.prop(s, "frame_start")
            row.prop(s, "frame_end")
        col.prop(s, "walk_speed")
        col.prop(s, "start_from_pose")
        col.prop(s, "location_mode")
        col.prop(s, "action_name")

        adv = layout.column(heading="Advanced")
        adv.prop(s, "prediction_fps")
        adv.prop(s, "network_iterations")
        adv.prop(s, "create_helpers")
        adv.prop(s, "store_contacts")

        layout.separator()
        layout.operator("ai4a.generate", icon="PLAY")
        if s.status:
            layout.label(text=s.status)


CLASSES = (AI4A_UL_style_keys, AI4A_PT_main, AI4A_PT_path, AI4A_PT_style, AI4A_PT_generate)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
