# Copyright (c) Meta Platforms, Inc. and affiliates.
import os
from pathlib import Path

import bpy

ADDON_ID = __package__


def guess_repo():
    """The add-on lives in <repo>/Blender/ai4animation_blender when used from a checkout."""
    candidate = Path(__file__).resolve().parents[2]
    if (candidate / "ai4animation").is_dir() and (candidate / "Demos" / "Authoring").is_dir():
        return str(candidate)
    return os.environ.get("AI4A_REPO", "")


class AI4A_Preferences(bpy.types.AddonPreferences):
    bl_idname = ADDON_ID

    repo_path: bpy.props.StringProperty(
        name="AI4AnimationPy Repository",
        description="Root folder of the ai4animationpy checkout (contains ai4animation/ and Demos/)",
        subtype="DIR_PATH",
        default=guess_repo(),
    )
    python_path: bpy.props.StringProperty(
        name="Model Python",
        description=(
            "Python executable of the environment where ai4animation and torch are installed "
            "(NOT Blender's Python), e.g. ~/miniconda3/envs/ai4animation/bin/python"
        ),
        subtype="FILE_PATH",
        default=os.environ.get("AI4A_PYTHON", ""),
    )
    timeout: bpy.props.IntProperty(
        name="Timeout (s)", default=1800, min=10, description="Abort the model run after this many seconds"
    )
    keep_files: bpy.props.BoolProperty(
        name="Keep Exchange Files",
        default=False,
        description="Keep request/result .npz files in the temp folder for debugging",
    )

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "repo_path")
        layout.prop(self, "python_path")
        row = layout.row()
        row.prop(self, "timeout")
        row.prop(self, "keep_files")
        layout.operator("ai4a.refresh_profile", icon="FILE_REFRESH")


def get(context=None):
    context = context or bpy.context
    return context.preferences.addons[ADDON_ID].preferences


def register():
    bpy.utils.register_class(AI4A_Preferences)


def unregister():
    bpy.utils.unregister_class(AI4A_Preferences)
