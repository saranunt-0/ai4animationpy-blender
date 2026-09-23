# Copyright (c) Meta Platforms, Inc. and affiliates.
"""AI4Animation for Blender.

Author a path (curve, voxel planner or animated target) and a style in
Blender, run the AI4Animation Authoring model in its own Python environment,
and bake the result onto the Geno armature.

The Blender <-> model conversion lives in `middleware` (pure NumPy) so it can
be tested without Blender. Nothing here imports `bpy` at module level, so the
runner (model environment) can import `ai4animation_blender.middleware`.
"""

bl_info = {
    "name": "AI4Animation",
    "author": "AI4AnimationPy contributors",
    "version": (0, 1, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar > AI4A",
    "description": "Path-guided neural character animation with AI4AnimationPy",
    "category": "Animation",
}


def register():
    from . import operators, preferences, properties, ui

    preferences.register()
    properties.register()
    operators.register()
    ui.register()


def unregister():
    from . import operators, preferences, properties, ui

    ui.unregister()
    operators.unregister()
    properties.unregister()
    preferences.unregister()
