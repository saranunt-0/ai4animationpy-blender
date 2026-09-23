# Copyright (c) Meta Platforms, Inc. and affiliates.
import json
import queue
import shutil
import tempfile
import threading
import time
from pathlib import Path

import bpy
import numpy as np

from . import blender_io as bio
from . import pipeline, preferences
from .middleware import features, rig
from .properties import get_profile


def _report_warnings(op, warnings):
    for w in warnings:
        op.report({"WARNING"}, w)


def _calibrate(context, arm, source):
    profile = get_profile(context.scene)
    calibration = rig.calibrate(bio.snapshot_armature(arm, context.scene), profile, source=source)
    bio.store_calibration(arm, calibration)
    return calibration


class AI4A_OT_import_rig(bpy.types.Operator):
    """Import the Geno model used by the network and calibrate it"""

    bl_idname = "ai4a.import_rig"
    bl_label = "Import Geno Rig"
    bl_options = {"REGISTER", "UNDO"}

    file_format: bpy.props.EnumProperty(
        items=(("GLB", "glTF (.glb)", ""), ("FBX", "FBX (.fbx)", "")), default="GLB"
    )

    def execute(self, context):
        repo = pipeline.bpy_path(preferences.get(context).repo_path)
        name = "Model.glb" if self.file_format == "GLB" else "Model.fbx"
        path = Path(repo) / "Demos" / "_ASSETS_" / "Geno" / name
        if not path.is_file():
            self.report({"ERROR"}, "Not found: %s (set the repository in the add-on preferences)" % path)
            return {"CANCELLED"}
        before = set(bpy.data.objects)
        if self.file_format == "GLB":
            bpy.ops.import_scene.gltf(filepath=str(path))
        else:
            bpy.ops.import_scene.fbx(filepath=str(path))
        new = [o for o in bpy.data.objects if o not in before and o.type == "ARMATURE"]
        if not new:
            self.report({"ERROR"}, "No armature found in %s" % path.name)
            return {"CANCELLED"}
        arm = new[0]
        context.view_layer.update()
        try:
            calibration = _calibrate(context, arm, "pose")
        except ValueError as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        context.scene.ai4a.armature = arm
        self.report({"INFO"}, "Imported and calibrated %s (max mismatch %.3f mm)" % (arm.name, 1000 * calibration.max_residual))
        return {"FINISHED"}


class AI4A_OT_calibrate(bpy.types.Operator):
    """Store how this armature's bones map to the model's bones.
    Run it while the armature shows the model's reference pose (right after import)"""

    bl_idname = "ai4a.calibrate"
    bl_label = "Calibrate"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        arm = context.scene.ai4a.armature or context.active_object
        if arm is None or arm.type != "ARMATURE":
            self.report({"ERROR"}, "Pick an armature first.")
            return {"CANCELLED"}
        errors = []
        for source in ("pose", "rest"):
            try:
                calibration = _calibrate(context, arm, source)
            except ValueError as error:
                errors.append("%s: %s" % (source, error))
                continue
            context.scene.ai4a.armature = arm
            self.report({"INFO"}, "Calibrated from %s pose (max mismatch %.3f mm)" % (source, 1000 * calibration.max_residual))
            return {"FINISHED"}
        self.report({"ERROR"}, " | ".join(errors))
        return {"CANCELLED"}


class AI4A_OT_refresh_profile(bpy.types.Operator):
    """Run the model environment once: checks the setup and reads bones, reference pose and styles"""

    bl_idname = "ai4a.refresh_profile"
    bl_label = "Test Model Environment"

    def execute(self, context):
        try:
            profile = pipeline.refresh_profile(context)
        except Exception as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        self.report({"INFO"}, "Model environment OK: %d bones, %d styles" % (profile.bone_count, len(profile.guidance_names)))
        return {"FINISHED"}


class AI4A_OT_capture_style(bpy.types.Operator):
    """Capture the armature's current pose as a guidance style (bone positions relative to the character root)"""

    bl_idname = "ai4a.capture_style"
    bl_label = "Capture Pose as Style"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        settings = context.scene.ai4a
        try:
            arm, calibration = pipeline.require_calibration(settings)
        except pipeline.PipelineError as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        profile = get_profile(context.scene)
        pose = calibration.to_ai4a(bio.snapshot_armature(arm, context.scene))
        positions = features.guidance_from_pose(pose, profile)
        name = settings.custom_style_name.strip() or "Custom"
        existing = {c.name: c for c in settings.custom_styles}
        item = existing.get(name) or settings.custom_styles.add()
        item.name = name
        item.data = json.dumps(np.round(positions, 6).tolist())
        self.report({"INFO"}, "Captured style '%s'" % name)
        return {"FINISHED"}


class AI4A_OT_remove_custom_style(bpy.types.Operator):
    bl_idname = "ai4a.remove_custom_style"
    bl_label = "Remove Custom Style"
    bl_options = {"REGISTER", "UNDO"}

    name: bpy.props.StringProperty()

    def execute(self, context):
        styles = context.scene.ai4a.custom_styles
        for i, c in enumerate(styles):
            if c.name == self.name:
                styles.remove(i)
                break
        return {"FINISHED"}


class AI4A_OT_style_key_add(bpy.types.Operator):
    """Switch to another style from the current frame on"""

    bl_idname = "ai4a.style_key_add"
    bl_label = "Add Style Key"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        settings = context.scene.ai4a
        key = settings.style_keys.add()
        key.frame = context.scene.frame_current
        key.style = settings.style
        settings.style_key_index = len(settings.style_keys) - 1
        return {"FINISHED"}


class AI4A_OT_style_key_remove(bpy.types.Operator):
    bl_idname = "ai4a.style_key_remove"
    bl_label = "Remove Style Key"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        settings = context.scene.ai4a
        if 0 <= settings.style_key_index < len(settings.style_keys):
            settings.style_keys.remove(settings.style_key_index)
            settings.style_key_index = max(0, settings.style_key_index - 1)
        return {"FINISHED"}


class AI4A_OT_setup_helpers(bpy.types.Operator):
    """Create the objects the selected path mode needs (curve, start/goal, obstacles, target)"""

    bl_idname = "ai4a.setup_helpers"
    bl_label = "Create Path Helpers"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        scene = context.scene
        s = scene.ai4a
        coll = bio.helper_collection(scene, "AI4A Setup")

        def empty(name, location, display="PLAIN_AXES"):
            obj = bpy.data.objects.get(name) or bpy.data.objects.new(name, None)
            obj.empty_display_type = display
            obj.empty_display_size = 0.4
            obj.location = location
            if obj.name not in coll.objects:
                coll.objects.link(obj)
            return obj

        if s.path_mode == "CURVE" and s.curve is None:
            data = bpy.data.curves.new("AI4A_Walk", type="CURVE")
            data.dimensions = "3D"
            spline = data.splines.new("BEZIER")
            spline.bezier_points.add(2)
            for p, co in zip(spline.bezier_points, ((0, 0, 0), (3, -3, 0), (6, 0, 0))):
                p.co = co
                p.handle_left_type = p.handle_right_type = "AUTO"
            obj = bpy.data.objects.new("AI4A_Walk", data)
            coll.objects.link(obj)
            s.curve = obj
        elif s.path_mode == "PLANNER":
            s.start_object = s.start_object or empty("AI4A_Start", (-4, 4, 0), "CONE")
            s.goal_object = s.goal_object or empty("AI4A_Goal_Target", (4, -4, 0), "SPHERE")
            if s.obstacles is None:
                obstacles = bpy.data.collections.new("AI4A Obstacles")
                coll.children.link(obstacles)
                bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, 0.5))
                cube = context.active_object
                cube.scale = (1.0, 2.0, 1.0)
                for c in list(cube.users_collection):
                    c.objects.unlink(cube)
                obstacles.objects.link(cube)
                s.obstacles = obstacles
        elif s.path_mode == "TARGET" and s.target_object is None:
            s.target_object = empty("AI4A_Target", (0, 0, 0), "SINGLE_ARROW")
        return {"FINISHED"}


class AI4A_OT_generate(bpy.types.Operator):
    """Run the AI4Animation model on the current setup and bake the motion onto the armature"""

    bl_idname = "ai4a.generate"
    bl_label = "Generate Animation"
    bl_options = {"REGISTER", "UNDO"}

    _timer = None
    _process = None
    _work = None
    _frames = None
    _result = None
    _log = None
    _lines = None
    _reader = None
    _started = 0.0

    def execute(self, context):
        """Blocking run (scripts, tests, F3 search without an event loop)."""
        try:
            report = pipeline.generate_blocking(context)
        except Exception as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        return self._finish_report(context, report)

    def invoke(self, context, event):
        try:
            self._work = tempfile.mkdtemp(prefix="ai4a_")
            request, self._frames = pipeline.write_request(context, self._work)
            self._result = Path(self._work) / "result.npz"
            command = pipeline.runner_command(
                context, "run", "--request", str(request), "--result", str(self._result)
            )
            self._process = pipeline.start_process(command)
        except Exception as error:
            self._cleanup(context)
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        self._log = []
        self._lines = queue.Queue()
        self._reader = threading.Thread(target=_pump, args=(self._process.stdout, self._lines), daemon=True)
        self._reader.start()
        self._started = time.time()
        self._timer = context.window_manager.event_timer_add(0.25, window=context.window)
        context.window_manager.modal_handler_add(self)
        context.scene.ai4a.status = "Running model..."
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if event.type == "ESC":
            self._process.kill()
            self._cleanup(context)
            self.report({"WARNING"}, "Cancelled")
            return {"CANCELLED"}
        if event.type != "TIMER":
            return {"PASS_THROUGH"}
        self._drain(context)
        if self._process.poll() is None:
            if time.time() - self._started > preferences.get(context).timeout:
                self._process.kill()
                self._cleanup(context)
                self.report({"ERROR"}, "Model run timed out")
                return {"CANCELLED"}
            for area in context.screen.areas if context.screen else []:
                area.tag_redraw()
            return {"RUNNING_MODAL"}

        self._reader.join(timeout=5.0)
        self._drain(context)
        code = self._process.returncode
        try:
            if code != 0:
                raise pipeline.PipelineError("Runner failed:\n" + pipeline.tail("".join(self._log)))
            report = pipeline.apply_result(context, self._result, self._frames)
        except Exception as error:
            self._cleanup(context)
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        self._cleanup(context)
        return self._finish_report(context, report)

    def _drain(self, context):
        while True:
            try:
                line = self._lines.get_nowait()
            except queue.Empty:
                return
            self._log.append(line)
            if line.startswith("[ai4a-runner] frame"):
                context.scene.ai4a.status = line.strip().replace("[ai4a-runner] ", "Model: ")

    def _finish_report(self, context, report):
        _report_warnings(self, report["warnings"])
        ok, text = pipeline.summary(report, context.scene.ai4a.location_mode)
        context.scene.ai4a.status = text
        self.report({"INFO"} if ok else {"WARNING"}, text)
        return {"FINISHED"}

    def _cleanup(self, context):
        if self._timer is not None:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
        if self._work and not preferences.get(context).keep_files:
            shutil.rmtree(self._work, ignore_errors=True)
        context.scene.ai4a.status = ""


def _pump(stream, lines):
    for line in iter(stream.readline, ""):
        lines.put(line)
    stream.close()


CLASSES = (
    AI4A_OT_import_rig,
    AI4A_OT_calibrate,
    AI4A_OT_refresh_profile,
    AI4A_OT_capture_style,
    AI4A_OT_remove_custom_style,
    AI4A_OT_style_key_add,
    AI4A_OT_style_key_remove,
    AI4A_OT_setup_helpers,
    AI4A_OT_generate,
)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
