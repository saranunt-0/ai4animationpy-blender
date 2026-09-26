# Copyright (c) Meta Platforms, Inc. and affiliates.
"""The only module that reads/writes Blender data (bpy).

It turns Blender objects into plain arrays (middleware.rig.ArmatureSnapshot,
polylines, boxes) and writes baked results back as F-curves. All maths lives
in the middleware.
"""

import bpy
import numpy as np
from mathutils import Vector
from mathutils.geometry import interpolate_bezier

from .middleware import conventions as cv
from .middleware.rig import ArmatureSnapshot, Calibration

CALIBRATION_KEY = "ai4a_calibration"


def meters_per_unit(scene):
    return float(scene.unit_settings.scale_length) or 1.0


# ----------------------------------------------------------------------------
# Armature
# ----------------------------------------------------------------------------


def snapshot_armature(obj, scene=None):
    """Plain-array copy of the armature's current evaluated state."""
    if obj is None or obj.type != "ARMATURE":
        raise ValueError("Select an armature object.")
    scene = scene or bpy.context.scene
    bones = obj.data.bones
    names = [b.name for b in bones]
    index = {n: i for i, n in enumerate(names)}
    parents = [index[b.parent.name] if b.parent else -1 for b in bones]
    rest = np.array([np.array(b.matrix_local) for b in bones])
    pose = np.array([np.array(obj.pose.bones[n].matrix) for n in names])
    basis = np.array([np.array(obj.pose.bones[n].matrix_basis) for n in names])
    return ArmatureSnapshot(
        bone_names=names,
        parent_indices=parents,
        rest_matrices=rest,
        armature_world=np.array(obj.matrix_world),
        meters_per_unit=meters_per_unit(scene),
        pose_matrices=pose,
        basis_matrices=basis,
    )


def load_calibration(obj):
    text = obj.get(CALIBRATION_KEY) if obj is not None else None
    if not text:
        return None
    return Calibration.from_json(text)


def store_calibration(obj, calibration):
    obj[CALIBRATION_KEY] = calibration.to_json()


def check_bone_setup(obj, calibration):
    """Settings under which the middleware's FK rule no longer matches Blender."""
    warnings = []
    names = set(calibration.blender_bone_names)
    for pb in obj.pose.bones:
        chain_relevant = pb.name in names or any(c.name in names for c in pb.children_recursive)
        if not chain_relevant:
            continue
        bone = pb.bone
        if any(not c.mute for c in pb.constraints):
            warnings.append("%s has constraints (result will differ)" % pb.name)
        if not bone.use_inherit_rotation:
            warnings.append("%s does not inherit rotation" % pb.name)
        if bone.inherit_scale not in {"FULL", "NONE", "NONE_LEGACY"}:
            warnings.append("%s uses inherit_scale=%s" % (pb.name, bone.inherit_scale))
        if not bone.use_local_location:
            warnings.append("%s has local location disabled" % pb.name)
    return warnings


def connected_bones(obj, calibration):
    return [n for n in calibration.blender_bone_names if obj.data.bones[n].use_connect]


# ----------------------------------------------------------------------------
# Scene geometry
# ----------------------------------------------------------------------------


def curve_world_points(obj, resolution=None):
    """First spline of a curve object as a world-space polyline (N, 3)."""
    if obj is None or obj.type != "CURVE" or not obj.data.splines:
        raise ValueError("Path object must be a curve with at least one spline.")
    spline = obj.data.splines[0]
    mw = obj.matrix_world
    points = []
    if spline.type == "BEZIER":
        bp = list(spline.bezier_points)
        res = int(resolution or spline.resolution_u or 12)
        pairs = list(zip(bp[:-1], bp[1:]))
        if spline.use_cyclic_u and len(bp) > 1:
            pairs.append((bp[-1], bp[0]))
        for a, b in pairs:
            seg = interpolate_bezier(a.co, a.handle_right, b.handle_left, b.co, res + 1)
            points.extend(seg[:-1])
        points.append(pairs[-1][1].co if pairs else bp[0].co)
    else:
        pts = [Vector(p.co[:3]) for p in spline.points]
        if spline.type == "NURBS":
            depsgraph = bpy.context.evaluated_depsgraph_get()
            mesh = obj.evaluated_get(depsgraph).to_mesh()
            pts = [v.co.copy() for v in mesh.vertices]
            obj.evaluated_get(depsgraph).to_mesh_clear()
        if spline.use_cyclic_u and pts:
            pts.append(pts[0])
        points = pts
    return np.array([np.array(mw @ Vector(p)) for p in points], dtype=float)


def object_world_corners(obj):
    return np.array([np.array(obj.matrix_world @ Vector(c)) for c in obj.bound_box], dtype=float)


def sample_world_matrices(scene, obj, frames):
    return sample_world_matrices_many(scene, [obj], frames)[0]


def sample_world_matrices_many(scene, objects, frames):
    """World matrices (F, 4, 4) of several objects in ONE pass over the frames.

    frame_set evaluates the whole scene (including the skinned character), so
    sampling every object separately would multiply the cost. None entries
    yield identity matrices (e.g. a joystick knob without a gate parent).
    """
    current = scene.frame_current
    out = [[] for _ in objects]
    try:
        for f in frames:
            scene.frame_set(int(f))
            for i, obj in enumerate(objects):
                out[i].append(np.eye(4) if obj is None else np.array(obj.matrix_world))
    finally:
        scene.frame_set(current)
    return [np.array(m) for m in out]


def sample_property(owner, prop, frames, id_data=None):
    """Per-frame values of a (possibly animated) property without frame_set."""
    id_data = id_data or owner.id_data
    value = float(getattr(owner, prop))
    anim = getattr(id_data, "animation_data", None)
    if anim is None or anim.action is None:
        return np.full(len(frames), value)
    path = owner.path_from_id(prop)
    fcurve = _find_fcurve(id_data, anim.action, path, 0)
    if fcurve is None:
        return np.full(len(frames), value)
    return np.array([fcurve.evaluate(float(f)) for f in frames])


def _find_fcurve(id_data, action, data_path, index):
    for fc in _action_fcurves(id_data, action):
        if fc.data_path == data_path and fc.array_index == index:
            return fc
    return None


def _action_fcurves(id_data, action):
    if hasattr(action, "layers") and len(action.layers):
        slot = id_data.animation_data.action_slot if id_data.animation_data else None
        for layer in action.layers:
            for strip in layer.strips:
                bag = strip.channelbag(slot) if slot is not None else None
                if bag is not None:
                    return list(bag.fcurves)
        return []
    return list(getattr(action, "fcurves", []))


# ----------------------------------------------------------------------------
# Baking
# ----------------------------------------------------------------------------


def _new_action_fcurves(obj, name):
    """Create an action on obj and return (fcurves collection, group factory).

    Blender 4.4+ uses slotted/layered actions; older versions expose
    action.fcurves directly.
    """
    action = bpy.data.actions.new(name)
    anim = obj.animation_data or obj.animation_data_create()
    anim.action = action
    if hasattr(action, "slots") and hasattr(action, "layers"):
        slot = anim.action_slot
        if slot is None:
            slot = action.slots.new(id_type="OBJECT", name=obj.name)
            anim.action_slot = slot
        layer = action.layers[0] if len(action.layers) else action.layers.new("Layer")
        strip = layer.strips[0] if len(layer.strips) else layer.strips.new(type="KEYFRAME")
        bag = strip.channelbag(slot, ensure=True)
        return action, bag.fcurves, (lambda group_name: bag.groups.new(group_name))
    return action, action.fcurves, (lambda group_name: action.groups.new(group_name))


def _write_curve(fcurves, group, data_path, index, frames, values):
    fc = fcurves.new(data_path, index=index)
    fc.group = group
    count = len(frames)
    fc.keyframe_points.add(count)
    co = np.empty(2 * count, dtype=np.float32)
    co[0::2] = frames
    co[1::2] = values
    fc.keyframe_points.foreach_set("co", co)
    for kp in fc.keyframe_points:
        kp.interpolation = "LINEAR"
    fc.update()
    return fc


def bake_basis(obj, basis, frame_numbers, bone_names, action_name, constant_bones=()):
    """Write matrix_basis sequences as loc/quat F-curves on a new action.

    basis: (F, Nb, 4, 4) for ALL bones in snapshot order (bone_names).
    Bones in constant_bones get a single key (unmapped bones keep their pose).
    """
    frames = np.asarray(frame_numbers, dtype=np.float32)
    action, fcurves, new_group = _new_action_fcurves(obj, action_name)
    loc = basis[..., :3, 3]
    quat = cv.make_quaternions_continuous(cv.quaternions_from_matrices(basis[..., :3, :3]), axis=0)
    for i, name in enumerate(bone_names):
        pb = obj.pose.bones[name]
        pb.rotation_mode = "QUATERNION"
        group = new_group(name)
        prefix = 'pose.bones["%s"].' % bpy.utils.escape_identifier(name)
        if name in constant_bones:
            f, l, q = frames[:1], loc[:1, i], quat[:1, i]
        else:
            f, l, q = frames, loc[:, i], quat[:, i]
        for k in range(3):
            _write_curve(fcurves, group, prefix + "location", k, f, l[:, k])
        for k in range(4):
            _write_curve(fcurves, group, prefix + "rotation_quaternion", k, f, q[:, k])
        pb.scale = (1.0, 1.0, 1.0)
    return action


def verify_bake(scene, obj, calibration, expected_ai4a, frame_numbers, samples=3):
    """Re-evaluate Blender at a few frames and compare against the model data.

    Returns (max position error in meters, max rotation error in degrees).
    """
    count = len(frame_numbers)
    picks = sorted({0, count // 2, count - 1})[:samples]
    current = scene.frame_current
    pos_err, rot_err = 0.0, 0.0
    try:
        for k in picks:
            scene.frame_set(int(frame_numbers[k]))
            snap = snapshot_armature(obj, scene)
            got = calibration.to_ai4a(snap)
            want = cv.orthonormalize_transforms(expected_ai4a[k])
            pos_err = max(pos_err, float(np.abs(got[:, :3, 3] - want[:, :3, 3]).max()))
            rot_err = max(rot_err, float(cv.rotation_angle_deg(got[:, :3, :3], want[:, :3, :3]).max()))
    finally:
        scene.frame_set(current)
    return pos_err, rot_err


# ----------------------------------------------------------------------------
# Helper objects
# ----------------------------------------------------------------------------


def helper_collection(scene, name="AI4A Helpers"):
    coll = bpy.data.collections.get(name)
    if coll is None:
        coll = bpy.data.collections.new(name)
    if coll.name not in scene.collection.children:
        scene.collection.children.link(coll)
    return coll


def create_polyline(scene, name, points_world, collection=None):
    data = bpy.data.curves.get(name) or bpy.data.curves.new(name, type="CURVE")
    data.dimensions = "3D"
    data.splines.clear()
    spline = data.splines.new("POLY")
    spline.points.add(max(0, len(points_world) - 1))
    for p, co in zip(spline.points, points_world):
        p.co = (float(co[0]), float(co[1]), float(co[2]), 1.0)
    obj = bpy.data.objects.get(name)
    if obj is None:
        obj = bpy.data.objects.new(name, data)
        (collection or scene.collection).objects.link(obj)
    obj.data = data
    return obj


def animate_empty(scene, name, world_matrices, frame_numbers, collection=None, display="ARROWS"):
    obj = bpy.data.objects.get(name)
    if obj is None:
        obj = bpy.data.objects.new(name, None)
        obj.empty_display_type = display
        obj.empty_display_size = 0.3
        (collection or scene.collection).objects.link(obj)
    if obj.animation_data and obj.animation_data.action:
        old = obj.animation_data.action
        obj.animation_data.action = None
        if old.users == 0:
            bpy.data.actions.remove(old)
    obj.rotation_mode = "QUATERNION"
    action, fcurves, new_group = _new_action_fcurves(obj, name + "_Action")
    group = new_group("Transform")
    frames = np.asarray(frame_numbers, dtype=np.float32)
    quat = cv.make_quaternions_continuous(cv.quaternions_from_matrices(world_matrices[:, :3, :3]), axis=0)
    for k in range(3):
        _write_curve(fcurves, group, "location", k, frames, world_matrices[:, k, 3])
    for k in range(4):
        _write_curve(fcurves, group, "rotation_quaternion", k, frames, quat[:, k])
    return obj


def store_contacts(obj, contact_names, contacts, frame_numbers):
    """Contacts as animated custom properties on the armature (drive IK etc.)."""
    frames = np.asarray(frame_numbers, dtype=np.float32)
    fcurves = _fcurves_of_current_action(obj)
    for c, name in enumerate(contact_names):
        key = "ai4a_contact_%s" % name
        obj[key] = float(contacts[0, c])
        fc = fcurves.new('["%s"]' % key, index=0)
        fc.keyframe_points.add(len(frames))
        co = np.empty(2 * len(frames), dtype=np.float32)
        co[0::2] = frames
        co[1::2] = contacts[:, c]
        fc.keyframe_points.foreach_set("co", co)
        fc.update()


def _fcurves_of_current_action(obj):
    anim = obj.animation_data
    action = anim.action
    if hasattr(action, "layers") and len(action.layers):
        strip = action.layers[0].strips[0]
        return strip.channelbag(anim.action_slot, ensure=True).fcurves
    return action.fcurves

