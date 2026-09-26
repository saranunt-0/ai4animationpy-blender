# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Synthetic Blender armatures that reproduce real importer quirks.

The model skeleton is the real Geno profile. The Blender side gets:
* random per-bone frame offsets (bone roll / "Y along bone" orientation),
* an armature object transform with rotation, translation and 0.01 scale
  (FBX-style unit scale),
* optionally a rest pose that differs from the reference pose by a rigid
  rotation, compensated on the root bone's basis (glTF-style),
* unmapped bones above, between and below mapped bones.
"""

import numpy as np

from ai4animation_blender.middleware import conventions as cv


def random_rotation(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    return cv.matrices_from_quaternions(q)


def rotation_about(axis, degrees):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    half = np.radians(degrees) / 2
    return cv.matrices_from_quaternions(np.r_[np.cos(half), np.sin(half) * axis])


def tr(rotation, position):
    m = np.eye(4)
    m[:3, :3] = rotation
    m[:3, 3] = position
    return m


class SyntheticRig:
    def __init__(self, profile, seed=0, scale=0.01, rest_rotated=True, meters_per_unit=1.0):
        rng = np.random.default_rng(seed)
        self.profile = profile
        self.meters_per_unit = meters_per_unit
        names = profile.bone_names
        # Blender bones are orthonormal; the profile is float32 (errors ~1e-6).
        ref = cv.orthonormalize_transforms(profile.reference_transforms)

        # Blender bone list: extra unmapped bones around the mapped ones.
        blender = ["Root"] + list(names)
        parents_by_name = {"Root": None}
        for n, p in zip(names, profile.parent_names):
            parents_by_name[n] = p if p is not None else "Root"
        # Neck1 between Neck and Head (exists in the real Geno file)
        blender.insert(blender.index("Head"), "Neck1")
        parents_by_name["Neck1"] = "Neck"
        parents_by_name["Head"] = "Neck1"
        blender += ["LeftHandIndex1", "HeadTop_End"]
        parents_by_name["LeftHandIndex1"] = "LeftHand"
        parents_by_name["HeadTop_End"] = "Head"
        # shuffle storage order to exercise topological sorting
        order = rng.permutation(len(blender))
        self.bone_names = [blender[i] for i in order]
        self.parent_indices = [
            self.bone_names.index(parents_by_name[n]) if parents_by_name[n] else -1
            for n in self.bone_names
        ]

        # Armature object: rotation + translation + uniform scale.
        self.armature_rotation = random_rotation(rng)
        self.armature_world = np.eye(4)
        self.armature_world[:3, :3] = self.armature_rotation * scale
        self.armature_world[:3, 3] = rng.normal(size=3)

        # Per-bone frame offsets: A_R = C(B)_R @ offset
        self.offsets = {n: random_rotation(rng) for n in self.bone_names}

        # Reference pose in Blender world (what the importer shows).
        world_ref = {}
        for k, n in enumerate(names):
            a = ref[k]
            world_ref[n] = self.ai4a_to_blender_world(a, n)
        neck, head = world_ref["Neck"], world_ref["Head"]
        world_ref["Neck1"] = tr(neck[:3, :3], 0.5 * (neck[:3, 3] + head[:3, 3]))
        world_ref["Root"] = tr(np.eye(3), np.zeros(3))
        hand = world_ref["LeftHand"]
        world_ref["LeftHandIndex1"] = hand @ tr(random_rotation(rng), [0.0, 0.1 / meters_per_unit, 0.0])
        world_ref["HeadTop_End"] = head @ tr(np.eye(3), [0.0, 0.15 / meters_per_unit, 0.0])
        self.reference_pose = np.stack([self.world_to_armature(world_ref[n]) for n in self.bone_names])

        # Rest pose: glTF-style rigid rotation of the whole skeleton about Root.
        spin = tr(rotation_about([1, 0, 0], 90.0), np.zeros(3)) if rest_rotated else np.eye(4)
        self.rest = np.einsum("ij,bjk->bik", spin, self.reference_pose)
        self.reference_basis = self.basis_from_pose(self.reference_pose)

    # ------------------------------------------------------------------
    def ai4a_to_blender_world(self, a, name):
        """Ground truth inverse mapping used to author the synthetic rig."""
        m = np.eye(4)
        m[:3, :3] = cv.AI4A_TO_BLENDER @ a[:3, :3] @ self.offsets[name].T
        m[:3, 3] = cv.AI4A_TO_BLENDER @ a[:3, 3] / self.meters_per_unit
        return m

    def world_to_armature(self, world):
        local = np.linalg.inv(self.armature_world) @ world
        return cv.rigid(local)

    def rest_relative(self, i):
        p = self.parent_indices[i]
        return self.rest[i] if p < 0 else np.linalg.inv(self.rest[p]) @ self.rest[i]

    def basis_from_pose(self, pose):
        basis = np.zeros_like(pose)
        for i in range(len(self.bone_names)):
            p = self.parent_indices[i]
            parent = np.eye(4) if p < 0 else pose[p]
            basis[i] = np.linalg.inv(self.rest_relative(i)) @ np.linalg.inv(parent) @ pose[i]
        return basis

    def fk(self, basis):
        """Blender's own pose evaluation: pose = parent_pose @ rest_rel @ basis."""
        pose = np.zeros_like(basis)
        done = set()

        def visit(i):
            if i in done:
                return
            p = self.parent_indices[i]
            if p >= 0:
                visit(p)
            parent = np.eye(4) if p < 0 else pose[p]
            pose[i] = parent @ self.rest_relative(i) @ basis[i]
            done.add(i)

        for i in range(len(self.bone_names)):
            visit(i)
        return pose

    def snapshot(self, pose=None):
        from ai4animation_blender.middleware.rig import ArmatureSnapshot

        pose = self.reference_pose if pose is None else pose
        return ArmatureSnapshot(
            bone_names=list(self.bone_names),
            parent_indices=list(self.parent_indices),
            rest_matrices=self.rest.copy(),
            armature_world=self.armature_world.copy(),
            meters_per_unit=self.meters_per_unit,
            pose_matrices=pose.copy(),
            basis_matrices=self.basis_from_pose(pose),
        )


def random_model_poses(profile, frames, seed=1, noise_deg=40.0, orthonormal=True):
    """Plausible-ish model output: reference pose with random bone rotations/offsets."""
    rng = np.random.default_rng(seed)
    ref = cv.orthonormalize_transforms(profile.reference_transforms)
    out = np.repeat(ref[None], frames, 0).copy()
    for f in range(frames):
        root = tr(rotation_about([0, 1, 0], rng.uniform(-180, 180)), [rng.uniform(-5, 5), 0, rng.uniform(-5, 5)])
        for b in range(ref.shape[0]):
            axis = rng.normal(size=3)
            out[f, b, :3, :3] = rotation_about(axis, rng.uniform(-noise_deg, noise_deg)) @ ref[b, :3, :3]
            out[f, b, :3, 3] = ref[b, :3, 3] + rng.normal(scale=0.02, size=3)
        out[f] = np.einsum("ij,bjk->bik", root, out[f])
    if not orthonormal:
        # Rotation.Look(z, y) style: y not perpendicular to z
        y = out[..., :3, 1] + 0.2 * out[..., :3, 2]
        y /= np.linalg.norm(y, axis=-1, keepdims=True)
        out[..., :3, 1] = y
        out[..., :3, 0] = np.cross(y, out[..., :3, 2])
    return out
