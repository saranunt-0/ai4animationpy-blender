# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Skeleton-level conversion between a Blender armature and the AI4Animation Actor.

Why a calibration is needed
---------------------------
The model consumes, per bone, the world position plus the Z (forward) and Y
(up) axes of the bone frame. Blender and AI4Animation disagree on bone frames:

* Blender bones point along their local +Y ("head -> tail"); importers pick
  a bone roll/orientation heuristic, so local axes differ from the source file.
* The glTF importer builds the *rest* pose from the bind matrices and puts the
  armature object under a rotated parent. For the Geno rig the Blender rest
  pose lies on its back and the importer compensates with a pose-basis
  rotation on "Hips" - only the *posed* skeleton matches the model.
* The FBX importer instead scales the armature object by 0.01.

Positions survive all of this, bone axes do not. A calibration stores, per
model bone, a constant rotation Off such that

    A_world = C(B_world) @ Off        (C = axis/unit change, see conventions)

computed once from a pose where Blender and the model are known to agree
(right after importing the model's own rig file). It is placement invariant:
the armature may be moved/rotated in the scene before calibrating.
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from . import conventions as cv

PROFILE_SCHEMA = "ai4a-rig-profile/1"
CALIBRATION_SCHEMA = "ai4a-calibration/1"

LOCATION_ALL = "ALL"  # key location on every mapped bone: exact model positions
LOCATION_ROOT = "ROOT"  # key location only on the top mapped bone: rigid FK chains


# ----------------------------------------------------------------------------
# Rig profile (the model side, produced by the runner from the real Actor)
# ----------------------------------------------------------------------------


@dataclass
class RigProfile:
    bone_names: List[str]
    parent_names: List[Optional[str]]
    reference_transforms: np.ndarray  # (J, 4, 4) AI4Animation world, meters
    root_bones: Dict[str, str]  # hip, left_hip, right_hip, left_shoulder, right_shoulder, neck
    contact_bones: List[str] = field(default_factory=list)
    guidance_names: List[str] = field(default_factory=list)
    sequence: Dict[str, float] = field(default_factory=dict)
    name: str = "Geno"
    model_file: str = ""

    @property
    def bone_count(self):
        return len(self.bone_names)

    def parent_indices(self):
        index = {n: i for i, n in enumerate(self.bone_names)}
        return [index.get(p, -1) if p is not None else -1 for p in self.parent_names]

    def to_dict(self):
        return {
            "schema": PROFILE_SCHEMA,
            "name": self.name,
            "model_file": self.model_file,
            "bone_names": list(self.bone_names),
            "parent_names": list(self.parent_names),
            "reference_transforms": np.asarray(self.reference_transforms, dtype=float)
            .reshape(-1, 16)
            .tolist(),
            "root_bones": dict(self.root_bones),
            "contact_bones": list(self.contact_bones),
            "guidance_names": list(self.guidance_names),
            "sequence": dict(self.sequence),
        }

    def to_json(self):
        return json.dumps(self.to_dict(), indent=1)

    @classmethod
    def from_dict(cls, data):
        if data.get("schema") != PROFILE_SCHEMA:
            raise ValueError(
                "Unsupported rig profile schema %r (expected %r)."
                % (data.get("schema"), PROFILE_SCHEMA)
            )
        return cls(
            bone_names=list(data["bone_names"]),
            parent_names=list(data["parent_names"]),
            reference_transforms=np.array(data["reference_transforms"], dtype=float).reshape(-1, 4, 4),
            root_bones=dict(data["root_bones"]),
            contact_bones=list(data.get("contact_bones", [])),
            guidance_names=list(data.get("guidance_names", [])),
            sequence=dict(data.get("sequence", {})),
            name=data.get("name", "Geno"),
            model_file=data.get("model_file", ""),
        )

    @classmethod
    def from_json(cls, text):
        return cls.from_dict(json.loads(text))


# ----------------------------------------------------------------------------
# Armature snapshot (the Blender side, plain arrays extracted from bpy)
# ----------------------------------------------------------------------------


@dataclass
class ArmatureSnapshot:
    """Plain-data copy of a Blender armature. Built by blender_io, testable offline.

    All bone matrices are in ARMATURE space, exactly as bpy reports them:
      rest_matrices  = bone.matrix_local
      pose_matrices  = pose_bone.matrix
      basis_matrices = pose_bone.matrix_basis
    """

    bone_names: List[str]
    parent_indices: List[int]
    rest_matrices: np.ndarray
    armature_world: np.ndarray
    meters_per_unit: float = 1.0
    pose_matrices: Optional[np.ndarray] = None
    basis_matrices: Optional[np.ndarray] = None

    def __post_init__(self):
        self.rest_matrices = cv.as_matrix(self.rest_matrices)
        self.armature_world = cv.as_matrix(self.armature_world)
        if self.pose_matrices is not None:
            self.pose_matrices = cv.as_matrix(self.pose_matrices)
        if self.basis_matrices is not None:
            self.basis_matrices = cv.as_matrix(self.basis_matrices)
        self.parent_indices = [int(p) for p in self.parent_indices]

    def index(self, name):
        return self.bone_names.index(name)

    def topological_order(self):
        """Bone indices ordered parents-before-children."""
        order, seen = [], set()

        def visit(i):
            if i in seen:
                return
            p = self.parent_indices[i]
            if p >= 0:
                visit(p)
            seen.add(i)
            order.append(i)

        for i in range(len(self.bone_names)):
            visit(i)
        return order

    def rest_relative(self):
        """parent.matrix_local^-1 @ bone.matrix_local (bone.matrix_local for roots)."""
        rel = self.rest_matrices.copy()
        for i, p in enumerate(self.parent_indices):
            if p >= 0:
                rel[i] = np.linalg.inv(self.rest_matrices[p]) @ self.rest_matrices[i]
        return rel


# ----------------------------------------------------------------------------
# Bone name mapping
# ----------------------------------------------------------------------------


def _normalize_name(name):
    base = name.split(":")[-1].split("|")[-1]
    return base.replace("_", "").replace(" ", "").lower()


def auto_bone_map(model_bone_names, blender_bone_names, overrides=None):
    """Map every model bone to a Blender bone name.

    Order: explicit override, exact, case-insensitive, then name without a
    namespace/prefix ("mixamorig:Hips" -> "hips"). Raises with the full list
    of problems instead of guessing.
    """
    overrides = dict(overrides or {})
    exact = set(blender_bone_names)
    lower = {}
    for b in blender_bone_names:
        lower.setdefault(b.lower(), []).append(b)
    normalized = {}
    for b in blender_bone_names:
        normalized.setdefault(_normalize_name(b), []).append(b)

    mapping, missing, ambiguous = {}, [], []
    for m in model_bone_names:
        if m in overrides:
            if overrides[m] not in exact:
                missing.append("%s (override %r not in armature)" % (m, overrides[m]))
            else:
                mapping[m] = overrides[m]
            continue
        if m in exact:
            mapping[m] = m
            continue
        for table, key in ((lower, m.lower()), (normalized, _normalize_name(m))):
            candidates = table.get(key, [])
            if len(candidates) == 1:
                mapping[m] = candidates[0]
                break
            if len(candidates) > 1:
                ambiguous.append("%s -> %s" % (m, candidates))
                break
        else:
            missing.append(m)
    if missing or ambiguous:
        raise ValueError(
            "Bone mapping failed. Missing: %s. Ambiguous: %s."
            % (missing or "none", ambiguous or "none")
        )
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("Two model bones map to the same Blender bone: %s" % mapping)
    return mapping


# ----------------------------------------------------------------------------
# Rigid alignment (placement invariance)
# ----------------------------------------------------------------------------


def fit_rigid(source, target):
    """Least-squares rotation R, translation t, scale s with target ~ s R source + t.

    Umeyama/Kabsch. Scale is returned for diagnostics only.
    """
    src = cv.as_matrix(source)
    dst = cv.as_matrix(target)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    a, b = src - mu_s, dst - mu_d
    u, sigma, vt = np.linalg.svd(b.T @ a)
    d = np.sign(np.linalg.det(u @ vt))
    fix = np.diag([1.0, 1.0, d])
    r = u @ fix @ vt
    var = np.sum(a * a)
    scale = float(np.sum(sigma * np.diag(fix)) / var) if var > 0 else 1.0
    t = mu_d - r @ mu_s
    return r, t, scale


# ----------------------------------------------------------------------------
# Calibration
# ----------------------------------------------------------------------------


@dataclass
class Calibration:
    model_bone_names: List[str]
    blender_bone_names: List[str]  # mapped bones, same order as model_bone_names
    offsets: np.ndarray  # (J, 3, 3): A_R = C(B)_R @ offsets
    reference_basis: Dict[str, np.ndarray]  # blender bone -> 4x4 basis at calibration
    position_residuals: np.ndarray  # (J,) meters after rigid alignment
    placement: Optional[np.ndarray] = None  # 4x4 rigid map scene->reference, info only
    source: str = "pose"

    @property
    def max_residual(self):
        return float(np.max(self.position_residuals))

    def to_dict(self):
        return {
            "schema": CALIBRATION_SCHEMA,
            "model_bone_names": list(self.model_bone_names),
            "blender_bone_names": list(self.blender_bone_names),
            "offsets": np.asarray(self.offsets).reshape(-1, 9).tolist(),
            "reference_basis": {
                k: np.asarray(v).reshape(16).tolist() for k, v in self.reference_basis.items()
            },
            "position_residuals": np.asarray(self.position_residuals).tolist(),
            "placement": None if self.placement is None else np.asarray(self.placement).reshape(16).tolist(),
            "source": self.source,
        }

    def to_json(self):
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data):
        if data.get("schema") != CALIBRATION_SCHEMA:
            raise ValueError("Unsupported calibration schema %r." % data.get("schema"))
        return cls(
            model_bone_names=list(data["model_bone_names"]),
            blender_bone_names=list(data["blender_bone_names"]),
            offsets=np.array(data["offsets"], dtype=float).reshape(-1, 3, 3),
            reference_basis={
                k: np.array(v, dtype=float).reshape(4, 4) for k, v in data["reference_basis"].items()
            },
            position_residuals=np.array(data["position_residuals"], dtype=float),
            placement=None if data.get("placement") is None else np.array(data["placement"], dtype=float).reshape(4, 4),
            source=data.get("source", "pose"),
        )

    @classmethod
    def from_json(cls, text):
        return cls.from_dict(json.loads(text))

    # ------------------------------------------------------------------
    # Blender -> AI4Animation
    # ------------------------------------------------------------------

    def mapped_indices(self, snapshot):
        return [snapshot.index(n) for n in self.blender_bone_names]

    def blender_world(self, snapshot, armature_space):
        """Armature-space bone matrices -> rigid Blender world matrices (mapped bones)."""
        pose = cv.as_matrix(armature_space)[..., self.mapped_indices(snapshot), :, :]
        world = np.einsum("ij,...bjk->...bik", snapshot.armature_world, pose)
        return cv.rigid(world)

    def to_ai4a(self, snapshot, armature_space=None):
        """Blender pose (armature space, all bones) -> AI4Animation world transforms (J, 4, 4)."""
        if armature_space is None:
            armature_space = snapshot.pose_matrices
        g = cv.transforms_blender_to_ai4a(
            self.blender_world(snapshot, armature_space), snapshot.meters_per_unit
        )
        g[..., :3, :3] = np.einsum("...bij,bjk->...bik", g[..., :3, :3], self.offsets)
        return g

    # ------------------------------------------------------------------
    # AI4Animation -> Blender
    # ------------------------------------------------------------------

    def to_blender_armature_space(self, snapshot, ai4a_transforms):
        """AI4Animation world (..., J, 4, 4) -> rigid armature-space targets for mapped bones."""
        a = cv.orthonormalize_transforms(ai4a_transforms)
        g = a.copy()
        g[..., :3, :3] = np.einsum(
            "...bij,bkj->...bik", a[..., :3, :3], self.offsets
        )  # A_R @ Off^T
        world = cv.transforms_ai4a_to_blender(g, snapshot.meters_per_unit)
        local = np.einsum("ij,...bjk->...bik", np.linalg.inv(snapshot.armature_world), world)
        return cv.rigid(local)

    def to_blender_basis(self, snapshot, ai4a_transforms, location_mode=LOCATION_ALL):
        """AI4Animation world transforms -> pose_bone.matrix_basis for EVERY Blender bone.

        ai4a_transforms: (F, J, 4, 4) or (J, 4, 4).
        Returns (basis, pose): both (F, Nb, 4, 4) (or (Nb, 4, 4)), where pose is
        the armature-space result Blender will show (pose_bone.matrix).

        Assumes default bone inheritance (inherit rotation, full scale inherit,
        local location) and no constraints; blender_io verifies this.
        """
        single = np.ndim(ai4a_transforms) == 3
        targets = self.to_blender_armature_space(
            snapshot, ai4a_transforms[None] if single else ai4a_transforms
        )
        frames = targets.shape[0]
        count = len(snapshot.bone_names)
        mapped = {snapshot.index(n): k for k, n in enumerate(self.blender_bone_names)}
        rest_rel = snapshot.rest_relative()
        rest_rel_inv = np.linalg.inv(rest_rel)
        identity = np.eye(4)

        top_mapped = set()
        for i in mapped:
            p = snapshot.parent_indices[i]
            while p >= 0 and p not in mapped:
                p = snapshot.parent_indices[p]
            if p < 0:
                top_mapped.add(i)

        basis = np.zeros((frames, count, 4, 4))
        pose = np.zeros((frames, count, 4, 4))
        for i in snapshot.topological_order():
            p = snapshot.parent_indices[i]
            parent_pose = pose[:, p] if p >= 0 else np.broadcast_to(identity, (frames, 4, 4))
            ref = self.reference_basis.get(snapshot.bone_names[i], identity)
            if i in mapped:
                target = targets[:, mapped[i]]
                local = np.einsum(
                    "ij,fjk->fik", rest_rel_inv[i], np.linalg.inv(parent_pose) @ target
                )
                if location_mode == LOCATION_ALL or i in top_mapped:
                    basis[:, i] = local
                else:
                    basis[:, i] = local
                    basis[:, i, :3, 3] = ref[:3, 3]
            else:
                basis[:, i] = ref
            pose[:, i] = parent_pose @ rest_rel[i] @ basis[:, i]
        if single:
            return basis[0], pose[0]
        return basis, pose


def calibrate(snapshot, profile, bone_map=None, source="pose", tolerance=0.02):
    """Compute per-bone frame offsets from a pose where Blender == model reference.

    snapshot: ArmatureSnapshot with pose_matrices (source="pose") captured
              right after importing the model's rig, or rest matrices
              (source="rest") when the Blender rest pose is the reference.
    tolerance: max allowed joint position mismatch (meters) after removing
               the armature's placement. Larger mismatches mean the armature
               is not in the reference pose or is a different rig.
    """
    if bone_map is None:
        bone_map = auto_bone_map(profile.bone_names, snapshot.bone_names)
    blender_names = [bone_map[n] for n in profile.bone_names]
    armature_space = snapshot.pose_matrices if source == "pose" else snapshot.rest_matrices
    if armature_space is None:
        raise ValueError("Snapshot has no %s matrices." % source)

    probe = Calibration(
        model_bone_names=list(profile.bone_names),
        blender_bone_names=blender_names,
        offsets=np.repeat(np.eye(3)[None], profile.bone_count, 0),
        reference_basis={},
        position_residuals=np.zeros(profile.bone_count),
    )
    g = cv.transforms_blender_to_ai4a(
        probe.blender_world(snapshot, armature_space), snapshot.meters_per_unit
    )
    ref = cv.orthonormalize_transforms(profile.reference_transforms)

    r, t, scale = fit_rigid(g[:, :3, 3], ref[:, :3, 3])
    aligned_pos = g[:, :3, 3] @ r.T + t
    residuals = np.linalg.norm(aligned_pos - ref[:, :3, 3], axis=-1)
    if residuals.max() > tolerance:
        worst = np.argsort(-residuals)[:5]
        detail = ", ".join(
            "%s=%.1fcm" % (profile.bone_names[i], 100 * residuals[i]) for i in worst
        )
        hint = ""
        if abs(scale - 1.0) > 0.05:
            hint = " Estimated scale mismatch x%.3f (check units / armature scale)." % scale
        raise ValueError(
            "Armature does not match the model reference pose (max %.1f cm > %.1f cm: %s).%s "
            "Calibrate right after importing the model rig, before posing it."
            % (100 * residuals.max(), 100 * tolerance, detail, hint)
        )

    aligned_rot = np.einsum("ij,bjk->bik", r, g[:, :3, :3])
    offsets = np.einsum("bji,bjk->bik", aligned_rot, ref[:, :3, :3])  # (Q G_R)^T A_R
    offsets = cv.orthonormalize_zy(offsets)

    basis = snapshot.basis_matrices
    reference_basis = {}
    for i, name in enumerate(snapshot.bone_names):
        if source == "pose" and basis is not None:
            reference_basis[name] = basis[i].copy()
        else:
            reference_basis[name] = np.eye(4)

    placement = np.eye(4)
    placement[:3, :3] = r
    placement[:3, 3] = t
    return Calibration(
        model_bone_names=list(profile.bone_names),
        blender_bone_names=blender_names,
        offsets=offsets,
        reference_basis=reference_basis,
        position_residuals=residuals,
        placement=placement,
        source=source,
    )
