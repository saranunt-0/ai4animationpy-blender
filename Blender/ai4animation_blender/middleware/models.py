# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Models the add-on can drive, and the characters each one animates.

Shared by Blender and the runner, so both sides agree on file locations,
root topology and input sizes. Paths are relative to the ai4animationpy
repository. A character is "paired" with a Blender armature by calibrating
the armature against that character's rig profile.
"""

from dataclasses import dataclass
from typing import Tuple

BIPED = "BIPED"
QUADRUPED = "QUADRUPED"

TOPOLOGY_BIPED = "BIPED"
TOPOLOGY_QUADRUPED = "QUADRUPED"

# Quadruped "styles": Auto = the demo's gait guidance chosen by speed; the
# others are the demo's action poses (gamepad R1 / L1 / L2).
QUADRUPED_AUTO = "Auto"
QUADRUPED_ACTIONS = ("Sit", "Stand", "Lie")
QUADRUPED_STYLES = (QUADRUPED_AUTO,) + QUADRUPED_ACTIONS

TRAJECTORY_SAMPLES = 16  # future root samples fed to the network (x/z position, direction, velocity)


@dataclass(frozen=True)
class Character:
    key: str  # stable id stored in calibrations and requests
    label: str
    model_file: str  # glTF the demo loads (and the add-on imports)
    profile_file: str  # shipped rig profile (profiles/<name>)
    fbx_file: str = ""  # optional FBX import (only where it keeps the bone hierarchy)


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    demo_dir: str  # the demo whose controller code is reused
    assets_dir: str  # holds Definitions.py
    network: str  # default network, relative to demo_dir
    postprocessor: str  # default contact network, relative to demo_dir
    root_topology: str
    characters: Tuple[Character, ...]
    network_iterations: int  # the demo's default
    network_features_per_bone: int  # per-bone state + guidance values in the network input
    max_speed: float  # m/s, top of the demo's speed range
    facing_control: bool  # facing independent of movement (right stick)
    goal_controller: bool  # has the Authoring goal controller

    def character(self, key):
        for c in self.characters:
            if c.key == key:
                return c
        raise KeyError("Model %s has no character %r (available: %s)" % (self.key, key, self.character_keys()))

    def character_keys(self):
        return [c.key for c in self.characters]

    def network_input_dim(self, bone_count):
        return self.network_features_per_bone * bone_count + TRAJECTORY_SAMPLES * 6

    @staticmethod
    def postprocessor_input_dim(bone_count, contact_count, sequence_length=16):
        # bone positions, forward, up, velocities + per future frame and contact
        # bone: distance, angle and velocity change
        return 12 * bone_count + 3 * contact_count * (sequence_length - 1)


MODELS = {
    BIPED: ModelSpec(
        key=BIPED,
        label="Biped",
        demo_dir="Demos/Authoring",
        assets_dir="Demos/_ASSETS_/Geno",
        network="Models/Network.pt",
        postprocessor="Models/PostProcessor.pt",
        root_topology=TOPOLOGY_BIPED,
        characters=(
            Character("geno", "Geno", "Demos/_ASSETS_/Geno/Model.glb", "geno_profile.json",
                      "Demos/_ASSETS_/Geno/Model.fbx"),
        ),
        network_iterations=3,
        network_features_per_bone=15,  # position, forward, up, velocity, guidance
        max_speed=3.0,
        facing_control=True,
        goal_controller=True,
    ),
    QUADRUPED: ModelSpec(
        key=QUADRUPED,
        label="Quadruped",
        demo_dir="Demos/Locomotion/Quadruped",
        assets_dir="Demos/_ASSETS_/Quadruped",
        network="Network.pt",
        postprocessor="Postprocessor.pt",
        root_topology=TOPOLOGY_QUADRUPED,
        characters=(
            Character("dog", "Dog", "Demos/_ASSETS_/Quadruped/Dog.glb", "dog_profile.json"),
            Character("wolf", "Wolf", "Demos/_ASSETS_/Quadruped/Wolf.glb", "wolf_profile.json"),
        ),
        network_iterations=1,
        network_features_per_bone=9,  # position, velocity, guidance
        max_speed=4.0,
        facing_control=False,
        goal_controller=False,
    ),
}


def get(key):
    try:
        return MODELS[key]
    except KeyError:
        raise KeyError("Unknown model %r (available: %s)" % (key, sorted(MODELS))) from None


def character_model(character_key):
    """The model that animates a character."""
    for spec in MODELS.values():
        if character_key in spec.character_keys():
            return spec
    raise KeyError("Unknown character %r" % character_key)
