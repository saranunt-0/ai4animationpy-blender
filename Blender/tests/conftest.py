# Copyright (c) Meta Platforms, Inc. and affiliates.
import os
import sys
from pathlib import Path

import pytest

BLENDER_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = BLENDER_DIR.parent
sys.path.insert(0, str(BLENDER_DIR))
sys.path.append(str(REPO_DIR))  # ai4animation without `pip install -e .` (parity tests)

PROFILE_PATH = BLENDER_DIR / "ai4animation_blender" / "profiles" / "geno_profile.json"


@pytest.fixture(scope="session")
def profile():
    from ai4animation_blender.middleware.rig import RigProfile

    return RigProfile.from_json(PROFILE_PATH.read_text())


@pytest.fixture(scope="session")
def repo_dir():
    return REPO_DIR


def model_python():
    """Python executable of the AI4Animation environment (torch + ai4animation)."""
    return os.environ.get("AI4A_PYTHON")
