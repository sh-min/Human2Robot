"""Repo-relative paths for the retargeting module.

Importable by sibling scripts so they all agree on where things live without
hard-coding absolute paths.
"""
import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))

# Vendored upstream
DEX_RETARGETING_DIR = os.path.join(REPO_ROOT, "third_party", "dex-retargeting")

# Local xhand assets (URDF + meshes) and configs
ASSETS_DIR = os.path.join(THIS_DIR, "assets")
URDF_ROOT = ASSETS_DIR                           # parent that contains "xhand/" subdir
CONFIG_DIR = os.path.join(THIS_DIR, "configs")

# Convenience yml paths
YML_RIGHT = os.path.join(CONFIG_DIR, "xhand_right_dexpilot.yml")
YML_LEFT = os.path.join(CONFIG_DIR, "xhand_left_dexpilot.yml")
