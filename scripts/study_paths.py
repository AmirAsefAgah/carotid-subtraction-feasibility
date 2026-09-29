"""Resolve local study paths without workstation-specific defaults."""

import os
from pathlib import Path

ROOT = Path(
    os.environ.get("CAROTID_STUDY_ROOT", Path(__file__).resolve().parents[1])
).resolve()
SOURCE_ROOT = Path(
    os.environ.get("CAROTID_SOURCE_ROOT", ROOT / "private" / "source")
).resolve()
