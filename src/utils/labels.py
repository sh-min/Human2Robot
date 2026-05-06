"""Skill label definitions shared across the pipeline."""

# 12 cube rotation actions (6 faces x CW/CCW) + TRANS (transition between skills)
ACTION_LABELS = [
    "BCW", "BCCW", "DCW", "DCCW", "FCW", "FCCW",
    "LCW", "LCCW", "RCW", "RCCW", "UCW", "UCCW",
    "TRANS",
]
