"""Google Research Football quirks, isolated as an "added environment".

The env itself lives in ``envs/custom_envs/gfootball.py`` (Gymnasium shim) and
is registered as ``GFootball/<scenario>-v0`` when gfootball is installed.  The
only framework-level quirk is video: the engine renders in software at
~10 steps/s, so per-step training clips are off.  The final evaluation video
still works (``capture_video: true``) — the shim turns rendering on only
while ``RecordVideo`` is actually asking for frames.
"""

from __future__ import annotations

import importlib.util

from envs.adapters.base import EnvAdapter, register_adapter

if importlib.util.find_spec("gfootball") is not None:
    register_adapter("GFootball/", EnvAdapter(supports_training_video=False))
