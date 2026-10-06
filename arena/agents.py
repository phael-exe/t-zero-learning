"""Agent handles: everything that can take one side of a match.

All handles share one small interface — ``name``, ``controlled_players``,
``reset()``, ``act(observations) -> list[int]``, ``close()`` — so the match
runner never cares what is behind a side:

- :class:`RemoteAgent`: a submission running in its own process
  (``arena/_host.py``), with the rules' timeouts enforced from outside.
  Used for every student agent in ``arena.check`` / ``match`` / ``tournament``.
- :class:`InProcessAgent`: a submission imported into the current process.
  Faster, no isolation — used by the env shim for training opponents.
- :class:`RandomAgent` and :class:`BuiltinAI`: the anchors. ``BuiltinAI``
  controls no player at all; the whole team is the engine's own AI.

:func:`make_agent` turns a spec string (``"random"``, ``"builtin"`` or a
folder path) into a handle.
"""
from __future__ import annotations

import importlib.util
import os
import pickle
import select
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from arena._host import _HEADER, _as_actions, _send
from arena.rules import RULES

HOST_SCRIPT = Path(__file__).with_name("_host.py")
ANCHORS = ("random", "builtin")

MANIFEST_KEYS = {"team", "members", "controlled_players", "deterministic", "description"}


class AgentError(Exception):
    """An agent broke the contract. ``kind`` is ``crash``, ``timeout`` or ``invalid``."""

    def __init__(self, agent: str, kind: str, detail: str):
        super().__init__(f"{agent}: {kind}: {detail}")
        self.agent = agent
        self.kind = kind
        self.detail = detail


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


@dataclass
class Manifest:
    team: str
    controlled_players: int
    deterministic: bool = True
    members: list = field(default_factory=list)
    description: str = ""
    unknown_keys: list = field(default_factory=list)


def read_manifest(folder: Path) -> Manifest:
    """Parse and validate ``<folder>/manifest.yml``; raises ValueError with a readable message."""
    path = Path(folder) / "manifest.yml"
    if not path.is_file():
        raise ValueError(f"{path} not found")
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping of keys")
    team = str(data.get("team") or "").strip()
    if not team:
        raise ValueError(f"{path}: 'team' is required")
    n = data.get("controlled_players")
    if not isinstance(n, int) or isinstance(n, bool) or not 1 <= n <= RULES.max_controlled_players:
        raise ValueError(
            f"{path}: 'controlled_players' must be an integer in "
            f"1..{RULES.max_controlled_players}, got {n!r}"
        )
    deterministic = data.get("deterministic", True)
    if not isinstance(deterministic, bool):
        raise ValueError(f"{path}: 'deterministic' must be true or false")
    return Manifest(
        team=team,
        controlled_players=n,
        deterministic=deterministic,
        members=list(data.get("members") or []),
        description=str(data.get("description") or ""),
        unknown_keys=sorted(set(data) - MANIFEST_KEYS),
    )


# ---------------------------------------------------------------------------
# Handles
# ---------------------------------------------------------------------------


class BaseAgent:
    name: str
    controlled_players: int

    def reset(self) -> None:
        pass

    def act(self, observations: list[dict]) -> list[int]:
        raise NotImplementedError

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class BuiltinAI(BaseAgent):
    """The engine's own AI plays the whole team (0 agent-controlled players)."""

    def __init__(self, name: str = "builtin"):
        self.name = name
        self.controlled_players = 0

    def act(self, observations):
        return []


class RandomAgent(BaseAgent):
    """Uniformly random actions for ``controlled_players`` slots."""

    def __init__(self, controlled_players: int = 1, seed: int | None = None, name: str = "random"):
        self.name = name
        self.controlled_players = int(controlled_players)
        self._rng = np.random.default_rng(seed)

    def act(self, observations):
        return [int(a) for a in self._rng.integers(0, RULES.num_actions, size=len(observations))]


class InProcessAgent(BaseAgent):
    """A submission imported into this interpreter (training opponents).

    The submission folder is put on ``sys.path`` while ``agent.py`` loads, so
    helper modules next to it resolve — but they live in the shared
    ``sys.modules``: two different opponents with a helper module of the same
    name would clash. No timeouts, no isolation.
    """

    def __init__(self, folder: str | Path):
        folder = Path(folder).resolve()
        manifest = read_manifest(folder)
        self.name = manifest.team
        self.controlled_players = manifest.controlled_players
        self.manifest = manifest
        module_name = f"_arena_submission_{abs(hash(str(folder)))}"
        spec = importlib.util.spec_from_file_location(module_name, folder / "agent.py")
        module = importlib.util.module_from_spec(spec)
        sys.path.insert(0, str(folder))
        try:
            spec.loader.exec_module(module)
        finally:
            sys.path.remove(str(folder))
        self._agent = module.Agent(folder)

    def reset(self):
        if hasattr(self._agent, "reset"):
            self._agent.reset()

    def act(self, observations):
        return _as_actions(self._agent.act(observations), len(observations), RULES.num_actions)


class RemoteAgent(BaseAgent):
    """A submission running in its own process, with the rules' limits enforced.

    Every failure — import error, exception in ``act``, wrong return value,
    exceeding a timeout, the process dying — surfaces as :class:`AgentError`.
    Timing: ``act_seconds`` collects the host-side compute time of each call.
    """

    def __init__(self, folder: str | Path, stderr=None, rules=RULES):
        self.folder = Path(folder).resolve()
        self.manifest = read_manifest(self.folder)
        self.name = self.manifest.team
        self.controlled_players = self.manifest.controlled_players
        self.rules = rules
        self.act_seconds: list[float] = []
        self.load_seconds: float | None = None
        self._first_act = True

        env = dict(os.environ)
        env.pop("PYTHONPATH", None)  # the agent must not see the t-zero checkout
        threads = str(rules.agent_threads)
        env.update(OMP_NUM_THREADS=threads, MKL_NUM_THREADS=threads,
                   OPENBLAS_NUM_THREADS=threads, PYTHONDONTWRITEBYTECODE="1")
        self._proc = subprocess.Popen(
            [sys.executable, str(HOST_SCRIPT), str(self.folder), str(rules.num_actions)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr,
            env=env,
        )
        reply = self._read(rules.load_timeout_s, "loading (Agent.__init__)")
        if reply[0] != "ready":
            self._fail(reply)
        self.load_seconds = float(reply[1])

    # --- protocol -----------------------------------------------------------

    def _read(self, timeout: float, doing: str):
        fd = self._proc.stdout.fileno()
        deadline = time.monotonic() + timeout
        buf = b""
        need = _HEADER.size
        header_done = False
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._proc.kill()  # busy agent: no polite close
                self._proc.wait()
                raise AgentError(self.name, "timeout", f"{doing} took longer than {timeout:g} s")
            ready, _, _ = select.select([fd], [], [], remaining)
            if not ready:
                continue
            chunk = os.read(fd, need - len(buf))
            if not chunk:
                code = self._proc.wait(timeout=5)
                raise AgentError(self.name, "crash", f"process exited (code {code}) while {doing}")
            buf += chunk
            if len(buf) == need:
                if header_done:
                    return pickle.loads(buf)
                (need,) = _HEADER.unpack(buf)
                buf = b""
                header_done = True

    def _request(self, message, timeout: float, doing: str):
        if self._proc.poll() is not None:
            raise AgentError(self.name, "crash", f"process already exited (code {self._proc.returncode})")
        try:
            _send(self._proc.stdin, message)
        except (BrokenPipeError, OSError):
            raise AgentError(self.name, "crash", f"process died before {doing}") from None
        reply = self._read(timeout, doing)
        if reply[0] == "error":
            self._fail(reply)
        return reply

    def _fail(self, reply) -> None:
        _, kind, detail = reply
        self.close()
        raise AgentError(self.name, kind, str(detail).strip())

    # --- interface ----------------------------------------------------------

    def reset(self):
        self._request(("reset",), self.rules.reset_timeout_s, "reset()")

    def act(self, observations):
        timeout = self.rules.first_act_timeout_s if self._first_act else self.rules.act_timeout_s
        self._first_act = False
        _, actions, seconds = self._request(("act", observations), timeout, "act()")
        self.act_seconds.append(float(seconds))
        return actions

    def close(self):
        proc = getattr(self, "_proc", None)
        if proc is None or proc.poll() is not None:
            return
        try:
            _send(proc.stdin, ("close",))
            proc.stdin.close()
            proc.wait(timeout=2)
        except Exception:
            pass
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def make_agent(spec: str | Path, stderr=None, seed: int | None = None) -> BaseAgent:
    """``"random"`` / ``"builtin"`` anchors, or a submission folder (run in its own process)."""
    if str(spec) == "random":
        return RandomAgent(1, seed=seed)
    if str(spec) == "builtin":
        return BuiltinAI()
    return RemoteAgent(spec, stderr=stderr)


def spec_name(spec: str | Path) -> str:
    """Display name for a spec without starting it (team from the manifest)."""
    if str(spec) in ANCHORS:
        return str(spec)
    try:
        return read_manifest(Path(spec)).team
    except ValueError:
        return Path(spec).name
