"""Agent host: runs ONE submission in its own process, spoken to over a pipe.

Started by :class:`arena.agents.RemoteAgent` as a plain script::

    python arena/_host.py <submission_dir>

Deliberately standalone (stdlib + numpy only, no ``arena`` imports): the
student's code must see a clean interpreter whose import path is its own
folder and site-packages — not the t-zero checkout. A submission that imports
``networks`` or ``algorithms`` fails here exactly as it would on the grader's
machine.

Protocol (length-prefixed pickles; the host answers every request once)::

    host -> parent   ("ready", load_seconds)      after Agent(path) returns
    parent -> host   ("reset",)                   -> ("ok", None, seconds)
    parent -> host   ("act", [obs, ...])          -> ("ok", [int, ...], seconds)
    parent -> host   ("close",)                   host exits
    any failure      ("error", kind, text)        then the host exits
                     kind: "invalid" (bad return value) or "crash" (exception)

Anything the agent prints (Python or C level) goes to stderr: fd 1 is
re-pointed at fd 2 before the agent is imported, so the protocol stream
cannot be corrupted.
"""
from __future__ import annotations

import importlib.util
import os
import pickle
import struct
import sys
import tempfile
import time
import traceback
from pathlib import Path

_HEADER = struct.Struct("!I")


def _send(stream, message) -> None:
    payload = pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)
    stream.write(_HEADER.pack(len(payload)) + payload)
    stream.flush()


def _recv(stream):
    header = stream.read(_HEADER.size)
    if len(header) < _HEADER.size:
        raise EOFError("parent closed the pipe")
    (size,) = _HEADER.unpack(header)
    payload = b""
    while len(payload) < size:
        chunk = stream.read(size - len(payload))
        if not chunk:
            raise EOFError("parent closed the pipe")
        payload += chunk
    return pickle.loads(payload)


def _as_actions(value, expected: int, num_actions: int) -> list[int]:
    """Normalise what ``act`` returned to ``list[int]`` or raise ValueError."""
    import numpy as np

    try:
        arr = np.asarray(value)
    except Exception as exc:  # e.g. a CUDA tensor
        raise ValueError(f"act() returned {type(value).__name__}, not a list of ints") from exc
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim == 1 and arr.shape[0] != expected:
        raise ValueError(
            f"act() got {expected} observation(s) but returned {arr.shape[0]} action(s)"
        )
    if arr.ndim != 1 or arr.dtype.kind not in "iub":
        raise ValueError(
            f"act() must return a list of {expected} int(s); got {value!r:.200}"
        )
    actions = [int(a) for a in arr]
    bad = [a for a in actions if not 0 <= a < num_actions]
    if bad:
        raise ValueError(f"action(s) {bad} outside [0, {num_actions})")
    return actions


def main() -> None:
    submission = Path(sys.argv[1]).resolve()
    num_actions = int(sys.argv[2]) if len(sys.argv) > 2 else 19

    # Private copies of the protocol pipes, then point fds 0/1 away from them.
    proto_in = os.fdopen(os.dup(0), "rb", buffering=0)
    proto_out = os.fdopen(os.dup(1), "wb", buffering=0)
    devnull = os.open(os.devnull, os.O_RDONLY)
    os.dup2(devnull, 0)
    os.dup2(2, 1)
    sys.stdin = open(os.devnull)
    sys.stdout = sys.stderr

    # Import path: the submission folder + the interpreter's own paths.
    # sys.path[0] is this script's directory (arena/) — replace it.
    sys.path[0] = str(submission)
    os.chdir(tempfile.mkdtemp(prefix="arena-agent-"))

    try:
        t0 = time.perf_counter()
        spec = importlib.util.spec_from_file_location("agent", submission / "agent.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules["agent"] = module
        spec.loader.exec_module(module)
        if not hasattr(module, "Agent"):
            raise AttributeError("agent.py does not define a class named Agent")
        agent = module.Agent(submission)
        _send(proto_out, ("ready", time.perf_counter() - t0))
    except BaseException:
        _send(proto_out, ("error", "crash", traceback.format_exc()))
        sys.exit(1)

    while True:
        try:
            message = _recv(proto_in)
        except EOFError:
            return
        command = message[0]
        try:
            if command == "close":
                return
            if command == "reset":
                t0 = time.perf_counter()
                if hasattr(agent, "reset"):
                    agent.reset()
                _send(proto_out, ("ok", None, time.perf_counter() - t0))
            elif command == "act":
                observations = message[1]
                t0 = time.perf_counter()
                value = agent.act(observations)
                elapsed = time.perf_counter() - t0
                try:
                    actions = _as_actions(value, len(observations), num_actions)
                except ValueError as exc:
                    _send(proto_out, ("error", "invalid", str(exc)))
                    sys.exit(1)
                _send(proto_out, ("ok", actions, elapsed))
            else:
                raise ValueError(f"unknown command {command!r}")
        except BaseException:
            _send(proto_out, ("error", "crash", traceback.format_exc()))
            sys.exit(1)


if __name__ == "__main__":
    main()
