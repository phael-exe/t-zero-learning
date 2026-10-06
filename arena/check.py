"""Self-check a submission exactly the way the grader will run it.

    python -m arena.check <submission_dir> [--matches 1] [--max-steps N] [--json report.json]

Run it inside the gfootball image (``docker compose run --rm gfootball ...``).
It checks, in order:

1. layout and ``manifest.yml``;
2. folder size;
3. loading ``Agent`` in an isolated process (only the submission folder and
   site-packages on the import path, one CPU thread) within the time limit;
4. the probe replay (``probe.pkl``), if present: do recorded actions match?
5. full matches against the ``random`` and ``builtin`` anchors, each in a fresh
   process, like the tournament: no crash, valid actions, within the time
   limits. It also reports the score and warns when the agent keeps
   repeating one action.

Exit code 0 = PASS (warnings allowed), 1 = FAIL.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from arena.agents import AgentError, RandomAgent, BuiltinAI, RemoteAgent, read_manifest
from arena.match import new_seed, play_match
from arena.probe import PROBE_FILE, load_probe
from arena.rules import RULES

PROBE_FAIL_BELOW = 0.90
PROBE_WARN_BELOW = 0.99
SAME_ACTION_WARN = 0.90


@dataclass
class Finding:
    level: str  # PASS | WARN | FAIL
    check: str
    message: str


class Report:
    def __init__(self):
        self.findings: list[Finding] = []
        self.matches: list[dict] = []

    def add(self, level, check, message):
        self.findings.append(Finding(level, check, message))
        mark = {"PASS": "  ok ", "WARN": "WARN ", "FAIL": "FAIL "}[level]
        print(f"[{mark}] {check}: {message}", flush=True)

    @property
    def passed(self) -> bool:
        return not any(f.level == "FAIL" for f in self.findings)


def _folder_mb(folder: Path) -> float:
    total = sum(p.stat().st_size for p in folder.rglob("*")
                if p.is_file() and "__pycache__" not in p.parts)
    return total / 2**20


def _check_probe(folder: Path, manifest, report: Report) -> None:
    probe_path = folder / PROBE_FILE
    if not probe_path.is_file():
        report.add("WARN", "probe", f"no {PROBE_FILE}: preprocessing/normalization not verified "
                   "(see arena/probe.py)")
        return
    try:
        probe = load_probe(probe_path)
    except Exception as exc:
        report.add("FAIL", "probe", f"cannot read {PROBE_FILE}: {exc}")
        return
    if probe["controlled_players"] != manifest.controlled_players:
        report.add("FAIL", "probe", f"probe was recorded with {probe['controlled_players']} "
                   f"player(s), manifest declares {manifest.controlled_players}")
        return
    matched = total = 0
    try:
        with RemoteAgent(folder) as agent:
            for episode in probe["episodes"]:
                agent.reset()
                for step in episode:
                    got = agent.act(step["obs"])
                    matched += sum(int(g == e) for g, e in zip(got, step["actions"]))
                    total += len(step["actions"])
    except AgentError as err:
        report.add("FAIL", "probe", f"agent failed during probe replay: {err.kind}: {err.detail}")
        return
    if total == 0:
        report.add("WARN", "probe", f"{PROBE_FILE} is empty")
        return
    rate = matched / total
    text = f"{matched}/{total} recorded actions reproduced ({rate:.1%})"
    if not manifest.deterministic:
        report.add("PASS" if rate >= PROBE_FAIL_BELOW else "WARN", "probe",
                   text + " — stochastic agent, informational only")
    elif rate < PROBE_FAIL_BELOW:
        report.add("FAIL", "probe", text + ": the submitted agent does not behave like the "
                   "recorded one (preprocessing, normalization stats or weights differ)")
    elif rate < PROBE_WARN_BELOW:
        report.add("WARN", "probe", text + ": small mismatches (float noise or ties?)")
    else:
        report.add("PASS", "probe", text)


def _play_check_matches(folder: Path, matches: int, max_steps, report: Report) -> None:
    act_ms: list[float] = []
    act_max_ms = 0.0
    actions = Counter()
    for anchor_name, make_anchor in (("random", lambda: RandomAgent(1)), ("builtin", BuiltinAI)):
        summary = Counter()
        for i in range(matches):
            try:
                agent = RemoteAgent(folder)
            except AgentError as err:
                report.add("FAIL", "load", f"{err.kind}: {err.detail}")
                return
            anchor = make_anchor()
            with agent:
                left, right = (agent, anchor) if i % 2 == 0 else (anchor, agent)
                result = play_match(left, right, seed=new_seed(), max_steps=max_steps)
            report.matches.append(asdict(result))
            ours_left = left is agent
            gf, ga = (result.goals_left, result.goals_right) if ours_left else (result.goals_right, result.goals_left)
            print(f"        match vs {anchor_name} ({'left' if ours_left else 'right'}): "
                  f"{gf}-{ga}, {result.steps} steps, {result.wall_seconds:.0f} s", flush=True)
            if result.forfeit == agent.name:
                report.add("FAIL", f"match vs {anchor_name}",
                           f"forfeit: {result.forfeit_reason}")
                return
            summary.update(gf=gf, ga=ga, w=int(gf > ga), d=int(gf == ga), l=int(gf < ga))
            act_ms.extend(1000 * s for s in agent.act_seconds[1:])  # first call = warm-up
            if agent.act_seconds:
                act_max_ms = max(act_max_ms, 1000 * max(agent.act_seconds))
            actions.update(result.action_counts.get(agent.name, {}))
        report.add("PASS", f"matches vs {anchor_name}",
                   f"{summary['w']}W {summary['d']}D {summary['l']}L, goals {summary['gf']}-{summary['ga']}")

    if act_ms:
        mean = float(np.mean(act_ms))
        text = f"act() mean {mean:.2f} ms, max {act_max_ms:.1f} ms (limit: mean {RULES.mean_act_ms:g} ms)"
        report.add("FAIL" if mean > RULES.mean_act_ms else "PASS", "speed", text)
    if actions:
        top, count = actions.most_common(1)[0]
        share = count / sum(actions.values())
        if share > SAME_ACTION_WARN:
            probe_ok = any(f.check == "probe" and f.level == "PASS" for f in report.findings)
            hint = ("the probe matched, so this is the policy itself (e.g. trained on another "
                    "scenario)" if probe_ok else "broken preprocessing often looks like this")
            report.add("WARN", "behaviour", f"action {top} is {share:.0%} of all actions — {hint}")


def run_check(folder: str | Path, matches: int = 1, max_steps: int | None = None) -> Report:
    folder = Path(folder).resolve()
    report = Report()
    if not (folder / "agent.py").is_file():
        report.add("FAIL", "layout", f"{folder / 'agent.py'} not found")
        return report
    try:
        manifest = read_manifest(folder)
    except ValueError as exc:
        report.add("FAIL", "manifest", str(exc))
        return report
    report.add("PASS", "manifest", f"team {manifest.team!r}, controlled_players={manifest.controlled_players}, "
               f"deterministic={manifest.deterministic}")
    if manifest.unknown_keys:
        report.add("WARN", "manifest", f"unknown keys ignored: {manifest.unknown_keys}")

    size = _folder_mb(folder)
    report.add("FAIL" if size > RULES.max_submission_mb else "PASS", "size",
               f"{size:.1f} MB (limit {RULES.max_submission_mb:g} MB)")

    try:
        with RemoteAgent(folder) as agent:
            load_s = agent.load_seconds
    except AgentError as err:
        report.add("FAIL", "load", f"{err.kind}: {err.detail}")
        return report
    report.add("PASS", "load", f"Agent loaded in {load_s:.2f} s (limit {RULES.load_timeout_s:g} s)")

    _check_probe(folder, manifest, report)
    if matches > 0:
        _play_check_matches(folder, matches, max_steps, report)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("submission", type=Path)
    parser.add_argument("--matches", type=int, default=1, help="matches per anchor (default 1; sides alternate)")
    parser.add_argument("--max-steps", type=int, default=None, help="cut matches short (quick iteration only)")
    parser.add_argument("--json", type=Path, default=None, help="also write the report as JSON")
    args = parser.parse_args(argv)

    print(f"arena.check {args.submission}  (scenario {RULES.scenario})")
    report = run_check(args.submission, matches=args.matches, max_steps=args.max_steps)
    if args.max_steps is not None and report.passed:
        report.add("WARN", "matches", f"cut at {args.max_steps} steps — run without --max-steps before submitting")
    verdict = "PASS" if report.passed else "FAIL"
    print(f"\n{verdict}")
    if args.json:
        args.json.write_text(json.dumps({"verdict": verdict,
                                         "findings": [asdict(f) for f in report.findings],
                                         "matches": report.matches}, indent=2))
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
