"""GFootball arena: the submission contract, a self-check, matches and tournaments.

A *submission* is a folder with ``agent.py`` (defining ``class Agent``),
``manifest.yml`` and whatever files the agent loads (see
``assignments/gfootball.md`` and ``arena/template/``).  The same code
runs on the student's machine (``python -m arena.check``) and on the grader's
(``arena.match``, ``arena.tournament``), always inside the gfootball image.

This package is deliberately separate from training: it never imports
``train.py`` or ``algorithms/``.  The only framework code that imports it is the
GFootball env shim, to load a submission as a training opponent.
"""
