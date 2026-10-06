"""Per-environment behaviour overrides ("adapters").

Most environments follow the plain Gymnasium interface and need *nothing* from
this package — :func:`get_adapter` returns a default :class:`EnvAdapter` whose
fields all mean "behave like a standard env".

Some environments have quirks: they ship their own vectoriser, apply certain
wrappers internally, can't record training video, or need a bespoke evaluation
protocol.  Instead of scattering ``if env_id == ...`` checks across the
framework, each such environment declares its quirks in ONE file and registers
them here.  The framework then asks the registry generic questions like
"does this env supply its own vectoriser?" without ever naming a specific env.

Layout
------
- ``adapters/base.py`` — the mechanism: the :class:`EnvAdapter` dataclass plus
  :func:`register_adapter` / :func:`get_adapter`.
- ``adapters/<env>.py`` — one file per quirky env (see ``adapters/metaworld.py``
  for a worked example).
- ``adapters/__init__.py`` (this file) — re-exports the mechanism and imports
  the bundled env adapters so that importing :mod:`envs.adapters` registers them.

Adding a new environment with quirks
------------------------------------
1. Create ``envs/adapters/myenv.py`` and register an adapter for your env-id
   prefix::

       from envs.adapters.base import EnvAdapter, register_adapter

       register_adapter(
           "MyEnv/",
           EnvAdapter(
               skip_episode_stats=True,          # env records episode stats itself
               supports_training_video=False,    # can't render during training
               make_vector_env=my_make_vec,      # custom vectoriser (optional)
               apply_wrappers=my_wrapper_stack,  # replaces the algorithm's
                                                 # preprocessing stack (optional)
               evaluate=my_evaluate,             # custom eval protocol (optional)
           ),
       )

2. Add ``from envs.adapters import myenv  # noqa: F401`` to the "bundled
   adapters" block at the bottom of this file.
"""

from envs.adapters.base import EnvAdapter, get_adapter, register_adapter

# ---------------------------------------------------------------------------
# Bundled adapters — imported for their registration side effects.
# Add one import line per new env adapter module.
# ---------------------------------------------------------------------------
from envs.adapters import metaworld  # noqa: F401
from envs.adapters import gfootball  # noqa: F401

__all__ = ["EnvAdapter", "get_adapter", "register_adapter"]
