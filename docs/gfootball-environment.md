# GFootball environment reference

This page covers what the Google Research Football environment gives you in
t-zero: how to create it, every setting, the observations, the actions, the
rewards, how episodes end, how to control several players, and how to choose
the opposing team (for example for self-play). How to hand in your agent is in
[assignments/gfootball.md](../assignments/gfootball.md).

Everything here was checked against the installed engine (gfootball 2.10.3 in
`docker/Dockerfile.gfootball`) and the shim in
[envs/custom_envs/gfootball.py](../envs/custom_envs/gfootball.py). Upstream's
README and paper sometimes disagree with the code. When they do, this page
follows the code.

All commands run inside the image:

```bash
docker compose run --rm gfootball python train.py --config dqn_gfootball_empty_goal
docker compose run --rm gfootball python -m pytest tests/test_gfootball.py
```

## 1. Quick facts

| | |
|---|---|
| Env ids | `GFootball/<scenario>-v0`, 16 scenarios (section 3) |
| Default observation | `Box(-inf, inf, (115,), float32)` (`simple115v2`) |
| Default action space | `Discrete(19)` |
| Who you control | by default 1 player, the **active** one (control moves between your players during play); the game AI plays your other players and your keeper. Section 8 |
| Default reward | `scoring,checkpoints`: ±1 per goal plus +0.1 shaping per zone |
| Time per step | 100 ms of game time (10 steps = 1 s) |
| Episode length | academy: ≤ 400 steps (ends early on goal, ball out or lost possession); `5_vs_5` and 11v11: 3000 steps |
| Speed (1 CPU core, no render) | ~250–310 steps/s |
| Rendering | software GL, ~10 steps/s at 640×360; only when recording video |
| Arena scenario | `5_vs_5`: the tournament is played there |

## 2. Creating the environment

With t-zero, choose the scenario with `env_id` and pass settings with
`env_kwargs`:

```yaml
env_id: GFootball/5_vs_5-v0
env_kwargs:
  rewards: scoring
  opponent: teams/snapshot_0050   # optional, section 9
```

or on the command line:

```bash
python train.py --config dqn_gfootball_empty_goal \
    --override env_id=GFootball/academy_3_vs_1_with_keeper-v0 \
    --override 'env_kwargs={"rewards": "scoring"}'
```

In your own code (a notebook, a custom training loop), use the same ids:

```python
import gymnasium as gym
import envs   # registers the GFootball/* ids (only if gfootball is installed)

env = gym.make("GFootball/5_vs_5-v0", controlled_players=2, opponent="random")
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

`env.unwrapped` is the `GFootballEnv` instance. You need it for
`raw_observations()` and `set_opponent()`. You can also build it directly with
`GFootballEnv(scenario="5_vs_5", ...)`.

## 3. Settings

Every keyword below works in `env_kwargs`, in `gym.make(...)` and in
`GFootballEnv(...)`.

### Settings of the t-zero shim

| kwarg | Default | Values / effect |
|---|---|---|
| `scenario` | from the id | Any gfootball scenario name, including your own (section 3.3). Overrides the id's scenario: `gym.make("GFootball/5_vs_5-v0", scenario="my_3v3")` |
| `representation` | `simple115v2` | `simple115v2`, `simple115`, `extracted` (SMM minimap), `pixels`, `pixels_gray`. **`raw` is not supported** (the shim fails): use `env.unwrapped.raw_observations()` instead. Section 5 |
| `rewards` | `scoring,checkpoints` | `scoring` or `scoring,checkpoints`. Section 7 |
| `controlled_players` | `1` | How many left-team players your agent controls. `> 1` changes the spaces. Section 8 |
| `opponent` | `None` (= `"builtin"`) | `"builtin"`, `"random"`, the path to a submission folder, or a Python object. Section 9 |
| `render_mode` | `None` | `None` or `"rgb_array"`. t-zero sets it on the video worker by itself |
| `render_resolution` | `(640, 360)` | `(width, height)` of rendered frames. Fixed once the first frame of a process is rendered |

### Settings passed through to `gfootball.env.create_environment`

Any other keyword goes unchanged to gfootball. The useful ones:

| kwarg | Effect |
|---|---|
| `stacked=True` | Stacks the last 4 observations along the last axis: `simple115v2` becomes `(460,)`, `extracted` becomes `(72, 96, 16)` |
| `channel_dimensions=(w, h)` | Size of `extracted` / `pixels` observations (default `(96, 72)`) |
| `other_config_options={...}` | Engine config keys, see below |
| `write_full_episode_dumps`, `write_goal_dumps`, `write_video`, `logdir` | gfootball's own replay dumps (not tested with the shim; `python -m arena.match --dump/--video` is the tested path) |

Do not pass `number_of_left_players_agent_controls`,
`number_of_right_players_agent_controls` or `render`: the shim sets them from
`controlled_players`, `opponent` and `render_mode`. Avoid `extra_players`
too. Its bundled `bot` player requires `action_set: full`, which would also
change your agent's action space. Use `opponent=` instead.

`other_config_options` keys that are useful:

| Key | Effect |
|---|---|
| `action_set` | `"default"` (19 actions) or `"v2"` (adds action 19 `builtin_ai`, section 6). `"full"` exists but is for the scripted bot |
| `reverse_team_processing` | Pins the order in which the engine processes the two teams. By default the shim picks it at random per episode, like upstream does per seed |
| `game_engine_random_seed` | Do not set this: the shim overwrites it on every `reset()` (section 4.2) |

Scenario parameters (duration, AI difficulty, offsides, when an episode ends)
**cannot** be changed through `other_config_options`. The scenario file sets
them on every reset. To change them, write your own scenario (3.3).

### 3.1 Academy scenarios

Short drills. They all share `game_duration = 400`, no offsides, and end on a
goal or when the ball goes out of play. All but `academy_corner` also end when
your team loses the ball. Your keeper is always played by the game AI.

| Scenario | Setup (your team attacks towards x = +1) |
|---|---|
| `academy_empty_goal_close` | 1 attacker with the ball near the box, empty goal. A random policy already scores sometimes |
| `academy_empty_goal` | 1 attacker at the centre spot, empty goal |
| `academy_run_to_score` | 1 attacker at the centre, 5 defenders chasing from behind, empty goal |
| `academy_run_to_score_with_keeper` | as above, plus a keeper |
| `academy_pass_and_shoot_with_keeper` | 2 attackers vs 1 defender + keeper; the defender marks the ball carrier on the wing, the other attacker is free in the centre |
| `academy_run_pass_and_shoot_with_keeper` | same attackers, but the defender starts between them, closer to the centre |
| `academy_3_vs_1_with_keeper` | 3 attackers vs 1 defender + keeper |
| `academy_corner` | corner kick, 11 vs 11; does *not* end on lost possession |
| `academy_counterattack_easy` | 4 attackers vs 1 defender in an 11v11 layout |
| `academy_counterattack_hard` | 4 attackers vs 2 defenders |
| `academy_single_goal_versus_lazy` | full 11v11, 3000 steps, opponents stand still (except their keeper) |

Most academy scenarios have only a few players per side. `simple115v2` fills
the missing players' slots with -1 (section 5.1).

### 3.2 Game scenarios

Games. A goal does *not* end the episode: play resumes with a kick-off until
the clock runs out.

| Scenario | Players | Steps | AI difficulty | Offsides |
|---|---|---|---|---|
| `1_vs_1_easy` | keeper only per side (you control the keeper) | 500 | 0.0 | on |
| `5_vs_5` (**arena**) | keeper + 4 outfield per side; keepers are never controllable | 3000 | 0.05 both teams | on |
| `11_vs_11_easy_stochastic` | 11 per side | 3000 | 0.05 opponent | on |
| `11_vs_11_stochastic` | 11 per side | 3000 | 0.6 opponent | on |
| `11_vs_11_hard_stochastic` | 11 per side | 3000 | 0.95 opponent | on |

"AI difficulty" is the engine's built-in AI strength (0 to 1). In `5_vs_5` it
applies to **both** teams: your AI-controlled teammates are as weak as the
opponents. The kick-off alternates between the teams from one episode to the
next.

### 3.3 Custom scenarios

A scenario is a Python file with a `build_scenario(builder)` function. Look at
the upstream ones for examples (inside the image:
`/usr/local/lib/python3.10/dist-packages/gfootball/scenarios/`). gfootball
imports scenarios as `gfootball.scenarios.<name>`. To load yours from the repo
without editing the installed package, add its folder to that package's search
path **before the env is created**:

```python
# my_scenarios/my_3v3.py
from gfootball.scenarios import *

def build_scenario(builder):
    builder.config().game_duration = 600
    builder.config().right_team_difficulty = 0.3
    builder.config().deterministic = False
    builder.config().end_episode_on_score = True
    builder.SetBallPosition(0.0, 0.0)
    builder.SetTeam(Team.e_Left)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK, controllable=False)
    builder.AddPlayer(0.0, 0.02, e_PlayerRole_CF)
    builder.AddPlayer(-0.2, 0.1, e_PlayerRole_CM)
    builder.SetTeam(Team.e_Right)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK, controllable=False)
    builder.AddPlayer(-0.1, 0.0, e_PlayerRole_CF)
    builder.AddPlayer(-0.2, -0.1, e_PlayerRole_CM)
```

```python
import os, gfootball.scenarios
gfootball.scenarios.__path__.append(os.path.abspath("my_scenarios"))
env = gym.make("GFootball/5_vs_5-v0", scenario="my_3v3", controlled_players=2)
```

Positions of the right team are written from *its own* point of view (it
attacks towards x = +1 too). Settings you can use: `game_duration`,
`left_team_difficulty`, `right_team_difficulty`, `deterministic`, `offsides`,
`end_episode_on_score`, `end_episode_on_out_of_play`,
`end_episode_on_possession_change`. `AddPlayer(x, y, role, lazy=False,
controllable=True)`: a `lazy` player never moves on its own. Use this for a
training curriculum. The tournament is always `5_vs_5`.

## 4. Episodes

### 4.1 How an episode ends

The engine only reports "done". The shim splits it the Gymnasium way:

| What happened | `terminated` | `truncated` |
|---|---|---|
| Academy: goal, ball out of play, possession lost | `True` | `False` |
| The clock ran out (`steps_left == 0`) with no goal on that step | `False` | `True` |
| Goal on the very last step | `True` | `False` |

Game scenarios (`5_vs_5`, 11v11) therefore **always end `truncated`** after
3000 steps: a goal does not end them. If your algorithm bootstraps, it must
bootstrap on truncation. When an episode ends, `info["steps_left"]` holds the
clock value. `info["score_reward"]` is +1 / -1 / 0 for the goal on that step,
without shaping.

### 4.2 Seeding

`reset(seed=s)` seeds the shim's generator. Each `reset()` then draws a new
engine seed from it. One run seed therefore gives a reproducible *sequence* of
different episodes: the same seed replays the same sequence given the same
actions. The seed only matters where the engine makes random decisions (AI
players, physics noise). `academy_empty_goal_close` has no opponent AI, so
given the same actions every episode is identical.

## 5. Observations

### 5.1 `simple115v2` (default): 115 floats

The same 115 numbers the arena helper
`Simple115StateWrapper.convert_observation(obs, True)` produces. They are
always from your team's point of view: you are `left_team` and attack towards
x = +1.

| Index | Content |
|---|---|
| 0–21 | left team (yours) positions: player *i* at `2i` (x), `2i+1` (y), 11 slots |
| 22–43 | left team movement per step (dx, dy), same layout |
| 44–65 | right team positions |
| 66–87 | right team movement |
| 88–90 | ball position (x, y, z) |
| 91–93 | ball movement per step (dx, dy, dz) |
| 94–96 | who has the ball, one-hot: nobody / you / opponent |
| 97–107 | **active player** one-hot (index into your team); which player this row belongs to |
| 108–114 | game mode one-hot: normal, kick-off, goal kick, free kick, corner, throw-in, penalty |

Each team always takes 11 slots. Slots for players who don't exist are filled
with -1: in `5_vs_5`, players 5–10 of each team (e.g. indices 10–21) are
always -1. Player 0 is the keeper in every scenario t-zero ships.

Coordinates: x ∈ [-1, 1] (goals at x = ±1), y ∈ [-0.42, 0.42], with **y
growing downwards** (so "top" means negative y). The ball's z is its height.

What the vector **does not** contain: the score, the clock (`steps_left`),
player roles, sticky-action state, tiredness, cards. If you need them, add
them yourself from the raw observation (5.3) with an observation wrapper, and
do the same preprocessing in your submission.

### 5.2 Other encodings

| `representation` | Shape (1 player) | Notes |
|---|---|---|
| `simple115` | `(115,)` | older layout: right team slots move when the left team has fewer than 11. Prefer `simple115v2` |
| `extracted` | `(72, 96, 4)` | "super mini-map": 4 planes (your team, their team, ball, active player), 255 where occupied. The shim returns float32 in [0, 255] |
| `pixels` / `pixels_gray` | `(72, 96, 3)` / `(72, 96, 1)` | rendered frames, downsampled. Needs rendering on every step (~10 steps/s), so it is impractically slow here |

With `controlled_players = N > 1`, a leading axis of size N is added. t-zero's
`discrete_control` stack (the DQN stack) accepts only flat vectors: image-like
encodings need a CNN network and a stack of your own (see
[adding-a-new-environment.md](adding-a-new-environment.md)).

### 5.3 Raw observations

`env.unwrapped.raw_observations()` returns a list with one dict per controlled
player. It is **exactly what the arena passes to your `Agent.act`**. This is
the input to design your submission's preprocessing on, and to record probes
with.

| Key | Shape | Meaning |
|---|---|---|
| `active` | int | index into `left_team` of the player this dict is for |
| `designated` | int | the player the engine designates for control (normally equal to `active`) |
| `left_team`, `right_team` | `(n, 2)` | positions |
| `left_team_direction`, `right_team_direction` | `(n, 2)` | movement per step |
| `left_team_roles`, `right_team_roles` | `(n,)` | 0 GK, 1 CB, 2 LB, 3 RB, 4 DM, 5 CM, 6 LM, 7 RM, 8 AM, 9 CF |
| `left_team_tired_factor`, `right_team_tired_factor` | `(n,)` | 0 = fresh, grows with fatigue |
| `left_team_yellow_card`, `right_team_yellow_card` | `(n,)` | booked players |
| `left_team_active`, `right_team_active` | `(n,)` | `False` for a sent-off player |
| `ball`, `ball_direction`, `ball_rotation` | `(3,)` | position, movement and spin per step |
| `ball_owned_team` | int | -1 nobody, 0 you, 1 opponent |
| `ball_owned_player` | int | index of the player holding the ball (-1 if nobody) |
| `game_mode` | int | 0 normal, 1 kick-off, 2 goal kick, 3 free kick, 4 corner, 5 throw-in, 6 penalty |
| `score` | `[yours, theirs]` | |
| `steps_left` | int | steps until the end of the episode |
| `sticky_actions` | `(10,)` | on/off for: left, top-left, top, top-right, right, bottom-right, bottom, bottom-left, sprint, dribble |

## 6. Actions

The default action set has 19 actions. Each step you pick one action per
controlled player.

| # | Action | # | Action |
|---|---|---|---|
| 0 | idle | 10 | high pass |
| 1 | left | 11 | short pass |
| 2 | top-left | 12 | shot |
| 3 | top | 13 | sprint |
| 4 | top-right | 14 | release direction |
| 5 | right | 15 | release sprint |
| 6 | bottom-right | 16 | sliding tackle |
| 7 | bottom | 17 | dribble |
| 8 | bottom-left | 18 | release dribble |
| 9 | long pass | | |

Things that surprise people:

- **Directions, sprint and dribble are sticky.** A direction stays on until
  another direction or *release direction* (14). Sprint and dribble stay on
  until their release action (15, 18). *Idle* (0) does not stop the player.
  The raw observation's `sticky_actions` shows what is currently on.
- Passes and shots are aimed with the direction currently held, and a sticky
  direction counts as held.
- The right team's actions are mirrored by the engine, so "right" always means
  "towards the opponent's goal" for both sides.

`other_config_options={"action_set": "v2"}` adds action 19, `builtin_ai`:
the engine's AI plays that player for this step. It is useful as a
"fall back to the AI" action during training, but **the arena accepts only
actions 0–18**. A submission would have to map 19 to something else.

## 7. Rewards

`scoring`: +1 when your team scores, -1 when it concedes. Nothing else.

`checkpoints` (added to `scoring`): +0.1 for each of 10 zones your player
reaches **while holding the ball**. The zones are distance bands around the
opponent's goal centre (1, 0), with thresholds 0.99, 0.90, 0.81, …, 0.19.
Details that matter:

- Each zone pays **once per episode**. A goal also pays any zones not yet
  collected, so the first goal is worth +2.0 in total with its zones (hence
  "episode return ≈ 2.0 means a goal" in the academy). Later goals in the same
  episode are worth +1.
- In the academy that is the whole episode. In `5_vs_5` and 11v11, once your
  team has collected the zones (or scored once), there is **no more shaping
  for the rest of the 3000 steps**. There, `checkpoints` is only a bonus for
  the first attack.
- It counts only when the controlled player is the one holding the ball, so
  passing to an AI teammate earns nothing.

With several controlled players, each player collects its own zones. See 8.2
for how the rewards are combined.

For any other shaping (possession, ball progress, defending, penalties for
bad passes), write a `gymnasium.RewardWrapper` that reads
`env.unwrapped.raw_observations()`. Rewards only affect training, never your
submission, so you are free here.

## 8. Which players you control

### 8.1 One player (the default) or several

With the default `controlled_players=1`, your agent does **not** control one
fixed player. It controls the **active player**: one control slot that the
engine hands from player to player as play moves, usually to the player
holding the ball or nearest to it (as in a football video game with one
gamepad). The game AI plays all your other players, and always your keeper.
`obs[97:108]` (or `raw_observations()[0]["active"]`) tells you who is active
right now.

`controlled_players=N` gives your agent N such slots: at any moment it
controls N different players of your team, and the game AI plays the rest,
keeper included. Each slot still moves between players (8.2).

| | `N = 1` | `N > 1` |
|---|---|---|
| observation | `(115,)` | `(N, 115)`: one row per controlled player |
| action space | `Discrete(19)` | `MultiDiscrete([19] * N)`: one action per row, same order |
| reward | the player's reward | **mean** over the N players (a goal is still +1) |
| `info["player_rewards"]` | not present | `(N,)` per-player rewards |
| `raw_observations()` | 1 dict | N dicts, same order as the rows |

`5_vs_5` has 4 outfield players per side, and the arena allows 1–4. The env
accepts larger N, but in `5_vs_5` the extra slots control nobody.

### 8.2 Which player is which

**A row is a control slot, not a fixed player.** The engine moves each slot to
the player it considers most relevant (usually the one nearest the ball) as
play goes on. Even with one controlled player and a policy that only idles,
control moves between players within the first 600 steps. The identity of row *i* is
`raw_observations()[i]["active"]`, or the active one-hot at indices 97–107 of
that row. Look up a role with `obs["left_team_roles"][obs["active"]]`.

This decides how you design per-player models: pick the network by active
player or role, never by row number. It also matters for credit assignment.
Row *i*'s reward in `player_rewards[i]` belongs to whoever held slot *i* on
that step.

Rewards: all N players get the same scoring reward (+1 / -1). Checkpoint
shaping is per player (section 7). The scalar `reward` is their mean. Use
`info["player_rewards"]` if you want per-player learning signals.

### 8.3 Training with several players in t-zero

The bundled DQN and its `discrete_control` stack are single-agent: they refuse
`N > 1` (the 2-D observation guard fires). A multi-player learner needs its
own wrapper stack and algorithm
([adding-a-new-algorithm.md](adding-a-new-algorithm.md)). The usual designs:

| Design | How it maps onto the env |
|---|---|
| **Shared policy (parameter sharing)** | Treat the N rows as a batch of N independent transitions per step: same network, per-row action, per-row reward from `player_rewards`. The simplest option, and it works with any N |
| **Independent learners** | One network per role (or per active player). Route row *i* by `active` / role, as above |
| **Joint policy** | One network reads all N rows (e.g. concatenated to `(N·115,)`) and outputs N actions, i.e. a `MultiDiscrete` policy (N heads of 19 logits). Reward = the team mean |
| **Centralized critic** (MAPPO-style) | Per-row actors, one critic on the joint observation |

A minimal parameter-sharing interaction loop, outside the t-zero training
loop:

```python
env = gym.make("GFootball/5_vs_5-v0", controlled_players=3, rewards="scoring,checkpoints")
obs, _ = env.reset(seed=0)                        # (3, 115)
for step in range(10_000):
    actions = policy(torch.as_tensor(obs)).argmax(-1).numpy()   # (3,) shared net over rows
    next_obs, reward, terminated, truncated, info = env.step(actions)
    per_player = info["player_rewards"]            # (3,)
    for i in range(3):                             # N transitions for one shared buffer
        buffer.add(obs[i], actions[i], per_player[i], next_obs[i], terminated)
    obs = next_obs if not (terminated or truncated) else env.reset()[0]
```

Remember that the arena runs your agent with one CPU thread and a 20 ms mean
budget for **all** your players together (`arena/rules.py`).

## 9. Choosing the opponent, and self-play

### 9.1 `opponent=`

By default the engine's AI plays the right team. `opponent` replaces that:

| Value | Right team played by |
|---|---|
| `None` / `"builtin"` | the engine AI, at the scenario's difficulty (0.05 in `5_vs_5`) |
| `"random"` | uniformly random actions for 1 player (the engine AI plays the rest of the team) |
| `"path/to/folder"` | an arena submission folder (`agent.py` + `manifest.yml`), loaded **in your process**. It controls `controlled_players` from its manifest |
| a Python object | anything with `controlled_players` (int), `reset()` and `act(observations) -> list[int]`; `close()` is optional |

The opponent sees **mirrored raw observations**, exactly as it would in the
arena: from its point of view it is `left_team` attacking x = +1. A policy
trained on the left can therefore play the right team unchanged, which is what
self-play needs. `reset()` is called at every kick-off (every `env.reset()`).
The opponent's rewards are not reported: your reward is always from your
team's side.

An opponent loaded from a folder shares your process's `sys.modules`. Give
helper modules distinctive names, or keep everything in `agent.py`. There are
no timeouts during training: a slow opponent just slows down your steps.
Measured with the template opponent, speed is about the same as the engine AI
(~280 vs ~310 steps/s).

### 9.2 Swapping the opponent during training

```python
env.unwrapped.set_opponent("teams/snapshot_0100")   # single env
envs.call("set_opponent", "teams/snapshot_0100")    # a Sync/AsyncVectorEnv (t-zero's `envs`)
```

The new opponent takes over at the **next `reset()`**, never in the middle of
an episode. If it controls a different number of players than the previous
one, the engine is rebuilt at that reset, which is slower than a normal
reset. Otherwise the swap costs nothing. With `AsyncVectorEnv`, a Python-object opponent is pickled to each
worker process, so it must be picklable. A folder path is always safe.

### 9.3 Self-play, leagues and curricula

t-zero does not ship a self-play or league algorithm. These are the building
blocks, and the design is yours.

**Snapshot opponent from memory.** Wrap a frozen copy of your network. The
opponent gets raw observations, so convert them the same way your submission
would:

```python
import copy, torch
from gfootball.env.wrappers import Simple115StateWrapper

class FrozenPolicy:
    controlled_players = 1

    def __init__(self, net):
        self.net = copy.deepcopy(net).eval()

    def reset(self):
        pass

    def act(self, observations):
        x = torch.as_tensor(Simple115StateWrapper.convert_observation(observations, True))
        with torch.no_grad():
            return self.net.act(x, deterministic=True).tolist()

# every K steps:
envs.call("set_opponent", FrozenPolicy(q_network))
```

**Snapshot opponent from disk.** Export a submission folder, e.g. with
`scripts/export_submission.py` (for runs it supports: `simple115v2`, one
player, no custom wrappers), and point `opponent` at it. This also tests your
submission's preprocessing along the way.

**League.** Keep a pool (`"builtin"`, `"random"`, `arena/template`, past
snapshots) and call `set_opponent` with a sampled entry every episode or every
few episodes. Sampling only recent snapshots risks forgetting how to beat old
strategies. Mixing in the engine AI keeps you honest against the tournament's
baseline anchors.

**Curriculum.** Change `env_id` between runs (academy, then `5_vs_5`), switch
the opponent as training goes on (`random`, then `builtin`, then snapshots),
or ease the game with a custom scenario (section 3.3: fewer players, lazy
opponents, a lower `right_team_difficulty`).

To measure progress, don't trust training reward against a moving opponent.
Play fixed opponents with the arena tools:

```bash
python -m arena.match my_team builtin -n 10
python -m arena.match my_team teams/snapshot_0050 -n 10
python -m arena.tournament teams/ --out results/ -n 10
```

## 10. Rendering and video

Rendering uses software GL (no GPU, no display needed) and runs at ~10
steps/s at 640×360. The shim turns it on only while something asks for frames
and turns it off again on the next step nobody rendered. A `render_mode`
env that isn't being recorded therefore runs at full speed. In t-zero,
`capture_video: true` records the final greedy evaluation (about 1 min per
400-step episode). Per-step training clips are disabled for this env. For
match videos, use `python -m arena.match A B --video out/`.

## 11. What t-zero supports and what you add

| Feature | Status |
|---|---|
| Single player, `simple115v2`, DQN | works out of the box: `configs/dqn_gfootball_empty_goal.yml` |
| Any scenario, reward variant, frame stack | `env_id` / `env_kwargs` |
| Export a single-player `simple115v2` run to a submission | `scripts/export_submission.py` |
| Opponents (`builtin` / `random` / folder / object), swapping at reset | in the env |
| Raw observations in the arena format, probes | `raw_observations()`, `arena/probe.py` |
| Several controlled players | env ready; **you add** the wrapper stack and algorithm (8.3) |
| SMM / pixels + CNN | env ready; **you add** a CNN network and a stack |
| PPO for discrete actions, self-play, leagues | **you add** (9.3) |
| Custom reward shaping | **you add** a reward wrapper (section 7) |
| Custom scenarios | `gfootball.scenarios.__path__` + `scenario=` (3.3) |
