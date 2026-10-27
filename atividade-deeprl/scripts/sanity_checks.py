"""Verificações antes de treinar qualquer coisa (Parte A da atividade).

1. Valida o ambiente com o verificador oficial do Gymnasium.
2. Mede a linha de base de uma política aleatória: sem ela, não dá para
   saber se "-400" é bom ou ruim.
3. Mostra o autoreset "next-step" dos ambientes vetorizados do Gymnasium 1.x,
   a fonte de um bug silencioso em replay buffers.

    python scripts/sanity_checks.py --env Acrobot-v1
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import _paths  # noqa: F401
import gymnasium as gym
import numpy as np
from gymnasium.utils.env_checker import check_env


def random_baseline(env_id: str, episodes: int = 20) -> None:
    env = gym.make(env_id)
    env.action_space.seed(0)
    returns, truncs = [], 0
    for ep in range(episodes):
        env.reset(seed=ep)
        total, done = 0.0, False
        while not done:
            _, r, terminated, truncated, _ = env.step(env.action_space.sample())
            total += float(r)
            done = terminated or truncated
        truncs += int(truncated and not terminated)
        returns.append(total)
    env.close()
    print(f"   retorno médio aleatório = {np.mean(returns):.1f} ± {np.std(returns):.1f}")
    print(f"   episódios encerrados por TRUNCAMENTO = {truncs}/{episodes}")


def autoreset_demo(env_id: str) -> None:
    envs = gym.make_vec(env_id, num_envs=2, vectorization_mode="sync")
    print(f"   modo de autoreset: {envs.metadata.get('autoreset_mode')}")
    obs, _ = envs.reset(seed=0)
    envs.action_space.seed(0)
    for t in range(2000):
        actions = envs.action_space.sample()
        next_obs, rewards, terminated, truncated, _ = envs.step(actions)
        ended = np.logical_or(terminated, truncated)
        if ended.any():
            i = int(np.argmax(ended))
            print(f"   passo {t}: sub-ambiente {i} terminou (terminated={terminated[i]}, truncated={truncated[i]})")
            print(f"   observação FINAL devolvida neste passo: {np.round(next_obs[i], 3)}")
            after_obs, after_r, after_term, after_trunc, _ = envs.step(envs.action_space.sample())
            print(f"   no passo seguinte, o sub-ambiente {i} devolve:")
            print(f"     observação de RESET = {np.round(after_obs[i], 3)}")
            print(f"     recompensa = {after_r[i]}, terminated={after_term[i]}, truncated={after_trunc[i]}")
            print("   → Essa transição (obs final → obs de reset) NÃO é real e não deve ir para o buffer.")
            break
        obs = next_obs
    envs.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env", default="Acrobot-v1")
    args = parser.parse_args()

    print(f"Gymnasium {gym.__version__}\n")
    print(f"1) Verificador do Gymnasium para {args.env}")
    env = gym.make(args.env)
    check_env(env.unwrapped)
    print(f"   OK | ação: {env.action_space} | observação: {env.observation_space}")
    print(f"   limite de passos (TimeLimit): {env.spec.max_episode_steps}")
    env.close()

    print("\n2) Linha de base aleatória")
    random_baseline(args.env)

    print("\n3) Autoreset em ambientes vetorizados")
    autoreset_demo(args.env)


if __name__ == "__main__":
    main()
