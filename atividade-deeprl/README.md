# Boas Práticas em Aprendizado por Reforço Profundo: atividade prática

Projeto de apoio à aula. O enunciado está em **[ATIVIDADE.md](ATIVIDADE.md)**.

O que tem aqui:

- **DQN** implementado em PyTorch, legível, para o `Acrobot-v1`.
- **PPO** do Stable-Baselines3 para o `CartPole-v1`, com normalização de observações.
- O mesmo protocolo de avaliação para os dois: ambiente separado, sementes fixas e política determinística.
- Agregação de várias sementes com IQM e intervalo de confiança de 95%.

Tudo roda em **CPU**. Não é preciso GPU.

## 1. Preparar o ambiente (Docker, recomendado)

Pré-requisito: [Docker Desktop](https://docs.docker.com/get-docker/) no Windows ou no macOS, ou Docker Engine com o plugin `compose` no Linux. O mesmo comando serve para Intel/AMD e para Apple Silicon.

```bash
# dentro da pasta do projeto
docker compose build            # 5–10 min na primeira vez; roda os testes no final
docker compose run --rm lab     # abre um terminal dentro do container
```

Se o build terminar com `passed`, está tudo certo. Os comandos das próximas seções são digitados **dentro** desse terminal.

Para acompanhar o treino, abra **outro** terminal no seu computador, na pasta do projeto:

```bash
docker compose up -d tensorboard
```

Depois acesse <http://localhost:6006>. Para desligar, use `docker compose down`.

> **Windows:** use o PowerShell ou o terminal do Docker Desktop, e deixe o projeto numa pasta sem acentos ou espaços.
> **Linux:** se aparecer `permission denied` no socket do Docker, rode com `sudo` ou adicione seu usuário ao grupo `docker`.

## 2. Alternativa sem Docker

Use esta opção se você não puder instalar o Docker (por exemplo, no laboratório). Ela exige Python 3.10 a 3.12.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.9.1
pip install -r requirements.txt
pytest -q tests                     # deve terminar com "passed"
```

No Windows sem Docker, rode `set PYTHONPATH=src` antes dos comandos. No Linux e no macOS, rode `export PYTHONPATH=src`. Os scripts já tentam encontrar `src/` sozinhos.

## 3. Comandos principais

```bash
# verificações antes de treinar
python scripts/sanity_checks.py --env Acrobot-v1

# uma execução (uma semente)
python scripts/train.py --config configs/dqn_acrobot.yaml --seed 0

# várias sementes; pode interromper e retomar
python scripts/run_experiment.py --config configs/dqn_acrobot.yaml --seeds 0 1 2 3 4

# variante, sem editar arquivo nenhum
python scripts/run_experiment.py --config configs/dqn_acrobot.yaml --seeds 0 1 2 3 4 \
    --set variant=sem_bootstrap --set hyperparameters.bootstrap_on_truncation=false

# tabela com IQM + IC 95% e curvas com faixa de confiança
python scripts/aggregate.py --experiment results/runs/dqn_acrobot

# reavaliar e assistir a um agente salvo
python scripts/evaluate.py --run results/runs/dqn_acrobot/padrao/seed0
python scripts/enjoy.py    --run results/runs/dqn_acrobot/padrao/seed0   # gera agente.gif
```

Qualquer chave do YAML pode ser trocada com `--set secao.chave=valor`. **Sempre** dê um nome novo com `--set variant=...` ao mudar um hiperparâmetro. Assim cada configuração fica numa pasta própria e nada é sobrescrito.

## 4. Onde ficam os resultados

```text
results/runs/<experimento>/<variante>/seed<N>/
├── config.yaml        configuração efetiva desta execução
├── metadata.json      versões, commit do git e comando usado
├── events.out.tfevents.*   logs do TensorBoard
├── eval.csv           avaliações periódicas
├── final.json         avaliação final (20 episódios)
└── model.pt | model.zip (+ vecnormalize.pkl no PPO)
```

## 5. Estrutura do projeto

```text
configs/            base.yaml (herdado) + um arquivo por experimento
src/rlpraticas/
├── agents/         dqn_agent.py: agente e alvo de Bellman (compute_td_target)
├── models/         q_network.py
├── trainers/       dqn.py (laço próprio) e ppo_sb3.py (Stable-Baselines3)
├── utils/          replay_buffer.py, reproducibility.py, stats.py
├── envs.py         mesma construção de ambiente no treino e na avaliação
└── evaluation.py   protocolo de avaliação
scripts/            pontos de entrada (train, run_experiment, aggregate, evaluate, enjoy, sanity_checks)
tests/              pytest
```

## 6. Problemas comuns

| Sintoma | Causa provável |
|---|---|
| `FileExistsError: ... já existe` | Você repetiu a mesma variante e semente. Use outro `--set variant=` ou `--overwrite`. |
| TensorBoard vazio | O TensorBoard só lê `results/runs`. Confira se o treino está gravando lá. |
| `ModuleNotFoundError: rlpraticas` (sem Docker) | Falta `PYTHONPATH=src`. |
| Treino muito lento | Feche outros programas. Rode as sementes em terminais separados se tiver vários núcleos. |

Versões fixadas: Python 3.12, PyTorch 2.9.1 (CPU), Gymnasium 1.3.0 e Stable-Baselines3 2.9.0.
