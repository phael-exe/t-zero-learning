# Projeto: Google Research Football

O projeto tem duas entregas:

1. **Relatório** (PDF): como vocês modelaram o problema, o que testaram e o que
   os dados mostram. Esta parte ainda pode receber ajustes até perto do prazo.
2. **Código**: uma pasta de submissão com o seu agente, que joga um torneio
   contra os agentes das outras equipes. O formato está fixado abaixo.

A referência completa do ambiente (configurações, observações, ações,
recompensas, vários jogadores, adversários e self-play) está em
[docs/gfootball-environment.md](../docs/gfootball-environment.md).

## Entrega 1: relatório

Um relatório em PDF com as seções abaixo. Sejam diretos: o relatório é
avaliado pelo que ele demonstra, não pelo tamanho.

1. **Modelagem como MDP.** Estados (qual representação de observação e por
   quê), ações, recompensa, fim de episódio, fator de desconto. Quantos
   jogadores vocês controlam e por quê.
2. **O que vocês mudaram e testaram no ambiente.** Por exemplo: novas funções
   de recompensa, treino com currículo, self-play, liga de adversários,
   controle de vários jogadores, outras representações. Para cada item, o que
   foi feito e como foi testado.
3. **Pipeline de treino e arquitetura.** Algoritmo, arquitetura da rede,
   etapas do treino (cenários, adversários, trocas entre fases). Não é preciso
   justificar cada ajuste de hiperparâmetro; mudanças de pipeline e de
   arquitetura, sim.
4. **Resultados e interpretação.** O que funcionou, o que não funcionou e por
   quê.

Regras:

- **Todo resultado precisa de dados**: tabelas e gráficos (curvas de treino,
  placares, taxas de vitória contra `random`, `builtin` e versões anteriores do
  próprio agente, etc.), com os runs identificados.
- **Toda decisão precisa de motivo observado.** Quando mudaram algo (uma
  recompensa, uma fase do currículo, a arquitetura), mostrem qual comportamento
  motivou a mudança, com o dado ou gráfico correspondente, e o que aconteceu
  depois dela. "Achamos que ficaria melhor" não é motivo; "o agente ficava
  parado com a bola, veja a figura X" é.

## Entrega 2: código (pasta de submissão)

O agente final é uma **pasta de submissão**. Faremos um torneio todos contra
todos com o agente de cada outra equipe no cenário **`5_vs_5`** (3000 passos,
goleiros controlados pela IA do jogo). O treino é livre: qualquer cenário,
qualquer algoritmo, currículo, self-play, vários jogadores. Só a *interface* da
pasta é fixa.

Antes de entregar, rodem o mesmo verificador que a correção usa:

```bash
docker compose run --rm gfootball python -m arena.check my_team
```

Uma submissão que não retorna `PASS` não entrará no torneio.

### 1. O que vai na pasta

```
my_team/
  agent.py          # define a classe Agent (abaixo)
  manifest.yml      # nome da equipe, quantos jogadores vocês controlam
  weights/ ...      # o que o agente carregar
  probe.pkl         # opcional, fortemente recomendado (seção 4)
```

`manifest.yml`:

```yaml
team: my-team-name         # único; aparece nas tabelas
members: ["Ana Souza", "João Lima"]
controlled_players: 1      # 1..4 jogadores de linha controlados
deterministic: true        # false se act() amostra ações
description: DQN, curriculum empty_goal -> 5_vs_5, frame stack 4
```

`agent.py`:

```python
class Agent:
    def __init__(self, path):              # path = a sua pasta (pathlib.Path)
        ...                                # carregue os pesos: path / "weights" / ...
    def reset(self):                       # chamado a cada saída de bola
        ...                                # limpe frame stacks / estado de RNN aqui
    def act(self, observations):           # lista de dicts de obs brutas, uma por jogador controlado
        return [...]                       # lista de ints em [0, 19), mesmo tamanho e ordem
```

Comecem de um exemplo que funciona:

- `arena/template/`: um agente escrito à mão (corre até a bola, conduz em
  direção ao gol, chuta), sem pesos. `cp -r arena/template my_team`.
- Uma pasta gerada a partir de um run do t-zero (seção 3).

### 2. O que o agente vê

Cada observação é o dict **bruto** (raw) do gfootball, sempre **do ponto de
vista da sua equipe**: vocês são o `left_team` e atacam em direção a x = +1.
Isso vale para qualquer lado do campo em que vocês de fato joguem; o motor
espelha posições e ações para o time da direita. Os campos mais usados:

| Chave | Significado |
|---|---|
| `active` | índice (em `left_team`) do jogador a que esta observação se refere |
| `left_team`, `right_team` | posições `(n, 2)`, x ∈ [-1, 1], y ∈ [-0.42, 0.42] |
| `left_team_direction`, `right_team_direction` | velocidades `(n, 2)` |
| `left_team_roles` | papel de cada jogador (0 = goleiro, …) |
| `ball`, `ball_direction` | posição / velocidade `(3,)` (z = altura) |
| `ball_owned_team` | -1 ninguém, 0 vocês, 1 adversário |
| `ball_owned_player` | índice do jogador com a bola |
| `game_mode` | 0 normal, 1–6 saída de bola / tiro de meta / falta / escanteio / lateral / pênalti |
| `score`, `steps_left` | `[seus, deles]`, passos até o fim |
| `sticky_actions` | `(10,)` quais ações persistentes estão ativas para este jogador |

As ações são o conjunto padrão do gfootball:

| | | | | |
|---|---|---|---|---|
| 0 idle | 1 left | 2 top-left | 3 top | 4 top-right |
| 5 right | 6 bottom-right | 7 bottom | 8 bottom-left | 9 long pass |
| 10 high pass | 11 short pass | 12 shot | 13 sprint | 14 release direction |
| 15 release sprint | 16 sliding | 17 dribble | 18 release dribble | |

Se vocês treinaram com o vetor `simple115v2` (a config padrão do t-zero), os
mesmos 115 números saem de

```python
from gfootball.env.wrappers import Simple115StateWrapper
x = Simple115StateWrapper.convert_observation(observations, True)   # (n_players, 115)
```

O "minimapa" SMM é `gfootball.env.observation_preprocessing.generate_smm(observations)`.

**No torneio, a única entrada é o dict bruto.** Qualquer representação que
possa ser calculada a partir dele pode ser usada, desde que a conversão rode
dentro de `act`: `simple115v2` e SMM servem. **Pixels não estão disponíveis**:
as partidas rodam sem renderização, então o dict não traz o campo `frame`, e
renderizar não caberia no limite de tempo de `act` (seção 6). Treinem com pixels
só se tiverem um plano para jogar sem eles.

#### Controlando mais de um jogador

Usem `controlled_players: N` (1–4). `act` então recebe N observações e devolve
N ações. O resto da equipe, e sempre o goleiro, é jogado pela IA do jogo.
Controlar mais jogadores **não é automaticamente mais forte**: a IA do jogo é
um companheiro razoável. Escolher N faz parte do projeto.

Tudo abaixo cabe na mesma interface:

| Projeto | Em `act` |
|---|---|
| um modelo, só o jogador ativo | `controlled_players: 1` |
| um modelo compartilhado por todos os jogadores | a mesma rede em cada `observations[i]` |
| um modelo por papel / por jogador | escolham o modelo por `obs["active"]` ou `obs["left_team_roles"][obs["active"]]` |
| uma política conjunta | leiam todas as observações de uma vez, devolvam todas as ações |

**Nunca identifiquem jogadores pela posição na lista.** Com menos de 4
jogadores controlados, o motor passa os seus slots para jogadores diferentes
durante a partida. Usem sempre `obs["active"]`.

### 3. De um run do t-zero a uma submissão

Se vocês treinaram com o t-zero em um env `GFootball/*`, `simple115v2`, um
jogador controlado, sem wrappers próprios:

```bash
docker compose run --rm gfootball python scripts/export_submission.py \
    runs/<exp>/<run_dir> --out my_team --team my-team-name
docker compose run --rm gfootball python -m arena.check my_team
```

O export embute a rede (incluindo as estatísticas de `use_obs_norm`) em
`weights/policy.pt` (TorchScript), escreve `agent.py` e `manifest.yml` e grava
um probe (seção 4). Ele recusa runs que não consegue reproduzir fielmente
(outra representação, vários jogadores, wrappers que alteram observações).
Nesses casos, escrevam o `agent.py` vocês mesmos.

O exportador reconstrói a rede do mesmo jeito que o treino, a partir das chaves
`network` e `network_kwargs` do `config.yml` do run. Uma rede própria também
exporta: adicionem o arquivo em `networks/`, exportem a classe em
`networks/__init__.py` e referenciem-na na config. Ela precisa manter a
interface `act(obs, deterministic=True)` do framework.

### 4. Pré-processamento, normalização e o probe

Regra: **tudo o que é necessário para transformar uma observação bruta em ação
precisa estar dentro da pasta.** O verificador roda o agente em um processo
limpo que só consegue importar a pasta e os pacotes da imagem (torch, numpy,
gfootball, …), **não o repositório t-zero**. Copiem para a pasta todo código de
que precisarem.

- **Estatísticas de normalização** precisam ser salvas junto com os pesos: como
  buffers na rede (`networks/normalization.py::RunningMeanStd` faz isso, e o
  `use_obs_norm` do t-zero já faz), ou como arquivo em `weights/`. Estatísticas
  guardadas dentro de um wrapper do Gymnasium (`NormalizeObservation`) **não**
  são salvas com o modelo. Esse é o bug clássico "funciona no treino, lixo no
  torneio".
- **Frame stacks / estado de RNN** recomeçam do zero em `reset()`.
- **Probe**: gravem alguns pares `(observations, actions)` *no código de
  treino*, onde vocês sabem que o agente funciona, e salvem como `probe.pkl`.
  O verificador os reproduz no `Agent` submetido e informa quantas ações ele
  reproduz. Com o env do t-zero, as observações no formato da arena são
  `env.unwrapped.raw_observations()`:

  ```python
  from arena.probe import ProbeRecorder
  rec = ProbeRecorder(controlled_players=1)
  obs, _ = env.reset(seed=0); rec.new_episode()
  for _ in range(300):
      action = my_policy(obs)                      # a política do lado do treino
      rec.add(env.unwrapped.raw_observations(), [action])
      obs, *_ = env.step(action)
  rec.save("my_team/probe.pkl")
  ```

  Abaixo de 90 % reproduzido (agentes determinísticos) é FAIL. O verificador
  também avisa quando uma única ação passa de 90 % de tudo o que o agente faz
  nas partidas de teste.

### 5. Os três comandos da arena

Todos dentro da imagem (`docker compose run --rm gfootball ...`). Todas as
opções e os arquivos de saída estão em [docs/arena.md](../docs/arena.md).

| Comando | O que faz |
|---|---|
| `python -m arena.check my_team` | o portão: estrutura, manifest, tamanho, tempo de carga, probe, uma partida completa contra `random` e uma contra `builtin`, velocidade |
| `python -m arena.match my_team other_team -n 10` | confronto direto, lados alternados; `random` / `builtin` também servem de adversário; `--video out/` gera mp4s (lento), `--dump out/` grava replays |
| `python -m arena.tournament teams/ --out results/ -n 10` | todos contra todos entre as pastas em `teams/` mais as âncoras, com tabela e ratings de Bradley–Terry |

Cada partida usa uma seed aleatória nova. Não esperem reproduzir uma partida
gol a gol; joguem partidas suficientes para ver uma tendência.

### 6. Regras (aplicadas pelo verificador e pelo torneio)

| Regra | Limite |
|---|---|
| Ambiente | a imagem Docker da disciplina, só CPU, **1 thread de CPU por agente**, sem rede |
| Pacotes | só o que a imagem tem; nada instalado em tempo de execução |
| Tamanho da pasta | 100 MB |
| `Agent(path)` | 30 s |
| `act` | média ≤ 20 ms por partida (todos os seus jogadores juntos); qualquer chamada isolada ≤ 1 s (primeira chamada: 10 s) |
| Arquivos | apenas leitura |
| Crash, timeout, ação inválida | vocês perdem aquela partida, registrada como derrota por 0–3 |

Os valores estão em `arena/rules.py`.

### 7. Ganchos de treino que o ambiente oferece

A referência completa está em
[docs/gfootball-environment.md](../docs/gfootball-environment.md). Esta seção é
a versão curta.

Os env ids `GFootball/<cenário>-v0` cobrem os 11 cenários da academy mais
`1_vs_1_easy`, `5_vs_5`, `11_vs_11_easy_stochastic`, `11_vs_11_stochastic` e
`11_vs_11_hard_stochastic`. Para treinar no cenário do torneio contra a IA do
jogo, basta `env_id: GFootball/5_vs_5-v0` na config, ou, para um run só:

```bash
python train.py --config dqn_gfootball_empty_goal --override env_id=GFootball/5_vs_5-v0
```

Kwargs do env (`env_kwargs:` na config, ou `gym.make(..., key=value)`):

| kwarg | Efeito |
|---|---|
| `rewards` | `scoring` (±1 por gol) ou `scoring,checkpoints` (+0.1 de shaping por zona) |
| `representation` | `simple115v2` (padrão), `extracted` (SMM), `pixels`, `raw`… (a pilha padrão espera um vetor plano; `pixels` não existe no torneio, veja a seção 2) |
| `controlled_players` | N > 1: obs `(N, …)`, ações `MultiDiscrete`, recompensa = média entre os seus jogadores. Exige pilha de wrappers / algoritmo próprios |
| `opponent` | `builtin` (padrão: a IA do jogo), `random`, ou **o caminho de uma pasta de submissão**: esse agente joga pelo outro time, vendo a visão espelhada exatamente como na arena |

E em tempo de execução, `env.unwrapped.set_opponent(spec)` (em vector envs:
`envs.call("set_opponent", spec)`) troca o adversário a partir do próximo
episódio. Com isso dá para montar, por exemplo:

- **self-play**: exportem um snapshot do agente a cada K passos e apontem
  `opponent` para ele;
- **uma liga**: sorteiem o adversário a cada episódio de um conjunto de
  snapshots e agentes scriptados;
- **um currículo**: troquem `env_id` (academy → `5_vs_5`) ou o adversário
  (`random` → `builtin` → seus snapshots) ao longo do treino.

Nada disso vem pronto no t-zero. É de vocês projetar, e esta é a base para
isso.

Um adversário carregado via `opponent:` roda **dentro** do processo de treino:
é rápido, mas os módulos auxiliares dele compartilham `sys.modules` com os
seus. Deem nomes distintos aos módulos auxiliares, ou mantenham tudo em
`agent.py`.

---

**Para mais detalhes leiam `docs/gfootball-environment.md`**

**Quaisquer problemas com o framework, ou melhorias desejadas, sinta-se a vontade para abrir um *issue* no Github.**