# Atividade: Soft Actor-Critic (SAC)

Individual ou em dupla (identifique a dupla no relatório).

O SAC junta peças que você já conhece e acrescenta uma ideia nova:

1. **Off-policy com replay buffer**, como o DQN: cada transição vai para um
   buffer e é reutilizada em muitos passos de gradiente.
2. **Ações contínuas com um ator**, como o PPO contínuo: a política é uma
   gaussiana, mas aqui ela passa por um $\tanh$ para ficar dentro dos limites
   do ambiente.
3. **Dois críticos $Q$** e o mínimo entre eles no alvo, contra a
   superestimação que você viu no DQN.
4. **Máxima entropia** (a ideia nova): o agente maximiza
   $\sum_t r_t + \alpha\,\mathcal{H}(\pi(\cdot|s_t))$. O coeficiente $\alpha$
   (temperatura) é ajustado automaticamente para manter a entropia da política
   perto de um alvo.

A implementação segue o `sac_continuous_action.py` do CleanRL e vive em
[algorithms/sac.py](../algorithms/sac.py). Desta vez quase tudo já vem pronto:
o buffer, as perdas do ator e da temperatura, as redes-alvo e o laço de
treino. Leia o arquivo inteiro antes de começar. As Partes 4 e 5 pedem que
você explique o que você viu apontando linhas desse arquivo.

## Preparação

Igual às atividades anteriores. No `.env`, use um projeto novo no
[Weights & Biases](https://wandb.ai):

```
WANDB_PROJECT=sac-assignment
```

O ambiente principal é o `Hopper-v5` (MuJoCo), que já está na imagem Docker do
curso e no `requirements.txt`. Para gravar o vídeo da avaliação em uma máquina
sem tela, use `MUJOCO_GL=egl`:

```bash
MUJOCO_GL=egl python train.py --config sac_hopper
```

**O SAC é lento por passo de ambiente**: ele faz um passo de gradiente a cada
passo do ambiente (o PPO faz vários por rollout, mas raramente). Em um núcleo
de CPU, conte com ~15–40 min por treino de 100 mil passos, dependendo da
máquina e da configuração. Se for rodar vários treinos ao mesmo tempo, limite
as threads de cada um, senão eles disputam CPU e todos ficam muito mais lentos:

```bash
OMP_NUM_THREADS=1 python train.py --config sac_hopper --override seed=2
```

Treinos com `checkpoint_every` podem ser retomados se forem interrompidos
(`python train.py --resume runs/<exp>/<run>`). O buffer vai junto no checkpoint.

## Partes 1 e 2: Implementação

Complete os dois blocos marcados com `YOUR CODE HERE` em `algorithms/sac.py`.

- **Parte 1, `squashed_gaussian_sample`**: amostrar uma ação de uma gaussiana
  "espremida" pelo $\tanh$ e calcular a sua log-probabilidade. A amostra
  $u \sim \mathcal{N}(\mu, \sigma)$ deve usar o **truque da
  reparametrização** ($u = \mu + \sigma\,\epsilon$, `rsample()`), para que o
  gradiente da perda do ator chegue em $\mu$ e $\sigma$ *através* da ação.
  A ação é $a = \tanh(u)\cdot\text{scale} + \text{bias}$, e pela mudança de
  variáveis
  $\log\pi(a|s) = \sum_i \big[\log\mathcal{N}(u_i) - \log(\text{scale}_i\,(1-\tanh^2 u_i) + 10^{-6})\big]$.
  Leia a docstring: ela diz as formas esperadas de cada saída.
- **Parte 2, `compute_soft_q_target`**: o alvo dos dois críticos,
  $y = r + \gamma\,(1-d)\,\big(\min(Q'_1, Q'_2) - \alpha\log\pi(a'|s')\big)$,
  com $a'$ amostrado da política *atual* em $s'$ e $Q'$ as redes-alvo. O
  $-\alpha\log\pi$ é o bônus de entropia do próximo estado.

Cada bloco tem poucas linhas. Verifique com:

```bash
python -m pytest tests/test_sac.py
```

Todos os testes devem passar. Depois rode o teste de sanidade no Pendulum
(~5 min): a avaliação final (`eval/mean_return`, impressa no fim do treino)
deve ficar acima de −200. Se não ficar, o erro está no seu código, não nos
hiperparâmetros. (O `charts/episodic_return_mean_last100` fica bem mais
baixo aqui: a janela de 100 episódios cobre o treino inteiro, inclusive a
fase de ações aleatórias.)

```bash
python train.py --config sac_pendulum
```

## Os gráficos com que você vai trabalhar

Os de sempre continuam lá (`charts/episodic_return_mean_last100`,
`charts/SPS`, `charts/num_updates`, `eval/mean_return`). Os do SAC:

| Gráfico | O que ele diz |
|---|---|
| `losses/qf1_values`, `losses/qf2_values` | o $Q$ médio que cada crítico dá às transições do buffer |
| `losses/qf1_loss`, `losses/qf2_loss` | o erro de cada crítico em relação ao alvo da Parte 2 |
| `losses/actor_loss` | $\overline{\alpha\log\pi - \min(Q_1, Q_2)}$ |
| `losses/alpha` | a temperatura atual |
| `losses/entropy` | $-\overline{\log\pi}$, a entropia estimada da política; compare com `charts/target_entropy` |
| `losses/alpha_loss` | a perda da temperatura (só com `autotune`) |
| `charts/q_estimate_ep_start` | o $Q$ que os críticos dão ao primeiro par (estado, ação) de um episódio |
| `charts/soft_return_ep_start` | o que esse episódio *realmente* rendeu a partir dali, nas mesmas unidades do $Q$ (recompensas descontadas mais o bônus de entropia) |
| `charts/q_minus_return` | a diferença entre os dois: positivo = os críticos superestimam |
| `eval/mean_return` | avaliação final com a ação **média** (determinística), 10 episódios |
| `eval/mean_return_sampled` | a mesma avaliação, com ações **amostradas** |

Como antes: **nunca leia um gráfico isoladamente.**

## Parte 3: Eficiência amostral no Hopper

O config [configs/sac_hopper.yml](../configs/sac_hopper.yml) traz os
hiperparâmetros padrão do CleanRL, com um orçamento apertado de **100 mil
passos de ambiente**. Com eles o SAC aprende, mas devagar. Seu trabalho é
tentar fazê-lo aprender mais com os mesmos passos e **mostrar se conseguiu**.

Não há um retorno-alvo. No Hopper com 100 mil passos, dois seeds do mesmo
config podem terminar muito longe um do outro, às vezes mais longe do que o
efeito de uma mudança. Por isso, aqui um run sozinho não prova nada:

- **3 seeds por config** (`--override seed=1`, `seed=2`, `seed=3`), incluindo
  o config inicial, que é a sua referência. Os 3 seeds podem rodar em
  paralelo (veja `OMP_NUM_THREADS` acima).
- **3 a 4 configs candidatos** além do inicial. Para cada um, olhe no fim do
  treino `eval/mean_return` e `charts/episodic_return_mean_last100`: a média
  e o intervalo (mín–máx) entre os 3 seeds.
- **Uma mudança só conta como melhora se a diferença for maior que a
  variação entre seeds.** Se os intervalos se sobrepõem quase inteiros, ela
  é indistinguível do ruído, e dizer isso com os números é uma conclusão
  válida. A nota vem do raciocínio, não do retorno.
- **Orçamento**: exatamente 100 mil passos de ambiente (`total_timesteps`).
  O tempo de relógio é livre: você pode, por exemplo, fazer mais passos de
  gradiente por passo de ambiente (`sac.updates_per_step`), e isso custa
  tempo.
- Mude o que quiser no config, via `--override` ou criando o seu próprio
  arquivo em `configs/`. Não mude o código do algoritmo.

O seu **config final** é o candidato com a melhor média, ou o config inicial
se nenhum candidato passou do ruído. Ele é o ponto de partida das Partes 4 e 5.

O que entra no relatório (~1 página):

1. **Uma tabela com todos os candidatos**, uma linha por mudança em relação
   ao config inicial (a primeira linha é o próprio config inicial):
   - chave e valor antigo → valor novo;
   - **uma frase com o motivo observado**, citando o run em que você o viu
     (por exemplo: "no run X, `losses/alpha` caiu para 0,02 antes de 20 mil
     passos, então...");
   - `eval/mean_return` e `charts/episodic_return_mean_last100` no fim, como
     média (mín–máx) dos 3 seeds;
   - o veredito: **melhora**, **indistinguível do ruído** ou **piora**.

   Não precisa de um gráfico por mudança: os runs citados no wandb são a
   evidência.
2. **Uma figura**: as curvas de retorno do config inicial e do final,
   3 seeds cada (as curvas dos seeds, ou média com faixa mín–máx). Se o
   final for o próprio inicial, mostre o inicial e o candidato que chegou
   mais perto.

## Antes das Partes 4 e 5: suas previsões

Antes de rodar cada experimento das Partes 4 e 5, escreva em poucas frases o
que você espera ver e por quê. No relatório, cada experimento começa por essa
previsão e termina dizendo se ela se confirmou e, se não, o que estava errado
no seu raciocínio. Errar a previsão não tira nota.

## Parte 4: O que cada peça faz

Parta do seu config final da Parte 3 e rode, 2 seeds cada:

1. **Sem ajuste automático da temperatura**: `sac.autotune=false` com um
   `sac.alpha` fixo. Escolha esse valor olhando a curva de `losses/alpha` dos
   seus runs da Parte 3 e justifique a escolha.
2. **Um crítico só**: `sac.clipped_double_q=false`. O alvo e a perda do ator
   passam a usar só $Q_1$ no lugar de $\min(Q_1, Q_2)$.

No relatório (~1 página), para cada ablação: **uma figura** (pode ter 2–3
painéis) com o config da Parte 3 e a ablação sobrepostos, e 3 a 5 frases:
a sua previsão, o que aconteceu e qual linha de `algorithms/sac.py` explica
o efeito. Gráficos úteis: `losses/alpha`, `losses/entropy` contra
`charts/target_entropy` e o retorno na ablação 1; `charts/q_minus_return`,
`losses/qf1_values` e o retorno na ablação 2.

## Parte 5: SAC vs PPO

Compare o seu melhor SAC com o PPO contínuo do repositório
([algorithms/ppo_continuous_action.py](../algorithms/ppo_continuous_action.py),
completo) no `Hopper-v5`, com o **mesmo orçamento de passos**. Use o PPO
como está, sem ajuste, 2 seeds:

```bash
python train.py --config ppo_pendulum --override env_id=Hopper-v5 total_timesteps=100000 ppo_continuous_action.gamma=0.99 track=true seed=1
```

No relatório (~1 página), com os dois algoritmos sobrepostos nos gráficos:

- **Eficiência amostral**: retorno × passos de ambiente (`global_step`),
  2 seeds de cada.
- **Tempo de relógio**: retorno × tempo (no wandb, eixo x "Relative Time").
  Qual dos dois chega mais longe com o mesmo *tempo*? Use `charts/SPS` e
  `charts/num_updates` para explicar, e diga como o seu `updates_per_step`
  da Parte 3 muda a resposta.
- **Mecanismo**: para cada diferença que você observou, qual peça do SAC a
  explica (replay off-policy, entropia, dois críticos, ator
  reparametrizado)? Inclua aqui uma estimativa de quantas vezes, em média,
  cada transição entra em um passo de gradiente em cada algoritmo.

## Entregáveis

Um relatório em PDF (máx. 4 páginas: ~1 por parte, e o resto para
introdução, links e folga) com as Partes 3, 4 e 5, contendo:

- um link para o seu fork, com `algorithms/sac.py` e os configs que você usou;
- **um link para o seu projeto do wandb** (público, ou compartilhado com os
  professores). Todo gráfico do relatório deve identificar runs que existem
  nesse projeto, e a tabela da Parte 3 deve bater com os configs registrados
  neles.
