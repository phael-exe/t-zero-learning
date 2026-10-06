# Atividade: Proximal Policy Optimization (PPO)

Individual ou em dupla (identifique a dupla no relatório).

O PPO é o A2C da atividade anterior com quatro mudanças:

1. **GAE** no lugar do retorno de n passos: uma média ponderada de todos os retornos de
   n passos, controlada por $\lambda$.
2. **Reuso de amostras**: cada rollout é usado em `update_epochs` épocas de
   minibatches embaralhados, em vez de um único passo de gradiente.
3. **Razão de probabilidades**: como a política muda durante essas épocas, a perda
   compara a política nova com a que *coletou* os dados,
   $r(\theta) = \pi_\theta(a|s)/\pi_{\theta_\text{old}}(a|s)$.
4. **Clipping**: a perda ignora o ganho de empurrar $r$ para fora de
   $[1-\epsilon, 1+\epsilon]$, o que impede que essas épocas afastem a política demais
   dos dados.

A implementação segue a linha do A2C. Ela vive em
[algorithms/ppo.py](../algorithms/ppo.py). Vale a pena abrir esse arquivo ao lado de
[algorithms/a2c.py](../algorithms/a2c.py): o rollout, o log e os checkpoints são
iguais, e as diferenças são exatamente as quatro acima.

## Preparação

Igual às atividades anteriores. No `.env`, use um projeto novo no
[Weights & Biases](https://wandb.ai):

```
WANDB_PROJECT=ppo-assignment
```

O ambiente desta atividade é o `LunarLander-v3`. Ele já está na imagem Docker do curso;
fora dela, instale o Box2D com
`pip install swig && pip install --no-build-isolation box2d-py`.

**A rede é a da atividade de A2C.** O PPO usa a mesma rede ator-crítico
([networks/discrete_actor_critic.py](../networks/discrete_actor_critic.py)), que você
completou na Parte 3 do A2C. Mantenha a sua versão desse arquivo no seu fork. Um dos
testes abaixo falha com uma mensagem clara se ela ainda estiver em branco.

## Partes 1, 2 e 3: Implementação

Complete os três blocos marcados com `YOUR CODE HERE` em `algorithms/ppo.py`.

- **Parte 1, `compute_gae`**: as vantagens GAE e os retornos (alvo do crítico),
  $\delta_t = r_t + \gamma\,(1-d_{t+1})\,V(s_{t+1}) - V(s_t)$ e
  $\hat A_t = \delta_t + \gamma\lambda\,(1-d_{t+1})\,\hat A_{t+1}$, percorrendo o rollout
  de trás para frente; os retornos são $\hat A_t + V(s_t)$. **Atenção à convenção de
  `dones`** (a do CleanRL, diferente da do A2C): `dones[t]` é o `done` devolvido pelo
  passo *anterior* a $t$. Leia a docstring. Com $\lambda=1$ o retorno é exatamente o
  retorno de n passos da Parte 1 do A2C, e um dos testes verifica isso.
- **Parte 2, `compute_clipped_policy_loss`**: a perda do PPO,
  $-\frac{1}{B}\sum_i \min\big(r_i A_i,\ \mathrm{clip}(r_i, 1-\epsilon, 1+\epsilon)\,A_i\big)$,
  com $r_i = \exp(\log\pi_\text{new} - \log\pi_\text{old})$. As vantagens chegam já
  normalizadas. Como no A2C, pense no que é constante do ponto de vista do gradiente.
- **Parte 3, `approx_kl_and_clipfrac`**: os dois diagnósticos que dizem quanto a
  política se moveu durante a atualização: o KL aproximado
  $\overline{(r-1) - \log r}$ ([por quê](http://joschu.net/blog/kl-approx.html)) e a
  fração de amostras com $|r - 1| > \epsilon$. Nenhum dos dois deve carregar gradiente.

Cada bloco tem poucas linhas. Verifique com:

```bash
python -m pytest tests/test_ppo.py
```

Todos os testes devem passar antes de você começar a Parte 4. Depois treine (sempre a
partir da raiz do repositório):

```bash
python train.py --config ppo_lunarlander
```

Dica: se for rodar vários treinos ao mesmo tempo, limite as threads de cada um, senão
eles disputam CPU e todos ficam muito mais lentos:

```bash
OMP_NUM_THREADS=1 python train.py --config ppo_lunarlander --override seed=2
```

## Os gráficos com que você vai trabalhar

Os do A2C continuam lá (`charts/episodic_return_mean_last100`, `losses/value_loss`,
`losses/explained_variance`, `losses/entropy`, `charts/advantage_std`,
`losses/grad_norm`, `charts/SPS`, `eval/mean_return`). Os novos:

| Gráfico | O que ele diz |
|---|---|
| `losses/approx_kl` | quanto a política se afastou da que coletou o rollout, no último minibatch da atualização |
| `losses/clipfrac` | fração das amostras em que o clipping estava ativo, na média da atualização |
| `losses/policy_loss` | a perda clipada (o sinal e o ruído importam mais que o valor) |
| `charts/num_updates` | passos de gradiente acumulados desde o início |
| `charts/epochs_run` | épocas realmente executadas (menor que `update_epochs` só com `target_kl`) |
| `charts/learning_rate` | a taxa de aprendizado atual (decai até 0 com `anneal_lr`) |

Como antes: **nunca leia um gráfico isoladamente.** A curva de retorno diz *se* algo
deu errado; `approx_kl`, `clipfrac`, `entropy` e `explained_variance` costumam dizer
*o quê*.

## Parte 4: Resolver o LunarLander

O config [configs/ppo_lunarlander.yml](../configs/ppo_lunarlander.yml) traz os
hiperparâmetros padrão do `ppo.py` do CleanRL, que foram ajustados para o CartPole.
No LunarLander eles aprendem, mas **não resolvem** o ambiente. Seu trabalho é
resolvê-lo:

- **Critério**: `eval/mean_return` ≥ 200 **e** `charts/episodic_return_mean_last100`
  ≥ 200 no fim do treino, em **2 seeds** (`--override seed=2`).
- **Orçamento**: no máximo 2 milhões de passos de ambiente (`total_timesteps`). O
  config inicial usa 1 milhão (~8 min em um núcleo de CPU).
- Mude o que quiser no config, via `--override` ou criando o seu próprio arquivo em
  `configs/`. Não mude o código do algoritmo.

O que entra no relatório:

1. **Uma tabela com todas as mudanças** em relação ao config inicial: chave, valor
   antigo → valor novo.
2. **Para cada mudança, o motivo observado**: qual comportamento nos gráficos levou a
   ela (com o gráfico e o run identificados) e o que aconteceu depois dela. "Achamos
   que ficaria melhor" não é motivo; "o `approx_kl` ficava perto de zero e o
   `clipfrac` em 0, então cada rollout era subaproveitado, veja a figura X" é.
3. **Os runs que não deram certo** também contam, quando explicam o caminho até o
   config final.
4. As curvas de retorno do config inicial e do final, 2 seeds cada.

## Parte 5: PPO vs A2C

Compare o seu melhor PPO com o seu A2C no LunarLander. O A2C é o *seu* (o arquivo
`algorithms/a2c.py` da atividade anterior) e deve ser uma tentativa honesta: comece do
`a2c_cartpole` com `--override env_id=LunarLander-v3`, ajuste o que julgar necessário
e use o **mesmo orçamento de passos** do PPO. Não precisa resolver o ambiente com o
A2C. Rode 2 seeds de cada.

Discuta, sempre com gráficos (os dois algoritmos sobrepostos, runs identificados):

- **Eficiência amostral**: retorno × passos de ambiente (`global_step`). Quantos
  passos cada um leva para chegar a retorno 0, 100 e 200 (se chegar)?
- **Tempo de relógio**: retorno × tempo (no wandb, eixo x "Relative Time"), tempo
  total e `charts/SPS`. Qual dos dois é mais rápido por passo de ambiente, e por quê?
  Use `charts/num_updates` para explicar.
- **Mesmo orçamento**: com os mesmos passos de ambiente, qual chega mais longe? E se o
  orçamento fosse o mesmo *tempo* de relógio, a resposta mudaria?
- **Estabilidade**: variação entre as seeds e quedas ao longo do treino.
- **Mecanismo**: para cada diferença que você observou, qual das quatro mudanças do
  PPO (GAE, reuso de amostras, razão de probabilidades, clipping) a explica? Aponte o
  gráfico que sustenta cada afirmação.

## Entregáveis

Um relatório em PDF (máx. 4 páginas) com as Partes 4 e 5, contendo um link para o seu
fork com `algorithms/ppo.py`, `networks/discrete_actor_critic.py` e os configs que
você usou.
