# Atividade: Boas Práticas em DeepRL

**Objetivo.** Ver na prática, com experimentos controlados, cada boa prática da aula:
- verificações antes de treinar;
- a distinção terminated vs truncated;
- avaliação separada do treino;
- monitoramento pelas métricas;
- comparação estatística entre configurações.

**Entrega.** Entregue a pasta `results/` compactada, sem os arquivos `model.*`, com os `summary*.md` e as figuras `curves_*.png`. Junte um relatório curto, em PDF ou Markdown, com até 5 páginas. O relatório responde às perguntas marcadas com **Q**. Sempre que uma resposta citar um número, ele deve vir de um arquivo em `results/`.

**Tempo de computação.** O tempo de cada parte está indicado no seu título. São estimativas para um notebook comum com 1 núcleo. Com mais núcleos, rode sementes em paralelo, em terminais separados, com `--seeds 0 1` em um e `--seeds 2 3 4` no outro. Deixe as Partes B e D rodando enquanto responde às outras.

Antes de começar, siga o [README](README.md) até o `docker compose build` terminar com `passed`.

---

## Parte A: Verificações antes de treinar (5 min)

```bash
python scripts/sanity_checks.py --env Acrobot-v1
python scripts/sanity_checks.py --env CartPole-v1
pytest -q tests
```

**Q1.** Qual é o retorno médio de uma política aleatória em cada ambiente? Por que esse número precisa aparecer em todo gráfico de resultados?

**Q2.** Na política aleatória do Acrobot, que fração dos episódios termina por **truncamento**? O que isso antecipa sobre a importância do tratamento de `truncated` no início do treino?

**Q3.** O item 3 do script mostra o autoreset dos ambientes vetorizados do Gymnasium 1.x. Explique com suas palavras qual transição aparece no passo seguinte ao fim de um episódio. Explique também por que guardá-la num replay buffer seria um erro.

**Q4.** Leia `compute_td_target` em `src/rlpraticas/agents/dqn_agent.py` e os testes em `tests/test_basics.py`. Por que `terminated` tem prioridade quando `terminated` e `truncated` são verdadeiros ao mesmo tempo?

---

## Parte B: Terminated vs truncated no DQN (20 a 30 min de computação)

Rode as duas variantes com 5 sementes cada:

```bash
python scripts/run_experiment.py --config configs/dqn_acrobot.yaml --seeds 0 1 2 3 4

python scripts/run_experiment.py --config configs/dqn_acrobot.yaml --seeds 0 1 2 3 4 \
    --set variant=sem_bootstrap --set hyperparameters.bootstrap_on_truncation=false

python scripts/aggregate.py --experiment results/runs/dqn_acrobot
python scripts/aggregate.py --experiment results/runs/dqn_acrobot --metric overestimation
```

**Q5.** Compare as variantes `padrao` e `sem_bootstrap` usando `summary.md` e `curves_mean_return.png`. Os intervalos de confiança se sobrepõem? Sua conclusão sobre a diferença é forte ou fraca? Agora compare as duas variantes em `curves_overestimation.png`. Os valores previstos diferem mais do que o retorno? Responda com base no que os seus dados mostram, e não no que você esperava ver.

**Q6.** No Acrobot a recompensa é −1 por passo. Ao tratar o truncamento como terminal, o alvo vira só `r = −1`. Isso **subestima** ou **superestima** o valor dos estados próximos do limite de tempo? Compare com um ambiente de recompensa positiva por passo, como o CartPole. O sentido do viés é sempre o mesmo?

**Q7.** No TensorBoard, observe `rollout/truncated_episode_frac`, a fração dos últimos 100 episódios que acabaram por truncamento. Em que fase do treino o truncamento é frequente? Relacione com a sua resposta da Q5. Observe também `train/truncated_frac`, a fração de **transições** truncadas em cada lote do replay buffer. Por que ela é tão menor que a anterior? Em que situações o tratamento do truncamento passaria a importar mais?

*Extra (opcional):* repita a Parte B com `--set max_episode_steps=200`, usando outros nomes de variante. Truncamentos mais frequentes mudam a conclusão?

---

## Parte C: Avaliação separada do treino (10 min, usa as execuções da Parte B)

No TensorBoard, abra `rollout/ep_rew_mean` (curva de treino) e `eval/mean_return` (curva de avaliação) da variante `padrao`.

**Q8.** As duas curvas diferem em quais trechos? Explique a diferença pelo que acontece com o ε (`rollout/exploration_rate`).

**Q9.** Rode `evaluate.py` em uma execução com `--epsilon 0.0` e depois com `--epsilon 0.05`. Quanto muda o retorno? Qual dos dois números você reportaria como "desempenho do agente" e por quê?

**Q10.** Veja `eval/mean_predicted_value`, `eval/mean_discounted_return` e `eval/overestimation`. Eles comparam o `max Q(s₀, a)` previsto com o retorno descontado observado. A rede superestima? Isso muda ao longo do treino? Agora veja `train/q_max`. No Acrobot toda recompensa é −1: que valores de Q são possíveis? O que significa um `q_max` positivo? Que papel tem o Double DQN (`double_dqn: true`) aqui, e ele basta para evitar isso?

**Q11.** Por que a curva `train/loss` sozinha não diz se o agente está melhorando? Mostre um trecho dos seus logs que ilustre isso.

---

## Parte D: PPO e monitoramento (15 a 25 min de computação)

```bash
python scripts/run_experiment.py --config configs/ppo_cartpole.yaml --seeds 0 1 2 3 4

python scripts/run_experiment.py --config configs/ppo_cartpole.yaml --seeds 0 1 2 3 4 \
    --set variant=lr_alto --set hyperparameters.learning_rate=0.01 \
    --set hyperparameters.n_epochs=40 --set hyperparameters.clip_range=0.5

python scripts/aggregate.py --experiment results/runs/ppo_cartpole
```

**Q12.** Em `train/`, encontre `approx_kl`, `clip_fraction`, `entropy_loss`, `explained_variance` e `value_loss`. Para cada uma, diga em uma frase o que ela mede e como se comportou na variante `padrao`.

**Q13.** `entropy_loss` é o **negativo** da entropia. Então, quando a política fica mais determinística, a curva sobe ou desce? Confira nos seus logs.

**Q14.** Compare `padrao` e `lr_alto`. Que métricas denunciam o problema? Alguma delas dá sinal **antes** de `eval/mean_return` piorar? Se não houve problema visível, diga isso e explique o que mais poderia ser alterado.

**Q15.** O PPO foi treinado com observações normalizadas (`VecNormalize`). Rode:

```bash
python scripts/evaluate.py --run results/runs/ppo_cartpole/padrao/seed0
python scripts/evaluate.py --run results/runs/ppo_cartpole/padrao/seed0 --sem-normalizacao
python scripts/evaluate.py --run results/runs/ppo_cartpole/padrao/seed0 --estocastico
```

Explique cada resultado. Por que o arquivo `vecnormalize.pkl` faz parte do modelo tanto quanto `model.zip`?

---

## Parte E: Quantas sementes bastam? (5 min, sem treino novo)

```bash
python scripts/aggregate.py --experiment results/runs/dqn_acrobot --seeds 0 1
python scripts/aggregate.py --experiment results/runs/dqn_acrobot --seeds 3 4
```

**Q16.** Com só duas sementes, as conclusões da Q5 mudam? Os dois pares de sementes concordam entre si? Use a variabilidade que você observou entre sementes para explicar por que reportamos a IQM (média interquartil) com intervalo de confiança por bootstrap, e não "média ± desvio" de 2 ou 3 execuções.

---

## Parte F: Desafio (opcional)

Com **orçamento fixo** de 100 mil passos e 5 sementes, encontre uma configuração do DQN no Acrobot com IQM final maior que a `padrao`, e cujo intervalo de confiança não se sobreponha ao dela. Mude só os hiperparâmetros, sempre com `--set variant=...`. No relatório, descreva o que tentou, inclusive o que **não** funcionou.

---

## Checklist de boas práticas (vale para o relatório inteiro)

- Todo número citado vem de um arquivo em `results/`, com o caminho indicado.
- Toda comparação usa as mesmas sementes nas duas variantes.
- Nenhum resultado foi obtido editando `configs/*.yaml` sem trocar a `variant`.
- O relatório informa as versões (estão em `metadata.json`).
