# CUPED-Study

Implementação didática de **CUPED** (Controlled-experiment Using Pre-Experiment Data) e variantes modernas em Python, com gold-standard cross-fitting, MLRATE e demonstração de robustez a outliers.

Companion code do artigo **"CUPED, ou como o Bing cortou o tempo de teste A/B pela metade usando dados que já tinham"**.

Dois runners independentes:

| Script | O que faz | Tempo |
|---|---|---|
| `run_experiment.py` | 5 partes em dataset sintético (experimento único, power simulation, A/A test, sensibilidade ao rho, outliers) | ~70s |
| `run_movielens.py` | Experimento semi-sintético em **MovieLens 25M** real, 3 cenários por nível de engajamento | ~3-5min |

## Quick start (sintético)

```bash
git clone https://github.com/francocontigo/CUPED-Study.git
cd CUPED-Study
pip install -r requirements.txt
python run_experiment.py
```

5 partes em ~70 segundos:

1. **Experimento único** (50k usuários, rho=0.6, efeito=2%) — compara os 5 métodos lado a lado
2. **Power simulation** (400 experimentos × 9 cenários) — mede o ganho real de power
3. **A/A test** (2.000 simulações com efeito=0) — verifica que CUPED é unbiased
4. **Sensibilidade ao rho** — confirma empiricamente que `redução = rho²`
5. **Robustez a outliers** — mostra como outliers em Y quebram CUPED e como winsorização salva

Para rodar só uma parte:
```bash
python run_movielens.py --true-effect 0.10
```

## Quick start (MovieLens 25M)

Baixe o dataset de https://files.grouplens.org/datasets/movielens/ml-25m.zip e descompacte na raiz do repo, gerando a pasta `ml-25m/`. Depois:

```bash
python run_movielens.py
```

O script roda um experimento semi-sintético, **dados reais de comportamento de 162k usuários**, atribuição aleatória 50/50, e efeito sintético multiplicativo de +5% sobre `Y` no grupo de tratamento. Compara os 5 métodos em 3 cenários (todos os usuários elegíveis, ativos com X≥5, heavy com X≥20).

CLI:
```bash
python run_movielens.py --true-effect 0.10      # efeito de +10%
python run_movielens.py --cutoff 2019-06-01     # outro ponto temporal
python run_movielens.py --pre-days 180          # janela pré maior
```

## Métodos implementados

| Método | Arquivo | Referência |
|---|---|---|
| t-test (Welch) | `cuped/methods.py:analisar_ttest` | baseline |
| CUPED clássico | `cuped/methods.py:analisar_cuped` | [Deng et al., WSDM 2013](https://www.exp-platform.com/Documents/2013-02-CUPED-ImprovingSensitivityOfControlledExperiments.pdf) |
| CUPED via OLS | `cuped/methods.py:analisar_cuped_regressao` | equivalente algébrico |
| CUPAC (cross-fitted) | `cuped/methods.py:analisar_cupac` | [Li, Tang, Bauman — DoorDash 2020](https://careersatdoordash.com/blog/improving-experimental-power-through-control-using-predictions-as-covariate-cupac/) |
| MLRATE | `cuped/methods.py:analisar_mlrate` | [Guo et al. — Meta, NeurIPS 2021](https://arxiv.org/abs/2106.07263) |

CUPAC e MLRATE implementam **cross-fitting com K=5 folds** (gold standard).

## Estrutura

```
CUPED-Study/
├── README.md
├── requirements.txt
├── run_experiment.py        # runner sintético, 5 partes
├── run_movielens.py         # runner MovieLens 25M, 3 cenários
├── cuped/
│   ├── __init__.py
│   ├── config.py            # seeds e hiperparâmetros centralizados
│   ├── data.py              # geração sintética
│   ├── methods.py           # t-test, CUPED, CUPED-OLS, CUPAC, MLRATE
│   └── analysis.py          # power simulation, sensibilidade, outliers
├── ml-25m/                  # baixe MovieLens 25M aqui (não versionado)
│   └── ratings.csv
└── results/
    └── output.txt           # exemplo de output do runner sintético
```

## Usar como biblioteca

```python
from cuped import (
    gerar_experimento,
    analisar_cuped, analisar_cupac, analisar_mlrate,
)

df = gerar_experimento(n_users=20_000, true_effect=0.03, rho_target=0.7, seed=42)
print(analisar_cuped(df))
print(analisar_cupac(df, features_pre=["X"], n_folds=5))
print(analisar_mlrate(df, features_pre=["X"], n_folds=5))
```

Em dados reais, basta um DataFrame com colunas `Y`, `X` (ou várias features pré), e `T` (0/1):

```python
import pandas as pd
from cuped import analisar_cuped, analisar_cupac

df = pd.read_csv("seu_experimento.csv").rename(columns={
    "spend_2weeks": "Y",
    "history_12m": "X",
    "treatment_group": "T",
})
print(analisar_cuped(df))
print(analisar_cupac(df, features_pre=["X", "recency", "frequency"]))
```

## Notas de implementação

- **theta estimado no pool inteiro** (controle + tratamento). Estimar por grupo introduz viés porque o theta passa a capturar parte do efeito do tratamento.
- **CUPAC e MLRATE usam cross-fitting** com K=5 folds. Para cada usuário, a predição vem de um modelo treinado em outros usuários (out-of-sample). Padrão moderno do mercado.
- **MLRATE inclui termo de interação T×g(X)**, que o torna consistente mesmo quando o modelo de ML é mal especificado. CUPAC pode até piorar variância se o modelo erra muito; MLRATE não.
- **Reprodutibilidade**: todas as seeds estão em `cuped/config.py`. Mude `SEED_BASE` lá pra regenerar tudo com sementes diferentes.

## Referências

- Deng, A., Xu, Y., Kohavi, R., Walker, T. (2013). "Improving the Sensitivity of Online Controlled Experiments by Utilizing Pre-Experiment Data". *WSDM 2013*.
- Xie, H., Aurisset, J. (2016). "Improving the Sensitivity of Online Controlled Experiments: Case Studies at Netflix". *KDD 2016*.
- Li, J., Tang, Y., Bauman, J. (2020). "Improving Experimental Power through Control Using Predictions as Covariate (CUPAC)". DoorDash Engineering Blog.
- Guo, Y. et al. (2021). "Machine Learning for Variance Reduction in Online Experiments (MLRATE)". *NeurIPS 2021*. [arXiv:2106.07263](https://arxiv.org/abs/2106.07263)
- Deng, A. et al. (2023). "From Augmentation to Decomposition: A New Look at CUPED in 2023". [arXiv:2312.02935](https://arxiv.org/abs/2312.02935)
- MovieLens 25M Dataset, GroupLens Research, https://grouplens.org/datasets/movielens/25m/

## Licença

MIT.
