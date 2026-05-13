"""
Análises agregadas:
    - power_simulation: roda N experimentos e mede taxa de detecção
    - sensibilidade_rho: redução de variância em função da correlação pré×durante
    - demo_outliers: mostra quebra do CUPED com outliers e o resgate via winsorização
"""

import numpy as np

from cuped.config import (
    SEED_POWER_SIM_OFFSET,
    SEED_SENSIBILIDADE,
    SEED_OUTLIERS,
)
from cuped.data import gerar_experimento
from cuped.methods import analisar_ttest, analisar_cuped, aplicar_cuped


def power_simulation(n_users, true_effect, rho, n_sims=400,
                      seed_offset=SEED_POWER_SIM_OFFSET):
    """
    Roda n_sims experimentos e mede taxa de detecção (power) para t-test e CUPED.

    Cada simulação usa seed = seed_offset + i, então rodadas são reprodutíveis
    e independentes entre si.
    """
    det_t, det_c = 0, 0
    vars_red, ates_t, ates_c = [], [], []

    for i in range(n_sims):
        d = gerar_experimento(
            n_users=n_users,
            true_effect=true_effect,
            rho_target=rho,
            seed=seed_offset + i,
        )
        r_t = analisar_ttest(d)
        r_c = analisar_cuped(d)

        det_t += int(r_t["p_value"] < 0.05)
        det_c += int(r_c["p_value"] < 0.05)
        vars_red.append(1 - r_c["variancia"] / r_t["variancia"])
        ates_t.append(r_t["ate"])
        ates_c.append(r_c["ate"])

    return {
        "n_users": n_users,
        "true_effect": true_effect,
        "rho_target": rho,
        "n_sims": n_sims,
        "power_ttest": det_t / n_sims,
        "power_cuped": det_c / n_sims,
        "var_reduction_mean": float(np.mean(vars_red)),
        "ate_ttest_mean": float(np.mean(ates_t)),
        "ate_ttest_std": float(np.std(ates_t)),
        "ate_cuped_mean": float(np.mean(ates_c)),
        "ate_cuped_std": float(np.std(ates_c)),
    }


def sensibilidade_rho(rhos=None, n_users=50_000, true_effect=0.02,
                       seed=SEED_SENSIBILIDADE):
    """
    Mede redução de variância em função do rho-alvo.

    A teoria diz que redução = rho². Esta função verifica empiricamente.
    """
    if rhos is None:
        rhos = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    resultados = []
    for rho in rhos:
        d = gerar_experimento(
            n_users=n_users,
            true_effect=true_effect,
            rho_target=rho,
            seed=seed,
        )
        rho_real = float(d[["X", "Y"]].corr().iloc[0, 1])
        var_y = d["Y"].var(ddof=1)
        var_cuped = aplicar_cuped(d)["Y_cuped"].var(ddof=1)
        redux = float(1 - var_cuped / var_y)
        resultados.append({
            "rho_target": rho,
            "rho_real": rho_real,
            "variance_reduction": redux,
            "sample_size_equivalente_pct": float((1 - redux) * 100),
        })
    return resultados


def demo_outliers(n_users=20_000, true_effect=0.05, rho=0.6,
                   n_outliers=15, outlier_magnitude=150,
                   seed=SEED_OUTLIERS):
    """
    Demonstra a fragilidade do CUPED a outliers em Y não-correlacionados com X.

    Cenário: simulamos usuários que tiveram um pico inesperado em Y (compra
    grande de presente, fraude, abuso, evento isolado), mas que tinham
    comportamento pré-experimento normal (X dentro da distribuição). É o caso
    realista, no qual o pré-experimento NÃO consegue prever o outlier.

    Por que isso quebra o CUPED?
    - theta = cov(Y, X) / var(X), e o outlier puxa cov(Y, X) sem aumentar
      proporcionalmente var(X).
    - Mais grave, mesmo quando theta fica próximo do correto, a variância
      residual de Y_cuped permanece inflada porque o ajuste linear não
      consegue capturar o pico.
    - Resultado, o intervalo de confiança fica largo e o p-value sobe.

    Winsorização (clip no percentil 99) resolve. É o que Amazon, Microsoft e
    qualquer plataforma séria faz como pré-processamento antes de qualquer
    análise de A/B, com ou sem CUPED.
    """
    df = gerar_experimento(
        n_users=n_users, true_effect=true_effect, rho_target=rho, seed=seed
    )

    # Outliers em Y APENAS (X permanece normal). Distribuídos aleatoriamente
    # entre tratamento e controle, com fração ligeiramente maior no controle
    # para simular o "azar amostral" que costuma ferrar com testes.
    rng = np.random.default_rng(seed + 1)
    ctrl_idx = df[df["T"] == 0].index
    trat_idx = df[df["T"] == 1].index

    # 60% dos outliers no controle, 40% no tratamento (assimetria de azar)
    n_ctrl = int(n_outliers * 0.6)
    n_trat = n_outliers - n_ctrl
    out_ctrl = rng.choice(ctrl_idx, size=n_ctrl, replace=False)
    out_trat = rng.choice(trat_idx, size=n_trat, replace=False)
    outlier_idx = np.concatenate([out_ctrl, out_trat])

    y_median = float(df["Y"].median())
    df.loc[outlier_idx, "Y"] = y_median * outlier_magnitude
    # X permanece intacto, os usuários parecem normais no pré-experimento

    # Versão crua, sem tratamento de outliers
    r_ttest_raw = analisar_ttest(df)
    r_cuped_raw = analisar_cuped(df)

    # Versão com winsorização no P99
    df_w = df.copy()
    for col in ["X", "Y"]:
        p99 = df_w[col].quantile(0.99)
        df_w[col] = df_w[col].clip(upper=p99)

    r_ttest_win = analisar_ttest(df_w)
    r_cuped_win = analisar_cuped(df_w)

    return {
        "config": {
            "n_users": n_users,
            "true_effect": true_effect,
            "rho_target": rho,
            "n_outliers": n_outliers,
            "outlier_magnitude": outlier_magnitude,
            "efeito_absoluto_esperado": true_effect * df.loc[df["T"] == 0, "Y"].mean(),
        },
        "sem_winsorizacao": {"ttest": r_ttest_raw, "cuped": r_cuped_raw},
        "com_winsorizacao": {"ttest": r_ttest_win, "cuped": r_cuped_win},
    }
