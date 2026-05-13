"""
Geração de dados sintéticos para experimentos A/B com correlação controlada
entre métrica pré e durante.
"""

import numpy as np
import pandas as pd


def gerar_experimento(n_users=50_000, true_effect=0.02, rho_target=0.6, seed=None):
    """
    Gera um experimento sintético.

    Modelo gerador:
        gamma_i ~ Gamma(shape=2, scale=5)              # nível latente do usuário
        X_i = gamma_i + ruido_pre                       # comportamento pré-experimento
        Y0_i = gamma_i + ruido_durante                  # comportamento baseline durante
        Y_i = Y0_i * (1 + true_effect * T_i)            # efeito multiplicativo

    A magnitude do ruído_durante é calibrada para atingir rho_target.

    Parameters
    ----------
    n_users : int
        Número de usuários no experimento.
    true_effect : float
        Lift relativo aplicado ao grupo de tratamento (ex: 0.02 = +2%).
    rho_target : float
        Correlação alvo entre X (pré) e Y (durante).
    seed : int or None
        Seed do gerador para reprodutibilidade.

    Returns
    -------
    pd.DataFrame com colunas user_id, X, Y, T (treatment indicator).
    """
    rng = np.random.default_rng(seed)

    gamma = rng.gamma(shape=2.0, scale=5.0, size=n_users)

    sigma_pre = 1.0
    X = gamma + rng.normal(0, sigma_pre, n_users)

    var_gamma = gamma.var()
    var_X = X.var()
    sigma_dur = np.sqrt(max(var_gamma**2 / (var_X * rho_target**2) - var_gamma, 0.1))
    Y_baseline = gamma + rng.normal(0, sigma_dur, n_users)

    T = rng.integers(0, 2, size=n_users)
    Y = np.where(T == 1, Y_baseline * (1 + true_effect), Y_baseline)

    return pd.DataFrame({
        "user_id": np.arange(n_users),
        "X": X,
        "Y": Y,
        "T": T,
    })
