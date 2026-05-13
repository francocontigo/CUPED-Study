"""
Métodos de análise para testes A/B com variance reduction.

Implementados:
    - t-test clássico (baseline)
    - CUPED (Deng et al., 2013)
    - CUPED via regressão OLS (equivalente algébrico)
    - CUPAC com cross-fitting (Li et al. DoorDash 2020 + Etsy 2025)
    - MLRATE (Guo et al. Meta, NeurIPS 2021)
"""

import numpy as np
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

from cuped.config import GBM_PARAMS_PADRAO, N_FOLDS_CROSS_FIT


# =============================================================================
# Helper: cross-fitted predictions
# =============================================================================
def _cross_fitted_predictions(X, y, model_params=None, n_folds=N_FOLDS_CROSS_FIT,
                               random_state=42):
    """
    Gera predições out-of-fold para evitar vazamento (gold standard).

    Para cada usuário i, a predição vem de um modelo que NÃO foi treinado com i.
    Isso evita que o modelo capture ruído individual ou (no caso CUPAC) o efeito
    do tratamento que estamos tentando estimar.

    Parameters
    ----------
    X : np.ndarray
        Features pré-experimento.
    y : np.ndarray
        Target (Y do experimento).
    model_params : dict
        Hiperparâmetros do GradientBoostingRegressor.
    n_folds : int
        Número de folds K.
    random_state : int
        Seed do KFold.

    Returns
    -------
    np.ndarray com predições out-of-sample do mesmo shape de y.
    """
    if model_params is None:
        model_params = GBM_PARAMS_PADRAO

    y_hat = np.zeros(len(y))
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    for train_idx, test_idx in kf.split(X):
        model = GradientBoostingRegressor(**model_params)
        model.fit(X[train_idx], y[train_idx])
        y_hat[test_idx] = model.predict(X[test_idx])
    return y_hat


# =============================================================================
# t-test clássico (baseline)
# =============================================================================
def analisar_ttest(df):
    """Welch t-test entre tratamento e controle, sem variance reduction."""
    y_c = df.loc[df["T"] == 0, "Y"].values
    y_t = df.loc[df["T"] == 1, "Y"].values
    diff = y_t.mean() - y_c.mean()
    se = np.sqrt(y_c.var(ddof=1) / len(y_c) + y_t.var(ddof=1) / len(y_t))
    _, p = stats.ttest_ind(y_t, y_c, equal_var=False)
    return {
        "metodo": "t-test",
        "ate": diff,
        "se": se,
        "ci_low": diff - 1.96 * se,
        "ci_high": diff + 1.96 * se,
        "p_value": p,
        "variancia": df["Y"].var(ddof=1),
    }


# =============================================================================
# CUPED clássico
# =============================================================================
def aplicar_cuped(df, cov_col="X"):
    """
    Aplica o ajuste CUPED:

        Y_cuped = Y - theta * (X - mean(X))
        theta   = cov(Y, X) / var(X)

    theta é estimado no pool inteiro (controle + tratamento) para preservar
    a propriedade de unbiased sob randomização.
    """
    df = df.copy()
    cov_xy = np.cov(df["Y"], df[cov_col], ddof=1)[0, 1]
    var_x = df[cov_col].var(ddof=1)
    theta = cov_xy / var_x
    df["Y_cuped"] = df["Y"] - theta * (df[cov_col] - df[cov_col].mean())
    df.attrs["theta"] = theta
    return df


def analisar_cuped(df, cov_col="X"):
    """t-test sobre a métrica ajustada pelo CUPED."""
    df_adj = aplicar_cuped(df, cov_col=cov_col)
    y_c = df_adj.loc[df_adj["T"] == 0, "Y_cuped"].values
    y_t = df_adj.loc[df_adj["T"] == 1, "Y_cuped"].values
    diff = y_t.mean() - y_c.mean()
    se = np.sqrt(y_c.var(ddof=1) / len(y_c) + y_t.var(ddof=1) / len(y_t))
    _, p = stats.ttest_ind(y_t, y_c, equal_var=False)
    return {
        "metodo": "CUPED",
        "ate": diff,
        "se": se,
        "ci_low": diff - 1.96 * se,
        "ci_high": diff + 1.96 * se,
        "p_value": p,
        "variancia": df_adj["Y_cuped"].var(ddof=1),
        "theta": df_adj.attrs["theta"],
    }


# =============================================================================
# CUPED via regressão OLS (equivalente algébrico)
# =============================================================================
def analisar_cuped_regressao(df):
    """
    CUPED via regressão linear:
        Y = beta0 + beta1*T + beta2*X + eps

    O coeficiente beta1 é o ATE ajustado por X. Equivalente algebricamente
    ao CUPED clássico quando X é centrada.
    """
    X_design = df[["T", "X"]].values
    y = df["Y"].values
    model = LinearRegression().fit(X_design, y)
    pred = model.predict(X_design)
    residuals = y - pred
    n, k = X_design.shape

    sigma2 = (residuals ** 2).sum() / (n - k - 1)
    design_with_const = np.column_stack([np.ones(n), X_design])
    XtX_inv = np.linalg.inv(design_with_const.T @ design_with_const)
    se_t = np.sqrt(sigma2 * XtX_inv[1, 1])
    ate = model.coef_[0]
    # stats.t.sf é numericamente estável para p-values pequenos (vs 1 - cdf)
    p = 2 * stats.t.sf(abs(ate / se_t), df=n - k - 1)

    return {
        "metodo": "CUPED-OLS",
        "ate": ate,
        "se": se_t,
        "ci_low": ate - 1.96 * se_t,
        "ci_high": ate + 1.96 * se_t,
        "p_value": p,
    }


# =============================================================================
# CUPAC com cross-fitting (gold standard, DoorDash 2020 → Etsy 2025)
# =============================================================================
def analisar_cupac(df, features_pre, model_params=None, n_folds=N_FOLDS_CROSS_FIT,
                    random_state=42):
    """
    CUPAC com cross-fitting:
        1. Treina um modelo de ML (GBM) sobre todo o dataset, em K folds,
           gerando predições out-of-sample.
        2. Usa essas predições g(X) como a covariável de CUPED.

    Por que cross-fitting?
        - Treinar só no controle (versão simplificada) reduz o sample size de
          treino pela metade e produz predição out-of-distribution no tratamento.
        - Treinar no pool inteiro sem cross-fitting introduz vazamento, o modelo
          aprende ruído individual e parte do efeito do tratamento.
        - Cross-fitting resolve as duas coisas, todo usuário é predito por um
          modelo treinado em outros usuários.

    Esse é o padrão usado em produção no DoorDash (Dash-AB 2022) e Etsy (2025).
    """
    if model_params is None:
        model_params = GBM_PARAMS_PADRAO

    df = df.copy()
    X = df[features_pre].values
    y = df["Y"].values
    df["Y_hat"] = _cross_fitted_predictions(
        X, y, model_params=model_params, n_folds=n_folds, random_state=random_state
    )

    # CUPED-style adjustment usando Y_hat como covariável
    cov_yh = np.cov(df["Y"], df["Y_hat"], ddof=1)[0, 1]
    var_yh = df["Y_hat"].var(ddof=1)
    theta = cov_yh / var_yh
    df["Y_cupac"] = df["Y"] - theta * (df["Y_hat"] - df["Y_hat"].mean())

    y_c = df.loc[df["T"] == 0, "Y_cupac"].values
    y_t = df.loc[df["T"] == 1, "Y_cupac"].values
    diff = y_t.mean() - y_c.mean()
    se = np.sqrt(y_c.var(ddof=1) / len(y_c) + y_t.var(ddof=1) / len(y_t))
    _, p = stats.ttest_ind(y_t, y_c, equal_var=False)

    return {
        "metodo": "CUPAC",
        "ate": diff,
        "se": se,
        "ci_low": diff - 1.96 * se,
        "ci_high": diff + 1.96 * se,
        "p_value": p,
        "variancia": df["Y_cupac"].var(ddof=1),
        "theta": theta,
        "n_folds": n_folds,
    }


# =============================================================================
# MLRATE — Guo et al. (Meta, NeurIPS 2021)
# =============================================================================
def analisar_mlrate(df, features_pre, model_params=None, n_folds=N_FOLDS_CROSS_FIT,
                     random_state=42):
    """
    MLRATE — Machine Learning Regression-Adjusted Treatment Effect Estimator.

    Regressão:
        Y = alpha + tau*T + beta*g(X)_centered + gamma*T*g(X)_centered + eps

    onde g(X) é a predição cross-fitted (mesma do CUPAC) e _centered = subtraindo
    a média. tau é o estimador do ATE.

    Vantagens sobre CUPAC:
        - Termo de interação T*g(X) permite que o efeito do tratamento dependa
          do baseline previsto. Captura HTE (heterogeneous treatment effects).
        - É **consistente mesmo se o modelo de ML for mal especificado**, porque
          T é binário e a randomização garante balanceamento.
        - Inferência via OLS clássico, intervalos de confiança honestos.

    Por que isso importa? CUPAC depende do modelo de ML capturar bem a relação
    X→Y. Se o modelo erra, CUPAC pode até piorar a variância. MLRATE é robusto
    a esse erro.

    Referência: Guo et al. (2021), arXiv:2106.07263.
    """
    if model_params is None:
        model_params = GBM_PARAMS_PADRAO

    X = df[features_pre].values
    y = df["Y"].values
    T = df["T"].values

    g_X = _cross_fitted_predictions(
        X, y, model_params=model_params, n_folds=n_folds, random_state=random_state
    )
    g_X_c = g_X - g_X.mean()

    # design: [T, g(X)_c, T * g(X)_c]
    design = np.column_stack([T, g_X_c, T * g_X_c])
    model = LinearRegression().fit(design, y)

    pred = model.predict(design)
    residuals = y - pred
    n, k = design.shape

    sigma2 = (residuals ** 2).sum() / (n - k - 1)
    design_with_const = np.column_stack([np.ones(n), design])
    XtX_inv = np.linalg.inv(design_with_const.T @ design_with_const)

    tau = model.coef_[0]
    se_tau = np.sqrt(sigma2 * XtX_inv[1, 1])
    p = 2 * stats.t.sf(abs(tau / se_tau), df=n - k - 1)

    return {
        "metodo": "MLRATE",
        "ate": tau,
        "se": se_tau,
        "ci_low": tau - 1.96 * se_tau,
        "ci_high": tau + 1.96 * se_tau,
        "p_value": p,
        "n_folds": n_folds,
    }
