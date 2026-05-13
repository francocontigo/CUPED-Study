"""
Constantes e seeds centralizados para reprodutibilidade.

Mude apenas SEED_BASE para regenerar todos os resultados com sementes diferentes.
"""

# Seed mestre. Tudo deriva dela.
SEED_BASE = 42

# Sub-seeds derivadas para cada experimento.
SEED_EXPERIMENTO_UNICO = SEED_BASE
SEED_POWER_SIM_OFFSET = 10_000      # power_sim usa SEED_POWER_SIM_OFFSET + i por iteração
SEED_AA_TEST_OFFSET = 50_000        # idem pro A/A test
SEED_SENSIBILIDADE = SEED_BASE + 1
SEED_OUTLIERS = SEED_BASE + 2

# Parâmetros padrão do GradientBoosting usado em CUPAC e MLRATE.
GBM_PARAMS_PADRAO = {
    "n_estimators": 100,
    "max_depth": 3,
    "learning_rate": 0.05,
    "random_state": SEED_BASE,
}

# Default de folds para cross-fitting.
N_FOLDS_CROSS_FIT = 5
