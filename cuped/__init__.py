"""
CUPED — Controlled-experiment Using Pre-Experiment Data.

Implementação didática de variance reduction em testes A/B, com
métodos de gold-standard (cross-fitting, MLRATE) e demonstração
de robustez a outliers.
"""

from cuped.data import gerar_experimento
from cuped.methods import (
    analisar_ttest,
    aplicar_cuped,
    analisar_cuped,
    analisar_cuped_regressao,
    analisar_cupac,
    analisar_mlrate,
)
from cuped.analysis import (
    power_simulation,
    sensibilidade_rho,
    demo_outliers,
)

__all__ = [
    "gerar_experimento",
    "analisar_ttest",
    "aplicar_cuped",
    "analisar_cuped",
    "analisar_cuped_regressao",
    "analisar_cupac",
    "analisar_mlrate",
    "power_simulation",
    "sensibilidade_rho",
    "demo_outliers",
]
