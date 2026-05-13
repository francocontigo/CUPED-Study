"""
Roda todos os experimentos do artigo CUPED e imprime os resultados.

Uso:
    python run_experiment.py                  # roda tudo
    python run_experiment.py --parte 5        # só a parte de outliers
    python run_experiment.py --power-sims 1000   # mais simulações
"""

import argparse

from cuped import (
    gerar_experimento,
    analisar_ttest,
    analisar_cuped,
    analisar_cuped_regressao,
    analisar_cupac,
    analisar_mlrate,
    power_simulation,
    sensibilidade_rho,
    demo_outliers,
)
from cuped.config import SEED_EXPERIMENTO_UNICO


def parte_1_experimento_unico():
    """Compara t-test, CUPED, CUPED-OLS, CUPAC e MLRATE num único experimento."""
    print("=" * 75)
    print("PARTE 1 — EXPERIMENTO ÚNICO")
    print("=" * 75)

    df = gerar_experimento(
        n_users=50_000,
        true_effect=0.02,
        rho_target=0.6,
        seed=SEED_EXPERIMENTO_UNICO,
    )
    rho_real = df[["X", "Y"]].corr().iloc[0, 1]
    y_mean_ctrl = df.loc[df["T"] == 0, "Y"].mean()

    print(f"\nN = {len(df):,} | rho(X,Y) real = {rho_real:.3f}")
    print(f"Média Y no controle = {y_mean_ctrl:.3f}")
    print(f"Efeito absoluto esperado = {0.02 * y_mean_ctrl:.4f}\n")

    # CUPAC e MLRATE usam cross-fitting (mais lento, ~3 segundos)
    resultados = [
        analisar_ttest(df),
        analisar_cuped(df),
        analisar_cuped_regressao(df),
        analisar_cupac(df, features_pre=["X"]),
        analisar_mlrate(df, features_pre=["X"]),
    ]

    print(f"{'Método':<12} | {'ATE':>8} | {'SE':>8} | {'CI 95%':>22} | {'p-value':>8}")
    print("-" * 75)
    for r in resultados:
        ci = f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]"
        print(
            f"{r['metodo']:<12} | {r['ate']:+8.4f} | {r['se']:>8.4f} | "
            f"{ci:>22} | {r['p_value']:>8.4f}"
        )

    # CUPED já tem variancia no resultado, não precisa recomputar
    cuped_res = resultados[1]
    var_y = resultados[0]["variancia"]
    var_cuped = cuped_res["variancia"]
    print(f"\nRedução de variância CUPED: {(1 - var_cuped / var_y) * 100:.1f}%")
    print(f"Predição teórica (rho²): {(rho_real ** 2) * 100:.1f}%")


def parte_2_power_simulation(n_sims=400):
    """Power simulation com efeito real fixo em 2%, variando N e rho."""
    print("\n" + "=" * 75)
    print(f"PARTE 2 — POWER SIMULATION ({n_sims} experimentos por cenário)")
    print("=" * 75)
    print("Efeito real = 2%, alpha = 5%\n")

    print(
        f"{'N':>8} | {'rho':>5} | {'pwr t-test':>10} | {'pwr CUPED':>10} | "
        f"{'var redux':>10} | {'std ATE t-test':>14} | {'std ATE CUPED':>14}"
    )
    print("-" * 90)
    for n in [5_000, 10_000, 25_000]:
        for rho in [0.3, 0.6, 0.8]:
            r = power_simulation(
                n_users=n, true_effect=0.02, rho=rho, n_sims=n_sims
            )
            print(
                f"{n:>8,} | {rho:>5.1f} | {r['power_ttest']*100:>9.1f}% | "
                f"{r['power_cuped']*100:>9.1f}% | "
                f"{r['var_reduction_mean']*100:>9.1f}% | "
                f"{r['ate_ttest_std']:>14.4f} | {r['ate_cuped_std']:>14.4f}"
            )


def parte_3_aa_test(n_sims=2_000):
    """A/A test: prova empírica de que CUPED é unbiased."""
    from cuped.config import SEED_AA_TEST_OFFSET

    print("\n" + "=" * 75)
    print(f"PARTE 3 — A/A TEST ({n_sims} simulações, efeito real = 0)")
    print("=" * 75)
    print("Taxa de falso positivo esperada: ~5% (alpha)\n")

    aa = power_simulation(
        n_users=20_000, true_effect=0.0, rho=0.6,
        n_sims=n_sims, seed_offset=SEED_AA_TEST_OFFSET,
    )
    print(f"  Falso positivo t-test: {aa['power_ttest']*100:.2f}%")
    print(f"  Falso positivo CUPED:  {aa['power_cuped']*100:.2f}%")
    print(f"  ATE médio t-test:  {aa['ate_ttest_mean']:+.5f}  (esperado: ~0)")
    print(f"  ATE médio CUPED:   {aa['ate_cuped_mean']:+.5f}  (esperado: ~0)")


def parte_4_sensibilidade():
    """Mostra a relação empírica entre rho e redução de variância."""
    print("\n" + "=" * 75)
    print("PARTE 4 — SENSIBILIDADE AO rho (correlação pré × durante)")
    print("=" * 75)
    print(f"\n  {'rho real':>10} | {'redução var':>12} | {'sample size equivalente':>25}")
    print("-" * 60)

    for r in sensibilidade_rho():
        print(
            f"  {r['rho_real']:>10.3f} | {r['variance_reduction']*100:>11.1f}% | "
            f"{r['sample_size_equivalente_pct']:>22.1f}% do N original"
        )


def parte_5_outliers():
    """Demonstra a quebra do CUPED com outliers e o resgate via winsorização."""
    print("\n" + "=" * 75)
    print("PARTE 5 — ROBUSTEZ A OUTLIERS")
    print("=" * 75)

    r = demo_outliers()
    cfg = r["config"]

    print(f"\n  N = {cfg['n_users']:,} | true_effect = {cfg['true_effect']*100:.0f}%")
    print(f"  {cfg['n_outliers']} outliers injetados em Y ({cfg['outlier_magnitude']}x "
          f"a mediana, X intacto)")
    print(f"  Efeito absoluto esperado: {cfg['efeito_absoluto_esperado']:+.4f}\n")

    print(f"  {'Cenário':<22} | {'Método':<8} | {'ATE':>8} | {'p-value':>8} | {'Detectou?':>10}")
    print("  " + "-" * 67)

    for cenario_label, cenario_key in [
        ("SEM winsorização", "sem_winsorizacao"),
        ("COM winsorização P99", "com_winsorizacao"),
    ]:
        for metodo in ["ttest", "cuped"]:
            res = r[cenario_key][metodo]
            detectou = "✓ sim" if res["p_value"] < 0.05 else "✗ não"
            print(
                f"  {cenario_label:<22} | {res['metodo']:<8} | "
                f"{res['ate']:+8.4f} | {res['p_value']:>8.4f} | {detectou:>10}"
            )

    print("\n  Leitura: outliers em Y não-correlacionados com X destroem o sinal.")
    print("  CUPED não resolve porque X (pré-experimento) não consegue prever o pico.")
    print("  Winsorização no P99 ANTES de aplicar CUPED restaura o poder do teste.")


def parte_6_movielens(ratings_path, true_effect=0.10, n_sims=200):
    """Aplica CUPED, CUPAC e MLRATE no MovieLens 25M com tratamento sintético."""
    import os
    if not os.path.exists(ratings_path):
        print("\n" + "=" * 75)
        print("PARTE 6 — MOVIELENS 25M (PULADA)")
        print("=" * 75)
        print(f"\n  ratings.csv não encontrado em {ratings_path}")
        print("  Baixe de https://files.grouplens.org/datasets/movielens/ml-25m.zip,")
        print("  descompacte, e passe --movielens-path <caminho>/ratings.csv\n")
        return

    from cuped import (
        load_movielens_panel,
        aplicar_tratamento_sintetico,
        power_simulation_movielens,
        analisar_ttest, analisar_cuped, analisar_cupac, analisar_mlrate,
    )

    print("\n" + "=" * 75)
    print("PARTE 6 — MOVIELENS 25M (dados reais + tratamento sintético)")
    print("=" * 75)

    panel = load_movielens_panel(
        ratings_path,
        cutoff="2018-01-01",
        pre_days=180,
        during_days=60,
        min_pre_ratings=1,
        apply_log=True,
    )

    print(f"  N = {len(panel):,} usuários ativos | pré: 180 dias | durante: 60 dias")
    print(f"  Cutoff: 2018-01-01 | métrica: ratings/usuário em log1p\n")

    print("  Correlações features pré × Y:")
    for col in ["X_count", "X_mean_rating", "X_last_30", "X_last_90", "X_recency"]:
        rho = panel[[col, "Y"]].corr().iloc[0, 1]
        print(f"    {col:<18}: rho = {rho:+.3f}")

    # --- Sub-parte A: experimento único com efeito de 10% ---
    print(f"\n  ── Experimento único, tratamento sintético +{true_effect*100:.0f}% ──")

    panel_t = aplicar_tratamento_sintetico(panel, true_effect=true_effect, seed=42)
    panel_simple = panel_t.rename(columns={"X_count": "X"})
    features = ["X_count", "X_mean_rating", "X_last_30",
                "X_last_90", "X_recency"]

    resultados = [
        analisar_ttest(panel_simple),
        analisar_cuped(panel_simple),
        analisar_cupac(panel_t, features_pre=features),
        analisar_mlrate(panel_t, features_pre=features),
    ]

    print(f"\n  {'Método':<10} | {'ATE':>8} | {'SE':>8} | {'p-value':>8} | {'Detectou?':>10}")
    print("  " + "-" * 60)
    for r in resultados:
        detectou = "✓ sim" if r["p_value"] < 0.05 else "✗ não"
        print(
            f"  {r['metodo']:<10} | {r['ate']:+8.4f} | {r['se']:>8.4f} | "
            f"{r['p_value']:>8.4f} | {detectou:>10}"
        )

    # --- Sub-parte B: power simulation re-randomizando T ---
    print(f"\n  ── Power simulation: {n_sims} randomizações de T (mesmo painel) ──")

    res = power_simulation_movielens(
        panel, true_effect=true_effect, n_sims=n_sims
    )
    print(f"  Effect verdadeiro: {true_effect*100:.0f}% | N: {res['n_users']:,}\n")
    print(f"  {'Método':<10} | {'Power':>8}")
    print("  " + "-" * 23)
    print(f"  {'t-test':<10} | {res['power_ttest']*100:>7.1f}%")
    print(f"  {'CUPED':<10} | {res['power_cuped']*100:>7.1f}%")
    print(f"  {'CUPAC':<10} | {res['power_cupac']*100:>7.1f}%")
    print(f"  {'MLRATE':<10} | {res['power_mlrate']*100:>7.1f}%")
    print(f"\n  Redução de variância CUPED (média): "
          f"{res['var_reduction_cuped']*100:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="CUPED demo runner")
    parser.add_argument(
        "--parte",
        type=int,
        choices=[0, 1, 2, 3, 4, 5, 6],
        default=0,
        help="0 = todas (exceto 6 se não houver dataset); 1-6 = parte específica",
    )
    parser.add_argument(
        "--power-sims",
        type=int,
        default=400,
        help="Simulações por cenário no power test (default: 400)",
    )
    parser.add_argument(
        "--aa-sims",
        type=int,
        default=2_000,
        help="Simulações no A/A test (default: 2000)",
    )
    parser.add_argument(
        "--movielens-path",
        type=str,
        default="data/ml-25m/ratings.csv",
        help="Caminho para o ratings.csv do MovieLens 25M",
    )
    parser.add_argument(
        "--ml-sims",
        type=int,
        default=100,
        help="Simulações no power test do MovieLens (default: 100, ~3-4 min)",
    )
    parser.add_argument(
        "--ml-effect",
        type=float,
        default=0.10,
        help="Tamanho do efeito sintético no MovieLens (default: 0.10 = 10%%)",
    )
    args = parser.parse_args()

    if args.parte in (0, 1):
        parte_1_experimento_unico()
    if args.parte in (0, 2):
        parte_2_power_simulation(n_sims=args.power_sims)
    if args.parte in (0, 3):
        parte_3_aa_test(n_sims=args.aa_sims)
    if args.parte in (0, 4):
        parte_4_sensibilidade()
    if args.parte in (0, 5):
        parte_5_outliers()
    if args.parte in (0, 6):
        parte_6_movielens(
            ratings_path=args.movielens_path,
            true_effect=args.ml_effect,
            n_sims=args.ml_sims,
        )


if __name__ == "__main__":
    main()
