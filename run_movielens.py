"""
run_movielens.py
================
Apply CUPED methods to a semi-synthetic A/B test built from MovieLens 25M.

Design
------
Real user behavior (rating activity over time) + synthetic random assignment
with a controlled treatment effect. Standard pedagogical setup when a real
A/B test dataset with pre-experiment data is not available.

    - Pre-experiment window:  [cutoff - pre_days, cutoff)
    - Experiment window:      [cutoff, cutoff + exp_days)
    - Y_baseline = rating count per user during the experiment window
    - Y          = Y_baseline * (1 + true_effect) for users in treatment group
    - X          = rating count per user during the pre-experiment window
    - Additional features for CUPAC / MLRATE: mean rating, std rating,
      number of distinct active days.

Random 50/50 assignment, multiplicative treatment effect on Y.

Usage
-----
    python run_movielens.py
    python run_movielens.py --true-effect 0.10
    python run_movielens.py --cutoff 2019-06-01 --pre-days 180

Expects MovieLens 25M unpacked at ./ml-25m/ratings.csv (default path).
Use --ratings-path to point elsewhere.

Runtime: ~3-5 minutes total on a laptop with 8GB RAM (depends mostly on the
size of the panel after filtering).
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from cuped import (
    analisar_ttest,
    analisar_cuped,
    analisar_cuped_regressao,
    analisar_cupac,
    analisar_mlrate,
)


DEFAULT_RATINGS_PATH = Path("ml-25m/ratings.csv")
FEATURE_COLUMNS = ["X_count", "X_mean_rating", "X_std_rating", "X_active_days"]


# =============================================================================
# Data loading
# =============================================================================
def load_ratings(path):
    """Load ratings.csv with memory-efficient dtypes (~600 MB peak in RAM)."""
    print(f"[load] {path}")
    t0 = time.time()
    df = pd.read_csv(
        path,
        usecols=["userId", "rating", "timestamp"],
        dtype={"userId": np.int32, "rating": np.float32, "timestamp": np.int64},
    )
    df["date"] = pd.to_datetime(df["timestamp"], unit="s")
    df = df.drop(columns=["timestamp"])
    print(f"[ok] {len(df):,} ratings loaded in {time.time() - t0:.1f}s")
    print(
        f"[info] date range: {df['date'].min().date()} "
        f"-> {df['date'].max().date()}"
    )
    return df


# =============================================================================
# Feature engineering
# =============================================================================
def build_features(ratings, window_start, window_end):
    """
    Compute per-user features over [window_start, window_end).
    Returns a DataFrame indexed by userId.
    """
    mask = (ratings["date"] >= window_start) & (ratings["date"] < window_end)
    pre = ratings.loc[mask].copy()
    pre["day"] = pre["date"].dt.normalize()

    feats = pre.groupby("userId").agg(
        X_count=("rating", "size"),
        X_mean_rating=("rating", "mean"),
        X_std_rating=("rating", "std"),
        X_active_days=("day", "nunique"),
    )
    # std is NaN for users with a single rating; impute zero
    feats["X_std_rating"] = feats["X_std_rating"].fillna(0)
    feats["X"] = feats["X_count"]  # primary covariate alias used by CUPED
    return feats


def build_experiment(ratings, cutoff, pre_days, exp_days,
                      true_effect, min_X, seed):
    """Construct a semi-synthetic A/B test panel."""
    pre_start = cutoff - pd.Timedelta(days=pre_days)
    exp_end = cutoff + pd.Timedelta(days=exp_days)

    print(f"\n[windows]  pre: {pre_start.date()} -> {cutoff.date()} ({pre_days}d)")
    print(f"[windows]  exp: {cutoff.date()} -> {exp_end.date()} ({exp_days}d)")
    print(f"[setup]    true effect: +{true_effect * 100:.1f}%")
    print(f"[setup]    min pre-activity (X): {min_X}")

    # Pre-window features
    feats = build_features(ratings, pre_start, cutoff)
    feats = feats[feats["X"] >= min_X]

    # During-window Y_baseline (count of ratings)
    mask = (ratings["date"] >= cutoff) & (ratings["date"] < exp_end)
    Y = ratings.loc[mask].groupby("userId").size().rename("Y_baseline")
    panel = feats.join(Y, how="left").fillna({"Y_baseline": 0})
    panel["Y_baseline"] = panel["Y_baseline"].astype(float)

    # Random 50/50 assignment + synthetic multiplicative effect
    rng = np.random.default_rng(seed)
    panel["T"] = rng.integers(0, 2, size=len(panel))
    panel["Y"] = np.where(
        panel["T"] == 1,
        panel["Y_baseline"] * (1 + true_effect),
        panel["Y_baseline"],
    )

    print(f"[panel]    N = {len(panel):,} eligible users")
    return panel.reset_index()


# =============================================================================
# Analysis
# =============================================================================
def print_results_table(results):
    """Pretty-print results table for a single scenario."""
    print(
        f"\n  {'Method':<12} | {'ATE':>9} | {'SE':>8} | "
        f"{'CI 95%':>24} | {'p-value':>8}"
    )
    print("  " + "-" * 76)
    for r in results:
        ci = f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]"
        print(
            f"  {r['metodo']:<12} | {r['ate']:+9.4f} | {r['se']:>8.4f} | "
            f"{ci:>24} | {r['p_value']:>8.4f}"
        )


def run_scenario(panel, features, label, true_effect):
    """Run all 5 methods on a panel subset and print results."""
    print(f"\n{'=' * 78}")
    print(f"SCENARIO: {label}")
    print(f"{'=' * 78}")

    if len(panel) == 0:
        print("  [skip] no users in this scenario")
        return None

    rho = float(panel[["X", "Y_baseline"]].corr().iloc[0, 1])
    mean_ctrl = float(panel.loc[panel["T"] == 0, "Y_baseline"].mean())
    expected_ate = true_effect * mean_ctrl

    print(f"  N                                   = {len(panel):,}")
    print(f"  rho(X, Y_baseline)                  = {rho:.4f}")
    print(f"  mean Y_baseline (control group)     = {mean_ctrl:.4f}")
    print(f"  expected ATE (true_effect * mean)   = {expected_ate:+.4f}")

    t0 = time.time()
    results = [
        analisar_ttest(panel),
        analisar_cuped(panel),
        analisar_cuped_regressao(panel),
        analisar_cupac(panel, features_pre=features),
        analisar_mlrate(panel, features_pre=features),
    ]
    print(f"  analysis time                       = {time.time() - t0:.1f}s")

    print_results_table(results)

    var_y = results[0]["variancia"]
    var_cuped = results[1]["variancia"]
    var_redux = 1 - var_cuped / var_y
    print(f"\n  Variance reduction (CUPED, empirical): {var_redux * 100:6.2f}%")
    print(f"  Theoretical bound (rho^2):             {(rho ** 2) * 100:6.2f}%")

    return {
        "scenario": label,
        "N": len(panel),
        "rho": rho,
        "expected_ate": expected_ate,
        "results": results,
        "var_reduction": var_redux,
    }


def print_summary(scenarios):
    """Final comparison table across scenarios."""
    print(f"\n{'=' * 78}")
    print("SUMMARY")
    print(f"{'=' * 78}")
    print(
        f"{'Scenario':<22} | {'N':>8} | {'rho':>6} | {'var redux':>10} | "
        f"{'t-test p':>9} | {'CUPED p':>9} | {'MLRATE p':>9}"
    )
    print("-" * 95)
    for s in scenarios:
        if s is None:
            continue
        rs = s["results"]
        p_t = rs[0]["p_value"]
        p_c = rs[1]["p_value"]
        p_m = rs[4]["p_value"]
        print(
            f"{s['scenario']:<22} | {s['N']:>8,} | {s['rho']:>6.3f} | "
            f"{s['var_reduction'] * 100:>9.2f}% | {p_t:>9.4f} | "
            f"{p_c:>9.4f} | {p_m:>9.4f}"
        )


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="MovieLens 25M CUPED experiment")
    parser.add_argument(
        "--ratings-path", default=str(DEFAULT_RATINGS_PATH),
        help="Path to ratings.csv (default: ml-25m/ratings.csv)",
    )
    parser.add_argument(
        "--cutoff", default="2019-08-01",
        help="Experiment cutoff date YYYY-MM-DD (default: 2019-08-01). "
             "Must allow pre_days before and exp_days after within the data range.",
    )
    parser.add_argument("--pre-days", type=int, default=90,
                        help="Pre-experiment window in days (default: 90)")
    parser.add_argument("--exp-days", type=int, default=30,
                        help="Experiment window in days (default: 30)")
    parser.add_argument(
        "--true-effect", type=float, default=0.05,
        help="Synthetic multiplicative treatment effect (default: 0.05 = +5%)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    total_t0 = time.time()

    ratings = load_ratings(Path(args.ratings_path))

    print(f"\n{'=' * 78}")
    print("MOVIELENS 25M -- SEMI-SYNTHETIC A/B TEST")
    print(f"{'=' * 78}")

    cutoff = pd.to_datetime(args.cutoff)
    panel = build_experiment(
        ratings, cutoff,
        pre_days=args.pre_days,
        exp_days=args.exp_days,
        true_effect=args.true_effect,
        min_X=1,
        seed=args.seed,
    )

    # Three scenarios with progressively more engaged users
    scenarios = [
        run_scenario(panel, FEATURE_COLUMNS,
                     "ALL eligible (X >= 1)", args.true_effect),
        run_scenario(panel[panel["X"] >= 5].copy(), FEATURE_COLUMNS,
                     "ACTIVE users (X >= 5)", args.true_effect),
        run_scenario(panel[panel["X"] >= 20].copy(), FEATURE_COLUMNS,
                     "HEAVY users (X >= 20)", args.true_effect),
    ]

    print_summary(scenarios)

    print(f"\n[done] total wall time: {time.time() - total_t0:.1f}s")


if __name__ == "__main__":
    main()
