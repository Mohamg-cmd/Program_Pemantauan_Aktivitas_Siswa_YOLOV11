"""
Aligned Multi-Criteria Decision-Making (MCDM) Benchmark Module (Targeted Fix v2.1)
Primary Workspace: MyDrive/YOLO11_AHP_SAW_IJITCS_REVALIDATION_2026

Implements:
1. Primary Decision Support: AHP-SAW (Max/Min linear normalization with epsilon=1e-6 zero handling)
2. Reviewer Priority 3 Baseline A: AHP-TOPSIS (Vector normalization + Euclidean closeness to PIS/NIS)
3. Reviewer Priority 3 Baseline B: AHP-ELECTRE I (Concordance C_ab, Discordance D_ab, Outranking graph, and Net Outranking Flow)
4. Multiplicative Baseline: AHP-WP (Weighted Product)
5. Unweighted Baseline: Equal-Weight SAW (w_j = 0.25)
6. Statistical Concordance & Holm-Bonferroni Suite (`compute_mcdm_concordance_holm_summary`):
   Computes Spearman rho, Fisher-z 95% CIs, Kendall tau, Mean/Max rank shifts, Top-1/Top-5 overlap,
   and Holm-Bonferroni step-down adjusted p-values across all baseline comparisons.
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, List
import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("YOLO11_IJITCS_MCDM")

CRITERIA_COLS = ["Angkat Tangan", "Berdiri", "Diam", "Tidur"]
IS_BENEFIT = [True, True, False, False]


def compute_ahp_eigenvector_weights(pairwise_matrix: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float]:
    if pairwise_matrix is None:
        pairwise_matrix = np.array([
            [1.0, 2.0, 3.0, 3.0],
            [0.5, 1.0, 2.0, 3.0],
            [1.0 / 3.0, 0.5, 1.0, 2.0],
            [1.0 / 3.0, 1.0 / 3.0, 0.5, 1.0],
        ], dtype=np.float64)
    eigvals, eigvecs = np.linalg.eig(pairwise_matrix.astype(np.float64))
    max_idx = int(np.argmax(eigvals.real))
    lambda_max = float(eigvals[max_idx].real)
    w = np.abs(eigvecs[:, max_idx].real)
    w = w / np.sum(w)
    n = pairwise_matrix.shape[0]
    ci = (lambda_max - n) / (n - 1)
    ri_table = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12}
    cr = ci / ri_table.get(n, 0.90)
    return w, cr


def holm_bonferroni_adjust(p_values: List[float]) -> List[float]:
    """Step-down Holm-Bonferroni multiple-testing correction enforcing monotonicity."""
    m = len(p_values)
    if m == 0:
        return []
    p_arr = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(p_arr, kind="mergesort")
    adjusted = np.zeros(m, dtype=np.float64)
    running_max = 0.0
    for rank_idx, orig_idx in enumerate(order):
        adj = min(1.0, float(p_arr[orig_idx]) * float(m - rank_idx))
        running_max = max(running_max, adj)
        adjusted[orig_idx] = running_max
    return adjusted.tolist()


def fisher_z_ci(r: float, n: int = 21) -> Tuple[float, float]:
    """95% confidence interval for correlation coefficient r via Fisher z-transformation."""
    r_clip = min(max(float(r), -0.999999), 0.999999)
    z = np.arctanh(r_clip)
    se = 1.0 / np.sqrt(max(n - 3, 1))
    return float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))


def evaluate_all_mcdm_methods(df_duration: pd.DataFrame, eps: float = 1e-6) -> pd.DataFrame:
    w_ahp, cr = compute_ahp_eigenvector_weights()
    logger.info("Verified AHP weights=%s, CR=%.6f", np.round(w_ahp, 6), cr)

    id_col = "entity" if "entity" in df_duration.columns else "student_id"
    X = df_duration[CRITERIA_COLS].to_numpy(dtype=np.float64)
    X_eps = np.where(X <= 0.0, eps, X)
    n, m = X.shape

    # 1. AHP-SAW & Equal-SAW
    R_saw = np.zeros_like(X_eps)
    for j in range(m):
        R_saw[:, j] = (X[:, j] / np.max(X_eps[:, j])) if IS_BENEFIT[j] else (np.min(X_eps[:, j]) / X_eps[:, j])
    saw_scores = R_saw @ w_ahp
    saw_ranks = stats.rankdata(-saw_scores, method="min").astype(int)
    eq_scores = R_saw @ np.full(m, 0.25)
    eq_ranks = stats.rankdata(-eq_scores, method="min").astype(int)

    # 2. AHP-TOPSIS
    V_top = (X_eps / np.sqrt(np.sum(X_eps ** 2, axis=0))) * w_ahp
    pis = np.array([np.max(V_top[:, j]) if IS_BENEFIT[j] else np.min(V_top[:, j]) for j in range(m)])
    nis = np.array([np.min(V_top[:, j]) if IS_BENEFIT[j] else np.max(V_top[:, j]) for j in range(m)])
    d_pos = np.sqrt(np.sum((V_top - pis) ** 2, axis=1))
    d_neg = np.sqrt(np.sum((V_top - nis) ** 2, axis=1))
    topsis_scores = d_neg / (d_pos + d_neg)
    topsis_ranks = stats.rankdata(-topsis_scores, method="min").astype(int)

    # 3. AHP-ELECTRE I
    U = np.zeros_like(X_eps)
    for j in range(m):
        c_min, c_max = np.min(X_eps[:, j]), np.max(X_eps[:, j])
        span = max(c_max - c_min, eps)
        U[:, j] = ((X_eps[:, j] - c_min) / span) if IS_BENEFIT[j] else ((c_max - X_eps[:, j]) / span)
    C_mat = np.zeros((n, n), dtype=np.float64)
    D_mat = np.zeros((n, n), dtype=np.float64)
    for a in range(n):
        for b in range(n):
            if a == b:
                continue
            C_mat[a, b] = np.sum(w_ahp[U[a, :] >= U[b, :] - 1e-12])
            D_mat[a, b] = np.max(np.maximum(0.0, U[b, :] - U[a, :]))
    phi_c = (np.sum(C_mat, axis=1) - np.sum(C_mat, axis=0)) / (n - 1)
    phi_d = (np.sum(D_mat, axis=1) - np.sum(D_mat, axis=0)) / (n - 1)
    electre_flow = phi_c - phi_d
    electre_ranks = stats.rankdata(-electre_flow, method="min").astype(int)

    # 4. AHP-WP
    wp_scores = np.ones(n, dtype=np.float64)
    for j in range(m):
        wp_scores *= (X_eps[:, j] ** (w_ahp[j] if IS_BENEFIT[j] else -w_ahp[j]))
    wp_scores /= np.max(wp_scores)
    wp_ranks = stats.rankdata(-wp_scores, method="min").astype(int)

    return pd.DataFrame({
        "student_id": df_duration[id_col].astype(str),
        "AHP_SAW_Score": [f"{v:.6f}" for v in saw_scores],
        "AHP_SAW_Rank": saw_ranks,
        "AHP_TOPSIS_Score": [f"{v:.6f}" for v in topsis_scores],
        "AHP_TOPSIS_Rank": topsis_ranks,
        "AHP_ELECTRE_NetFlow": [f"{v:.6f}" for v in electre_flow],
        "AHP_ELECTRE_Rank": electre_ranks,
        "AHP_WP_Score": [f"{v:.6f}" for v in wp_scores],
        "AHP_WP_Rank": wp_ranks,
        "Equal_SAW_Score": [f"{v:.6f}" for v in eq_scores],
        "Equal_SAW_Rank": eq_ranks,
    }).sort_values("AHP_SAW_Rank").reset_index(drop=True)


def compute_mcdm_concordance_holm_summary(df_mcdm: pd.DataFrame) -> pd.DataFrame:
    """
    Computes Spearman rho, Fisher-z 95% CIs, Kendall tau, Mean/Max rank shifts, Top-1 match,
    Top-5 overlap, and Holm-Bonferroni corrected p-values across the 4 MCDM baselines.
    """
    df = df_mcdm.sort_values("student_id").reset_index(drop=True)
    n_students = len(df)
    saw_scores = df["AHP_SAW_Score"].astype(float).to_numpy()
    saw_ranks = df["AHP_SAW_Rank"].astype(int).to_numpy()
    top5_saw = set(df["student_id"][saw_ranks <= 5])
    top1_saw = df["student_id"].iloc[int(np.argmin(saw_ranks))]

    comparisons = [
        ("AHP-SAW vs AHP-TOPSIS (Reviewer Priority 3)", "AHP_TOPSIS_Score", "AHP_TOPSIS_Rank"),
        ("AHP-SAW vs AHP-ELECTRE I (Reviewer Priority 3)", "AHP_ELECTRE_NetFlow", "AHP_ELECTRE_Rank"),
        ("AHP-SAW vs AHP-WP (Multiplicative Baseline)", "AHP_WP_Score", "AHP_WP_Rank"),
        ("AHP-SAW vs Equal-Weight SAW (Unweighted Baseline)", "Equal_SAW_Score", "Equal_SAW_Rank"),
    ]
    rows = []
    raw_p_rho, raw_p_tau = [], []
    for label, sc_col, rk_col in comparisons:
        sc = df[sc_col].astype(float).to_numpy()
        rk = df[rk_col].astype(int).to_numpy()
        rho, p_r = stats.spearmanr(saw_scores, sc)
        tau, p_t = stats.kendalltau(saw_scores, sc)
        ci_lo, ci_hi = fisher_z_ci(rho, n=n_students)
        top5_other = set(df["student_id"][rk <= 5])
        top5_ov = len(top5_saw & top5_other) / 5.0
        top1_other = df["student_id"].iloc[int(np.argmin(rk))]
        mean_shift = float(np.mean(np.abs(saw_ranks - rk)))
        max_shift = int(np.max(np.abs(saw_ranks - rk)))
        raw_p_rho.append(float(p_r))
        raw_p_tau.append(float(p_t))
        rows.append({
            "Comparison_Pair": label,
            "Spearman_Rho": round(float(rho), 4),
            "Fisher_z_95CI_Low": round(float(ci_lo), 4),
            "Fisher_z_95CI_High": round(float(ci_hi), 4),
            "Raw_p_Rho": float(p_r),
            "Kendall_Tau": round(float(tau), 4),
            "Raw_p_Tau": float(p_t),
            "Mean_Rank_Shift": round(mean_shift, 2),
            "Max_Rank_Shift": max_shift,
            "Top1_Match": bool(top1_saw == top1_other),
            "Top5_Overlap_Ratio": round(float(top5_ov), 2),
        })

    holm_rho = holm_bonferroni_adjust(raw_p_rho)
    holm_tau = holm_bonferroni_adjust(raw_p_tau)
    for i, r in enumerate(rows):
        r["Holm_Bonferroni_Adj_p_Rho"] = holm_rho[i]
        r["Holm_Bonferroni_Adj_p_Tau"] = holm_tau[i]
        r["Significant_Alpha_0.05_Holm"] = bool(holm_rho[i] < 0.05)

    return pd.DataFrame(rows)
