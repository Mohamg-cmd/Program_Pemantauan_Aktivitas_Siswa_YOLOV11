"""
Within-Session Temporal Consistency and Robustness Analysis Engine (Targeted Fix v2.1)
Primary Workspace: MyDrive/YOLO11_AHP_SAW_IJITCS_REVALIDATION_2026

Locked Temporal Segmentation Specification (Section 2 Mandatory Rule):
- Segmen_1 (00:00 - 05:00): frames 0 .. 7,499 (exactly 7,500 frames = 300.00 s at 25 FPS; 21-student ceiling = 6,300.00 s)
- Segmen_2 (05:00 - 10:00): frames 7,500 .. 14,999 (exactly 7,500 frames = 300.00 s at 25 FPS; 21-student ceiling = 6,300.00 s)
- Segmen_3 (10:00 - 15:00): frames 15,000 .. 22,499 (exactly 7,500 frames = 300.00 s at 25 FPS; 21-student ceiling = 6,300.00 s)
- Residual_Recording (15:00 - 15:38.84): frames 22,500 .. 23,470 (971 frames = 38.84 s at 25 FPS; 21-student ceiling = 815.64 s)
  *Explicitly separated from Segmen_3.*
- Full_Session (00:00 - 15:38.84): frames 0 .. 23,470 (all 23,471 frames = 938.84 s at 25 FPS; 21-student ceiling = 19,715.64 s)
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger('YOLO11_IJITCS_TemporalConsistency')


FPS_LOCKED: float = 25.0
TOTAL_FRAMES_LOCKED: int = 23471
CRITERIA_COLS: List[str] = ["Angkat Tangan", "Berdiri", "Diam", "Tidur"]
IS_BENEFIT_COL: List[bool] = [True, True, False, False]

LOCKED_TEMPORAL_WINDOWS: List[Dict[str, Any]] = [
    {"window_id": "Segmen_1", "label": "Segmen 1 (00:00-05:00)", "start_frame": 0, "end_frame": 7499, "n_frames": 7500, "duration_sec": 300.00, "theoretical_cap_sec": 6300.00, "is_primary_segment": True},
    {"window_id": "Segmen_2", "label": "Segmen 2 (05:00-10:00)", "start_frame": 7500, "end_frame": 14999, "n_frames": 7500, "duration_sec": 300.00, "theoretical_cap_sec": 6300.00, "is_primary_segment": True},
    {"window_id": "Segmen_3", "label": "Segmen 3 (10:00-15:00)", "start_frame": 15000, "end_frame": 22499, "n_frames": 7500, "duration_sec": 300.00, "theoretical_cap_sec": 6300.00, "is_primary_segment": True},
    {"window_id": "Residual_Recording", "label": "Residual Recording (15:00-15:38.84)", "start_frame": 22500, "end_frame": 23470, "n_frames": 971, "duration_sec": 38.84, "theoretical_cap_sec": 815.64, "is_primary_segment": False},
    {"window_id": "Cum_10Min", "label": "Cumulative 10-Min (00:00-10:00)", "start_frame": 0, "end_frame": 14999, "n_frames": 15000, "duration_sec": 600.00, "theoretical_cap_sec": 12600.00, "is_primary_segment": False},
    {"window_id": "Cum_15Min", "label": "Cumulative 15-Min (00:00-15:00)", "start_frame": 0, "end_frame": 22499, "n_frames": 22500, "duration_sec": 900.00, "theoretical_cap_sec": 18900.00, "is_primary_segment": False},
    {"window_id": "Full_Session", "label": "Full Session (00:00-15:38.84)", "start_frame": 0, "end_frame": 23470, "n_frames": 23471, "duration_sec": 938.84, "theoretical_cap_sec": 19715.64, "is_primary_segment": False},
]


def load_and_deduplicate_identity_assignments(identity_csv_path: Path) -> pd.DataFrame:
    """
    Critical Fix (BUG-02): Reads `identity_assignments.csv` using the actual schema column `assignment_status`
    (from `src/identity.py` lines 392 & 844, with fallback to `mapping_status` if needed), filters `== 'assigned'`,
    and deduplicates `(frame, student_id)` pairs so no student is ever double-counted in a single frame.
    """
    identity_csv_path = Path(identity_csv_path)
    if not identity_csv_path.exists():
        raise FileNotFoundError(f"identity_assignments.csv not found at {identity_csv_path}")

    header_cols = pd.read_csv(identity_csv_path, nrows=0).columns.tolist()
    if "assignment_status" in header_cols:
        status_col = "assignment_status"
    elif "mapping_status" in header_cols:
        status_col = "mapping_status"
    else:
        raise KeyError(f"Neither 'assignment_status' nor 'mapping_status' found in {identity_csv_path}: {header_cols}")

    use_cols = ["frame", "student_id", "class_name", status_col]
    for opt_col in ["track_id", "track_identity_confidence", "track_seat_purity", "confidence"]:
        if opt_col in header_cols:
            use_cols.append(opt_col)

    df_raw = pd.read_csv(identity_csv_path, usecols=use_cols)
    df_assigned = df_raw[df_raw[status_col].astype(str) == "assigned"].copy()
    df_assigned["frame"] = pd.to_numeric(df_assigned["frame"], errors="raise").astype(int)
    df_assigned["student_id"] = df_assigned["student_id"].astype(str).str.strip()

    # Deterministic lexicographic deduplication matching src/identity.py:build_student_level_detections
    sort_cols = ["frame", "student_id"]
    asc_flags = [True, True]
    for prio_col, asc in [("track_identity_confidence", False), ("track_seat_purity", False), ("confidence", False), ("track_id", True)]:
        if prio_col in df_assigned.columns:
            sort_cols.append(prio_col)
            asc_flags.append(asc)

    df_assigned = df_assigned.sort_values(sort_cols, ascending=asc_flags, kind="mergesort")
    dup_before = int(df_assigned.duplicated(subset=["frame", "student_id"]).sum())
    df_dedup = df_assigned.drop_duplicates(subset=["frame", "student_id"], keep="first").reset_index(drop=True)
    logger.info(
        "Loaded identity_assignments.csv via status_col='%s': raw=%d, assigned=%d, duplicates_removed=%d, unique_primary=%d",
        status_col, len(df_raw), len(df_assigned), dup_before, len(df_dedup),
    )
    return df_dedup


def build_direct_frame_counted_matrices_and_profiles(
    df_primary: pd.DataFrame,
    student_ids: Optional[List[str]] = None,
    fps: float = FPS_LOCKED,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Forms exact 21-student x 4-activity duration matrices (in seconds at FPS=25.0) directly from frame-level
    observations across:
    - Segmen_1 (frames 0..7499 = 300.00 s)
    - Segmen_2 (frames 7500..14999 = 300.00 s)
    - Segmen_3 (frames 15000..22499 = 300.00 s)
    - Residual_Recording (frames 22500..23470 = 38.84 s)
    - Cum_10Min (frames 0..14999 = 600.00 s)
    - Cum_15Min (frames 0..22499 = 900.00 s)
    - Full_Session (frames 0..23470 = 938.84 s)
    """
    if student_ids is None:
        student_ids = [f"S{i:02d}" for i in range(1, 22)]

    matrices_by_window: Dict[str, pd.DataFrame] = {}
    profile_rows: List[Dict[str, Any]] = []

    for w in LOCKED_TEMPORAL_WINDOWS:
        wid = w["window_id"]
        f0, f1 = int(w["start_frame"]), int(w["end_frame"])
        dur_sec = float(w["duration_sec"])
        cap_sec = float(w["theoretical_cap_sec"])

        sub = df_primary[(df_primary["frame"] >= f0) & (df_primary["frame"] <= f1)]
        freq_mat = (
            sub.groupby(["student_id", "class_name"])["frame"]
            .nunique()
            .unstack(fill_value=0)
            .reindex(index=student_ids, columns=CRITERIA_COLS, fill_value=0)
        )
        dur_mat = freq_mat.astype(np.float64) / float(fps)
        dur_mat.index.name = "student_id"
        matrices_by_window[wid] = dur_mat

        student_totals = dur_mat.sum(axis=1)
        max_single_student = float(student_totals.max()) if len(student_totals) else 0.0
        total_obs_sec = float(student_totals.sum())
        class_totals = dur_mat.sum(axis=0)

        profile_rows.append({
            "Window_ID": wid,
            "Window_Label": w["label"],
            "Frame_Range": f"{f0}-{f1}",
            "Window_Frames": int(w["n_frames"]),
            "Window_Duration_Sec": dur_sec,
            "Theoretical_Cap_Sec": cap_sec,
            "Direct_Observed_Sec": round(total_obs_sec, 4),
            "Coverage_Pct": round((total_obs_sec / cap_sec) * 100.0, 2) if cap_sec > 0 else 0.0,
            "Max_Single_Student_Sec": round(max_single_student, 4),
            "Exceeds_Cap_Anomaly": bool(total_obs_sec > cap_sec + 1e-6 or max_single_student > dur_sec + 1e-6),
            "Angkat_Tangan_Sec": round(float(class_totals["Angkat Tangan"]), 4),
            "Berdiri_Sec": round(float(class_totals["Berdiri"]), 4),
            "Diam_Sec": round(float(class_totals["Diam"]), 4),
            "Tidur_Sec": round(float(class_totals["Tidur"]), 4),
        })

    profiles_df = pd.DataFrame(profile_rows)
    return profiles_df, matrices_by_window


def compute_biproportional_ipf_3x300_plus_residual(
    dur_full_df: pd.DataFrame,
    track_to_student_df: pd.DataFrame,
    track_summary_df: pd.DataFrame,
    fps: float = FPS_LOCKED,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Computes the Biproportional Iterative Proportional Fitting (IPF) segment duration matrices and profiles
    under the locked 4-part partition:
    - Segmen_1 (0..7499, cap 300.00 s/student)
    - Segmen_2 (7500..14999, cap 300.00 s/student)
    - Segmen_3 (15000..22499, cap 300.00 s/student)
    - Residual_Recording (22500..23470, cap 38.84 s/student)
    while preserving exact full-session student x activity totals (`19,363.72 s` across 23,471 frames).
    """
    id_col = "entity" if "entity" in dur_full_df.columns else "student_id"
    dur_full = dur_full_df.copy().sort_values(id_col).reset_index(drop=True)
    student_ids = dur_full[id_col].astype(str).tolist()

    m = pd.merge(
        track_to_student_df,
        track_summary_df[["track_id", "span_frames", "dominant_activity"]],
        on="track_id",
        how="left",
    )
    ass = m[m["mapping_status"].astype(str) == "assigned"].copy()

    disjoint_windows = [
        ("Segmen_1", "Segmen 1 (00:00-05:00)", 0, 7499, 7500, 300.00, 6300.00),
        ("Segmen_2", "Segmen 2 (05:00-10:00)", 7500, 14999, 7500, 300.00, 6300.00),
        ("Segmen_3", "Segmen 3 (10:00-15:00)", 15000, 22499, 7500, 300.00, 6300.00),
        ("Residual_Recording", "Residual Recording (15:00-15:38.84)", 22500, 23470, 971, 38.84, 815.64),
    ]

    for wid, _, f0, f1, _, _, _ in disjoint_windows:
        ov_start = np.maximum(ass["first_frame"].to_numpy(dtype=np.float64), float(f0))
        ov_end = np.minimum(ass["last_frame"].to_numpy(dtype=np.float64), float(f1))
        ov_span = np.maximum(ov_end - ov_start + 1.0, 0.0)
        span_f = np.maximum(ass["span_frames"].to_numpy(dtype=np.float64), 1.0)
        frac = ov_span / span_f
        ass[f"sec_{wid}"] = (ass["n_frames"].to_numpy(dtype=np.float64) * frac) / float(fps)

    # Build initial prior tensor P[student, window, class] of shape (21, 4, 4)
    n_students = len(student_ids)
    n_win = len(disjoint_windows)
    n_cls = len(CRITERIA_COLS)
    win_caps = np.array([w[5] for w in disjoint_windows], dtype=np.float64)
    full_matrix = dur_full[CRITERIA_COLS].to_numpy(dtype=np.float64)  # (21, 4)

    tensor = np.zeros((n_students, n_win, n_cls), dtype=np.float64)
    for s_idx, sid in enumerate(student_ids):
        sub_s = ass[ass["student_id"].astype(str) == sid]
        all_w = np.array([sub_s[f"sec_{w[0]}"].sum() for w in disjoint_windows], dtype=np.float64)
        if all_w.sum() <= 0:
            all_w = win_caps / win_caps.sum()
        else:
            all_w = all_w / all_w.sum()

        for c_idx, cname in enumerate(CRITERIA_COLS):
            tot_sc = full_matrix[s_idx, c_idx]
            if tot_sc <= 0:
                continue
            sub_sc = sub_s[sub_s["dominant_activity"].astype(str) == cname]
            cw = np.array([sub_sc[f"sec_{w[0]}"].sum() for w in disjoint_windows], dtype=np.float64)
            if cw.sum() > 0:
                cw = 0.75 * (cw / cw.sum()) + 0.25 * all_w
            else:
                cw = all_w.copy()
            tensor[s_idx, :, c_idx] = tot_sc * cw

        # Enforce per-student per-window ceiling <= win_caps[w_idx] via iterative water-filling projection
        for _ in range(50):
            win_sums = tensor[s_idx].sum(axis=1)  # (4,)
            over_mask = win_sums > win_caps + 1e-9
            if not np.any(over_mask):
                break
            for w_idx in np.where(over_mask)[0]:
                scale_down = win_caps[w_idx] / win_sums[w_idx]
                excess = tensor[s_idx, w_idx, :] * (1.0 - scale_down)
                tensor[s_idx, w_idx, :] *= scale_down
                under_mask = ~over_mask
                slack = np.maximum(win_caps[under_mask] - tensor[s_idx, under_mask, :].sum(axis=1), 0.0)
                if slack.sum() > 0:
                    weights_under = slack / slack.sum()
                    tensor[s_idx, under_mask, :] += weights_under[:, None] * excess[None, :]

            # Re-normalize across windows to preserve exact full-session student x class totals
            cur_cls_sums = tensor[s_idx].sum(axis=0)
            for c_idx in range(n_cls):
                if cur_cls_sums[c_idx] > 0:
                    tensor[s_idx, :, c_idx] *= (full_matrix[s_idx, c_idx] / cur_cls_sums[c_idx])

    ipf_matrices: Dict[str, pd.DataFrame] = {}
    for w_idx, (wid, _, _, _, _, _, _) in enumerate(disjoint_windows):
        df_w = pd.DataFrame(tensor[:, w_idx, :], index=student_ids, columns=CRITERIA_COLS)
        df_w.index.name = "student_id"
        ipf_matrices[wid] = df_w

    ipf_matrices["Cum_10Min"] = ipf_matrices["Segmen_1"] + ipf_matrices["Segmen_2"]
    ipf_matrices["Cum_15Min"] = ipf_matrices["Segmen_1"] + ipf_matrices["Segmen_2"] + ipf_matrices["Segmen_3"]
    ipf_matrices["Full_Session"] = ipf_matrices["Cum_15Min"] + ipf_matrices["Residual_Recording"]

    ipf_rows = []
    for w in LOCKED_TEMPORAL_WINDOWS:
        wid = w["window_id"]
        mat = ipf_matrices[wid]
        st_sums = mat.sum(axis=1)
        tot_sec = float(st_sums.sum())
        max_st = float(st_sums.max())
        cls_sums = mat.sum(axis=0)
        ipf_rows.append({
            "Window_ID": wid,
            "Window_Label": w["label"],
            "Frame_Range": f"{w['start_frame']}-{w['end_frame']}",
            "Window_Frames": int(w["n_frames"]),
            "Window_Duration_Sec": float(w["duration_sec"]),
            "Theoretical_Cap_Sec": float(w["theoretical_cap_sec"]),
            "IPF_Observed_Sec": round(tot_sec, 4),
            "Coverage_Pct": round((tot_sec / float(w["theoretical_cap_sec"])) * 100.0, 2),
            "Max_Single_Student_Sec": round(max_st, 4),
            "Exceeds_Cap_Anomaly": bool(tot_sec > float(w["theoretical_cap_sec"]) + 1e-4 or max_st > float(w["duration_sec"]) + 1e-4),
            "Angkat_Tangan_Sec": round(float(cls_sums["Angkat Tangan"]), 4),
            "Berdiri_Sec": round(float(cls_sums["Berdiri"]), 4),
            "Diam_Sec": round(float(cls_sums["Diam"]), 4),
            "Tidur_Sec": round(float(cls_sums["Tidur"]), 4),
        })

    return pd.DataFrame(ipf_rows), ipf_matrices


def compare_direct_vs_ipf_profiles(direct_df: pd.DataFrame, ipf_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes exact absolute difference (Direct - IPF in seconds) and relative percentage difference
    (|Direct - IPF| / max(Direct, 1e-6) * 100%) across all windows and activity classes.
    """
    m = pd.merge(
        direct_df[["Window_ID", "Window_Label", "Frame_Range", "Window_Duration_Sec", "Direct_Observed_Sec", "Angkat_Tangan_Sec", "Berdiri_Sec", "Diam_Sec", "Tidur_Sec"]],
        ipf_df[["Window_ID", "IPF_Observed_Sec", "Angkat_Tangan_Sec", "Berdiri_Sec", "Diam_Sec", "Tidur_Sec"]],
        on="Window_ID",
        suffixes=("_Direct", "_IPF"),
    )
    m["Abs_Diff_Total_Sec"] = (m["Direct_Observed_Sec"] - m["IPF_Observed_Sec"]).round(4)
    m["Rel_Diff_Total_Pct"] = (
        np.abs(m["Direct_Observed_Sec"] - m["IPF_Observed_Sec"]) / np.maximum(m["Direct_Observed_Sec"], 1e-6) * 100.0
    ).round(4)

    for col in ["Angkat_Tangan_Sec", "Berdiri_Sec", "Diam_Sec", "Tidur_Sec"]:
        d_col = f"{col}_Direct"
        i_col = f"{col}_IPF"
        short = col.replace("_Sec", "")
        m[f"Abs_Diff_{short}_Sec"] = (m[d_col] - m[i_col]).round(4)
        m[f"Rel_Diff_{short}_Pct"] = (
            np.abs(m[d_col] - m[i_col]) / np.maximum(m[d_col], 1e-6) * 100.0
        ).round(4)

    return m


def compute_ahp_saw_segment_rankings(
    window_matrices: Dict[str, pd.DataFrame],
    eps: float = 1e-6,
) -> pd.DataFrame:
    """
    Self-contained AHP-SAW score and rank computation across Segmen_1 (300s), Segmen_2 (300s),
    Segmen_3 (300s), Residual_Recording (38.84s), Cum_10Min (600s), Cum_15Min (900s), and Full_Session (938.84s).
    """
    A = np.array([
        [1.0, 2.0, 3.0, 3.0],
        [0.5, 1.0, 2.0, 3.0],
        [1.0 / 3.0, 0.5, 1.0, 2.0],
        [1.0 / 3.0, 1.0 / 3.0, 0.5, 1.0],
    ], dtype=np.float64)
    eigvals, eigvecs = np.linalg.eig(A)
    max_idx = int(np.argmax(eigvals.real))
    w_ahp = np.abs(eigvecs[:, max_idx].real)
    w_ahp = w_ahp / np.sum(w_ahp)

    first_mat = next(iter(window_matrices.values()))
    student_ids = first_mat.index.astype(str).tolist()
    out_df = pd.DataFrame({"student_id": student_ids})

    for w in LOCKED_TEMPORAL_WINDOWS:
        wid = w["window_id"]
        if wid not in window_matrices:
            continue
        mat = window_matrices[wid].reindex(index=student_ids, columns=CRITERIA_COLS, fill_value=0.0)
        X = mat.to_numpy(dtype=np.float64)
        X_eps = np.where(X <= 0.0, eps, X)
        R = np.zeros_like(X_eps)
        for j in range(len(CRITERIA_COLS)):
            if IS_BENEFIT_COL[j]:
                mx = np.max(X_eps[:, j])
                R[:, j] = (X[:, j] / mx) if mx > 0 else 0.0
            else:
                R[:, j] = np.min(X_eps[:, j]) / X_eps[:, j]
        scores = R @ w_ahp
        ranks = stats.rankdata(-scores, method="min").astype(int)
        out_df[f"{wid}_Score"] = [round(float(v), 6) for v in scores]
        out_df[f"{wid}_Rank"] = ranks

    if "Full_Session_Rank" in out_df.columns:
        out_df = out_df.sort_values("Full_Session_Rank").reset_index(drop=True)
    return out_df


def holm_bonferroni_step_down(p_values: List[float]) -> List[float]:
    """Self-contained Holm-Bonferroni step-down multiple-testing correction."""
    m = len(p_values)
    if m == 0:
        return []
    p_arr = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(p_arr, kind="mergesort")
    adj_out = np.zeros(m, dtype=np.float64)
    running_max = 0.0
    for idx_pos, orig_i in enumerate(order):
        adj = min(1.0, float(p_arr[orig_i]) * float(m - idx_pos))
        running_max = max(running_max, adj)
        adj_out[orig_i] = running_max
    return adj_out.tolist()


def compute_temporal_concordance_holm_bonferroni(rankings_df: pd.DataFrame) -> pd.DataFrame:
    """
    Self-contained Spearman rho, Fisher-z 95% CI, Kendall tau, rank shift, top-k overlap,
    and Holm-Bonferroni step-down correction across the locked 3x300s segments and cumulative windows.
    """
    df = rankings_df.sort_values("student_id").reset_index(drop=True)
    n = len(df)
    pairs = [
        ("Segmen 1 (00:00-05:00) vs Full Session (00:00-15:38.84)", "Segmen_1", "Full_Session"),
        ("Segmen 2 (05:00-10:00) vs Full Session (00:00-15:38.84)", "Segmen_2", "Full_Session"),
        ("Segmen 3 (10:00-15:00) vs Full Session (00:00-15:38.84)", "Segmen_3", "Full_Session"),
        ("Segmen 1 (00:00-05:00) vs Segmen 2 (05:00-10:00)", "Segmen_1", "Segmen_2"),
        ("Segmen 2 (05:00-10:00) vs Segmen 3 (10:00-15:00)", "Segmen_2", "Segmen_3"),
        ("Segmen 1 (00:00-05:00) vs Segmen 3 (10:00-15:00)", "Segmen_1", "Segmen_3"),
        ("Cumulative 5-Min (00:00-05:00) vs Cumulative 10-Min (00:00-10:00)", "Segmen_1", "Cum_10Min"),
        ("Cumulative 10-Min (00:00-10:00) vs Cumulative 15-Min (00:00-15:00)", "Cum_10Min", "Cum_15Min"),
        ("Cumulative 15-Min (00:00-15:00) vs Full Session (00:00-15:38.84)", "Cum_15Min", "Full_Session"),
    ]
    rows = []
    raw_p_rho, raw_p_tau = [], []
    for label, w_a, w_b in pairs:
        sc_a = df[f"{w_a}_Score"].astype(float).to_numpy()
        sc_b = df[f"{w_b}_Score"].astype(float).to_numpy()
        rk_a = df[f"{w_a}_Rank"].astype(int).to_numpy()
        rk_b = df[f"{w_b}_Rank"].astype(int).to_numpy()

        rho, p_r = stats.spearmanr(sc_a, sc_b)
        tau, p_t = stats.kendalltau(sc_a, sc_b)
        r_clip = min(max(float(rho), -0.999999), 0.999999)
        z = np.arctanh(r_clip)
        se = 1.0 / np.sqrt(max(n - 3, 1))
        ci_lo, ci_hi = float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))

        top1_a = df["student_id"].iloc[int(np.argmin(rk_a))]
        top1_b = df["student_id"].iloc[int(np.argmin(rk_b))]
        top5_ov = len(set(df["student_id"][rk_a <= 5]) & set(df["student_id"][rk_b <= 5])) / 5.0
        top10_ov = len(set(df["student_id"][rk_a <= 10]) & set(df["student_id"][rk_b <= 10])) / 10.0

        raw_p_rho.append(float(p_r))
        raw_p_tau.append(float(p_t))
        rows.append({
            "Comparison": label,
            "Spearman_Rho": round(float(rho), 4),
            "Fisher_z_95CI_Low": round(ci_lo, 4),
            "Fisher_z_95CI_High": round(ci_hi, 4),
            "Raw_p_Rho": float(p_r),
            "Kendall_Tau": round(float(tau), 4),
            "Raw_p_Tau": float(p_t),
            "Mean_Rank_Shift": round(float(np.mean(np.abs(rk_a - rk_b))), 2),
            "Max_Rank_Shift": int(np.max(np.abs(rk_a - rk_b))),
            "Top1_Match": bool(top1_a == top1_b),
            "Top5_Overlap": round(float(top5_ov), 2),
            "Top10_Overlap": round(float(top10_ov), 2),
        })

    adj_rho = holm_bonferroni_step_down(raw_p_rho)
    adj_tau = holm_bonferroni_step_down(raw_p_tau)
    for i, r in enumerate(rows):
        r["Holm_Adj_p_Rho"] = adj_rho[i]
        r["Holm_Adj_p_Tau"] = adj_tau[i]
        r["Significant_Alpha_0.05_Holm"] = bool(adj_rho[i] < 0.05)

    return pd.DataFrame(rows)