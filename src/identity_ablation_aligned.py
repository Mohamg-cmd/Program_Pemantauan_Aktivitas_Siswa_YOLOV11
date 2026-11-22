"""
Fixed-Seat Identity Component Diagnostic Ablation & Controlled Seating Perturbation Suite (Targeted Fix v2.1)
Primary Workspace: MyDrive/YOLO11_AHP_SAW_IJITCS_REVALIDATION_2026

Fixes Applied (BUG-01 & BUG-08 + Section 3 Controlled Seating Perturbation):
1. Fixed `ABLATION_VARIANTS` / `ABLATION_CONFIG_OVERRIDES` naming consistency so both symbols are exported.
2. Ensured each of the 7 ablation variants genuinely modifies the active mathematical component in `src/identity.py`:
   - Variant 1 (`1_Full_Fixed_Seat_Identity`): Unmodified baseline (`purity_agg_weight=0.45, mean_agg_weight=0.35, median_agg_weight=0.20`).
   - Variant 2 (`2_Without_Temporal_Evidence`): Overrides track-level dominant seat selection so `purity_agg_weight=0.0`
     (`aggregate = 0.55 * mean_scores + 0.45 * median_scores`), sets `weight_temporal=0.0`, and removes purity dominance.
   - Variant 3 (`3_Without_Appearance_Evidence`): Explicitly checks whether external ReID similarity features (`reid_similarity_df`)
     are present. Because `detections.csv` was generated with `with_reid=False`, it flags
     `status = "REQUIRES_EXTERNAL_REID_FEATURES"` instead of falsely claiming ReID ablation was executed on ReID-free detections!
   - Variant 4 (`4_Without_ROI_Bonus`): Sets `roi_bonus = 0.0` (removes rectangular ROI containment bonus).
   - Variant 5 (`5_Without_Short_Track_Smoothing_Gate`): Sets `short_track_frames = 1` and `short_track_min_seat_score = 0.45`.
   - Variant 6 (`6_Without_Ambiguity_Hysteresis_Margin`): Sets `track_ambiguity_margin = 0.0` and disables ambiguity rejection.
   - Variant 7 (`7_Without_Long_Gap_Reassociation`): Actively identifies track fragments with `long_gap == True`
     (`intertrack_gap_frames > 90` frames) via `merge_track_fragments()` and revokes their seat assignments (`status -> unassigned`)
     before recomputing frame conflict resolution and validation metrics.
3. Added Section 3 Reviewer Cross-Scenario Response Suite: `run_controlled_seating_perturbation_suite()`, which tests
   controlled seat geometry jitter (`5px..50px`), ROI scaling (`0.50x..1.25x`), row/column pitch compression (`15%`),
   and adjacent seat swaps while keeping actual observations strictly separated from controlled simulations.
"""
from __future__ import annotations
import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("YOLO11_IJITCS_IdentityAblation")

ABLATION_CONFIG_OVERRIDES: Dict[str, Dict[str, Any]] = {
    "1_Full_Fixed_Seat_Identity": {
        "description": "Full Fixed-Seat Identity Resolution baseline (spatial decay + ROI bonus + temporal purity dominance + short-track gate + ambiguity margin + long-gap acceptance)",
        "cfg_overrides": {},
        "purity_agg_weight": 0.45,
        "reject_long_gap_fragments": False,
        "disable_ambiguity_gate": False,
        "requires_reid_features": False,
    },
    "2_Without_Temporal_Evidence": {
        "description": "Removes temporal purity dominance (purity_agg_weight=0.0 in track seat scoring, weight_temporal=0.0, min_track_seat_purity=0.0)",
        "cfg_overrides": {"weight_temporal": 0.0, "min_track_seat_purity": 0.0},
        "purity_agg_weight": 0.0,
        "reject_long_gap_fragments": False,
        "disable_ambiguity_gate": False,
        "requires_reid_features": False,
    },
    "3_Without_Appearance_Evidence": {
        "description": "Secondary ReID tie-breaker ablation (NOTE: detections.csv has with_reid=False; requires external ReID embeddings or video re-inference with ReID enabled to test active delta)",
        "cfg_overrides": {"allow_reid_secondary": False, "weight_reid": 0.0},
        "purity_agg_weight": 0.45,
        "reject_long_gap_fragments": False,
        "disable_ambiguity_gate": False,
        "requires_reid_features": True,
    },
    "4_Without_ROI_Bonus": {
        "description": "Removes rectangular seat ROI containment bonus (roi_bonus=0.0; pure Euclidean distance decay)",
        "cfg_overrides": {"roi_bonus": 0.0},
        "purity_agg_weight": 0.45,
        "reject_long_gap_fragments": False,
        "disable_ambiguity_gate": False,
        "requires_reid_features": False,
    },
    "5_Without_Short_Track_Smoothing_Gate": {
        "description": "Disables stricter spatial score gate on short tracks < 10 frames (short_track_frames=1, short_track_min_seat_score=0.45)",
        "cfg_overrides": {"short_track_frames": 1, "short_track_min_seat_score": 0.45},
        "purity_agg_weight": 0.45,
        "reject_long_gap_fragments": False,
        "disable_ambiguity_gate": False,
        "requires_reid_features": False,
    },
    "6_Without_Ambiguity_Hysteresis_Margin": {
        "description": "Disables inter-seat ambiguity margin rejection (track_ambiguity_margin=0.0; forces top-1 seat assignment without ambiguity gating)",
        "cfg_overrides": {"track_ambiguity_margin": 0.0},
        "purity_agg_weight": 0.45,
        "reject_long_gap_fragments": False,
        "disable_ambiguity_gate": True,
        "requires_reid_features": False,
    },
    "7_Without_Long_Gap_Reassociation": {
        "description": "Rejects all track fragments following an inter-track gap > 90 frames (3.60 s) instead of re-associating via fixed-seat evidence",
        "cfg_overrides": {"reassociation_gap_frames": 90},
        "purity_agg_weight": 0.45,
        "reject_long_gap_fragments": True,
        "disable_ambiguity_gate": False,
        "requires_reid_features": False,
    },
}

# Exported list alias to prevent NameError: ABLATION_VARIANTS in Notebook 08 Cell 5
ABLATION_VARIANTS: List[Dict[str, Any]] = [
    {"variant_id": k, **v} for k, v in ABLATION_CONFIG_OVERRIDES.items()
]


def _assign_tracks_with_ablation_hooks(
    detections_df: pd.DataFrame,
    seats_df: pd.DataFrame,
    cfg,
    base_identity_module,
    purity_agg_weight: float = 0.45,
    disable_ambiguity_gate: bool = False,
    reject_long_gap_fragments: bool = False,
    reid_similarity_df: Optional[pd.DataFrame] = None,
    fps: float = 25.0,
    total_video_frames: int = 23471,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Executes `assign_track_to_seat` with genuine hooks for `purity_agg_weight`, `disable_ambiguity_gate`,
    and `reject_long_gap_fragments` so every ablation variant modifies the exact mechanism claimed.
    """
    orig_quality_fn = base_identity_module._track_quality_from_scores

    def _hooked_track_quality(score_matrix, seats, track_values, cfg_inner):
        res = orig_quality_fn(score_matrix, seats, track_values, cfg_inner)
        if res is None:
            return None
        if abs(purity_agg_weight - 0.45) > 1e-9:
            valid_best = score_matrix.max(axis=1) > 0
            mean_scores = score_matrix[valid_best].mean(axis=0)
            median_scores = np.median(score_matrix[valid_best], axis=0)
            best_idx_frames = np.argmax(score_matrix, axis=1)
            best_counts = np.bincount(best_idx_frames[valid_best], minlength=score_matrix.shape[1]).astype(float)
            purity_vec = best_counts / max(int(valid_best.sum()), 1)
            rem = max(1.0 - purity_agg_weight, 0.0)
            w_mean = rem * (0.35 / 0.55)
            w_med = rem * (0.20 / 0.55)
            aggregate = purity_agg_weight * purity_vec + w_mean * mean_scores + w_med * median_scores
            order = np.argsort(aggregate)[::-1]
            best = int(order[0])
            second = float(aggregate[order[1]]) if len(order) > 1 else 0.0
            res["aggregate"] = aggregate
            res["best_idx"] = best
            res["best_score"] = float(aggregate[best])
            res["second_score"] = second
            res["margin"] = float(aggregate[best] - second)
            res["purity"] = float(purity_vec[best])
            res["mean_score"] = float(mean_scores[best])
            res["median_score"] = float(median_scores[best])
        if disable_ambiguity_gate:
            res["margin"] = 1.0
        return res

    try:
        base_identity_module._track_quality_from_scores = _hooked_track_quality
        assignments = base_identity_module.assign_track_to_seat(
            detections_df, seats_df, cfg, reid_similarity=reid_similarity_df
        )
    finally:
        base_identity_module._track_quality_from_scores = orig_quality_fn

    if reject_long_gap_fragments:
        t2s, _ = base_identity_module.merge_track_fragments(
            assignments, fps=fps, config=cfg, total_video_frames=total_video_frames, seats=seats_df
        )
        if not t2s.empty and "long_gap" in t2s.columns:
            lg_tids = set(t2s.loc[t2s["long_gap"].fillna(False).astype(bool), "track_id"].astype(int).tolist())
            if lg_tids:
                # Reset duplicate_track back to assigned before revoking long-gap tracks and re-resolving conflicts
                was_dup = assignments["assignment_status"].eq("duplicate_track")
                assignments.loc[was_dup, "assignment_status"] = "assigned"
                in_lg = assignments["track_id"].astype(int).isin(lg_tids)
                assignments.loc[in_lg, "student_id"] = None
                assignments.loc[in_lg, "seat_id"] = None
                assignments.loc[in_lg, "assignment_status"] = "unassigned"
                assignments.loc[in_lg, "track_mapping_status"] = "unassigned"
                assignments = base_identity_module._resolve_frame_conflicts(assignments)

    val_rep = base_identity_module.validate_seat_assignment(
        assignments, seats_df, cfg, total_video_frames=total_video_frames
    )
    return assignments, val_rep


def run_component_ablation(
    detections_df: pd.DataFrame,
    seats_df: pd.DataFrame,
    base_identity_module,
    reid_similarity_df: Optional[pd.DataFrame] = None,
    fps: float = 25.0,
    total_video_frames: int = 23471,
    enable_ablation: bool = False,
) -> pd.DataFrame:
    if not enable_ablation:
        raise RuntimeError(
            "[SAFETY GUARD LOCKED] Component-level identity ablation over 497,848 detections is locked. "
            "Set ENABLE_IDENTITY_ABLATION = True in Cell 1 to execute."
        )
    reid_available = bool(reid_similarity_df is not None and not reid_similarity_df.empty)
    results = []
    baseline_primary_map: Optional[pd.Series] = None

    for var_name, spec in ABLATION_CONFIG_OVERRIDES.items():
        if spec.get("requires_reid_features", False) and not reid_available:
            results.append({
                "variant": var_name,
                "execution_status": "REQUIRES_EXTERNAL_REID_FEATURES_OR_VIDEO_REINFERENCE",
                "reid_features_available": False,
                "reject_long_gap": False,
                "n_primary_assigned": None,
                "primary_assignment_rate_pct": None,
                "n_assigned_students": None,
                "mean_seat_purity_preprop_pct": None,
                "mean_student_frame_coverage_pct": None,
                "n_unassigned_detections": None,
                "n_long_gap_fragments": None,
                "identity_mapping_consistency_vs_full_pct": None,
                "notes": "detections.csv was generated with with_reid=False; run appearance feature extraction on video_part001.mp4 to evaluate ReID tie-breaker delta.",
            })
            continue

        cfg = base_identity_module.IdentityConfig.from_dict(dict(spec.get("cfg_overrides", {})))
        assignments, val_rep = _assign_tracks_with_ablation_hooks(
            detections_df=detections_df,
            seats_df=seats_df,
            cfg=cfg,
            base_identity_module=base_identity_module,
            purity_agg_weight=float(spec.get("purity_agg_weight", 0.45)),
            disable_ambiguity_gate=bool(spec.get("disable_ambiguity_gate", False)),
            reject_long_gap_fragments=bool(spec.get("reject_long_gap_fragments", False)),
            reid_similarity_df=reid_similarity_df,
            fps=fps,
            total_video_frames=total_video_frames,
        )
        cur_map = assignments["student_id"].fillna("UNASSIGNED").astype(str)
        if var_name == "1_Full_Fixed_Seat_Identity":
            baseline_primary_map = cur_map
            consistency_pct = 100.0
        else:
            consistency_pct = float((cur_map == baseline_primary_map).mean() * 100.0) if baseline_primary_map is not None else None

        results.append({
            "variant": var_name,
            "execution_status": "EXECUTED",
            "reid_features_available": reid_available,
            "reject_long_gap": bool(spec.get("reject_long_gap_fragments", False)),
            "n_primary_assigned": int(val_rep["n_primary_assigned"]),
            "primary_assignment_rate_pct": round(float(val_rep["primary_assignment_rate"]) * 100.0, 2),
            "n_assigned_students": int(val_rep["n_assigned_students"]),
            "mean_seat_purity_preprop_pct": round(float(val_rep["mean_seat_purity"]) * 100.0, 2),
            "mean_student_frame_coverage_pct": round(float(val_rep["mean_student_frame_coverage"]) * 100.0, 2),
            "n_unassigned_detections": int(val_rep["n_unassigned_detections"]),
            "n_long_gap_fragments": int(val_rep["n_long_gap_fragments"]),
            "identity_mapping_consistency_vs_full_pct": round(consistency_pct, 2) if consistency_pct is not None else None,
            "notes": spec["description"],
        })

    return pd.DataFrame(results)


# ==============================================================================
# SECTION 3: CONTROLLED SEATING ARRANGEMENT PERTURBATION SUITE
# (Strictly separated from actual observations; never claimed as cross-camera/classroom)
# ==============================================================================
def generate_perturbed_seat_map(
    base_seats_df: pd.DataFrame,
    scenario_type: str,
    param_value: float = 0.0,
    seed: int = 42,
    frame_w: float = 640.0,
    frame_h: float = 480.0,
) -> pd.DataFrame:
    """
    Generates a controlled perturbation of `student_seats.csv` while preserving the original seat map untouched.
    Supported `scenario_type`:
    - 'baseline': Unmodified `student_seats.csv` (Actual Calibration Reference).
    - 'spatial_jitter_px': Adds zero-mean Gaussian perturbation N(0, sigma^2) with sigma = `param_value` (px)
      to `(seat_x, seat_y)` and shifts `[roi_x1, roi_y1, roi_x2, roi_y2]` accordingly.
    - 'roi_scale': Scales each seat ROI width and height by `param_value` (e.g., 0.50, 0.75, 1.25) around `(seat_x, seat_y)`.
    - 'row_pitch_compression': Compresses vertical `seat_y` and ROI y-coordinates toward the vertical classroom center by `param_value` fraction.
    - 'col_pitch_compression': Compresses horizontal `seat_x` and ROI x-coordinates toward the horizontal classroom center by `param_value` fraction.
    - 'adjacent_seat_swap': Swaps seat coordinates of `int(param_value)` adjacent student seat pairs to simulate calibration permutation.
    """
    df = base_seats_df.copy().sort_values("student_id").reset_index(drop=True)
    if scenario_type == "baseline":
        return df

    rng = np.random.default_rng(seed)
    w_box = df["roi_x2"] - df["roi_x1"]
    h_box = df["roi_y2"] - df["roi_y1"]

    if scenario_type == "spatial_jitter_px":
        dx = rng.normal(0.0, float(param_value), size=len(df))
        dy = rng.normal(0.0, float(param_value), size=len(df))
        df["seat_x"] = np.clip(df["seat_x"] + dx, 10.0, frame_w - 10.0)
        df["seat_y"] = np.clip(df["seat_y"] + dy, 10.0, frame_h - 10.0)
        df["roi_x1"] = np.clip(df["roi_x1"] + dx, 0.0, frame_w - 20.0)
        df["roi_y1"] = np.clip(df["roi_y1"] + dy, 0.0, frame_h - 20.0)
        df["roi_x2"] = np.maximum(df["roi_x1"] + 10.0, np.clip(df["roi_x2"] + dx, 20.0, frame_w))
        df["roi_y2"] = np.maximum(df["roi_y1"] + 10.0, np.clip(df["roi_y2"] + dy, 20.0, frame_h))

    elif scenario_type == "roi_scale":
        scale = float(param_value)
        cx = 0.5 * (df["roi_x1"] + df["roi_x2"])
        cy = 0.5 * (df["roi_y1"] + df["roi_y2"])
        new_w = np.maximum(w_box * scale, 12.0)
        new_h = np.maximum(h_box * scale, 12.0)
        df["roi_x1"] = np.clip(cx - 0.5 * new_w, 0.0, frame_w - 15.0)
        df["roi_y1"] = np.clip(cy - 0.5 * new_h, 0.0, frame_h - 15.0)
        df["roi_x2"] = np.maximum(df["roi_x1"] + 10.0, np.clip(cx + 0.5 * new_w, 15.0, frame_w))
        df["roi_y2"] = np.maximum(df["roi_y1"] + 10.0, np.clip(cy + 0.5 * new_h, 15.0, frame_h))

    elif scenario_type == "row_pitch_compression":
        comp = float(param_value)
        y_center = float(df["seat_y"].mean())
        dy = (y_center - df["seat_y"]) * comp
        df["seat_y"] = df["seat_y"] + dy
        df["roi_y1"] = np.clip(df["roi_y1"] + dy, 0.0, frame_h - 20.0)
        df["roi_y2"] = np.maximum(df["roi_y1"] + 10.0, np.clip(df["roi_y2"] + dy, 20.0, frame_h))

    elif scenario_type == "col_pitch_compression":
        comp = float(param_value)
        x_center = float(df["seat_x"].mean())
        dx = (x_center - df["seat_x"]) * comp
        df["seat_x"] = df["seat_x"] + dx
        df["roi_x1"] = np.clip(df["roi_x1"] + dx, 0.0, frame_w - 20.0)
        df["roi_x2"] = np.maximum(df["roi_x1"] + 10.0, np.clip(df["roi_x2"] + dx, 20.0, frame_w))

    elif scenario_type == "adjacent_seat_swap":
        n_pairs = int(param_value)
        coord_cols = ["seat_x", "seat_y", "roi_x1", "roi_y1", "roi_x2", "roi_y2"]
        for p_idx in range(min(n_pairs, len(df) // 2)):
            i1, i2 = 2 * p_idx, 2 * p_idx + 1
            tmp = df.loc[i1, coord_cols].copy()
            df.loc[i1, coord_cols] = df.loc[i2, coord_cols].values
            df.loc[i2, coord_cols] = tmp.values
    else:
        raise ValueError(f"Unknown seating perturbation scenario_type: {scenario_type!r}")

    return df


CONTROLLED_PERTURBATION_SCENARIOS: List[Dict[str, Any]] = [
    {"scenario_id": "S0_Baseline_Original_Seats", "data_domain": "ACTUAL_OBSERVATION", "scenario_type": "baseline", "param_value": 0.0, "description": "Original calibrated 21-seat geometry (student_seats.csv)"},
    {"scenario_id": "S1_Spatial_Jitter_5px", "data_domain": "CONTROLLED_SIMULATION", "scenario_type": "spatial_jitter_px", "param_value": 5.0, "description": "Gaussian seat-anchor & ROI shift sigma = 5 px"},
    {"scenario_id": "S2_Spatial_Jitter_10px", "data_domain": "CONTROLLED_SIMULATION", "scenario_type": "spatial_jitter_px", "param_value": 10.0, "description": "Gaussian seat-anchor & ROI shift sigma = 10 px"},
    {"scenario_id": "S3_Spatial_Jitter_15px", "data_domain": "CONTROLLED_SIMULATION", "scenario_type": "spatial_jitter_px", "param_value": 15.0, "description": "Gaussian seat-anchor & ROI shift sigma = 15 px"},
    {"scenario_id": "S4_Spatial_Jitter_25px", "data_domain": "CONTROLLED_SIMULATION", "scenario_type": "spatial_jitter_px", "param_value": 25.0, "description": "Gaussian seat-anchor & ROI shift sigma = 25 px"},
    {"scenario_id": "S5_Spatial_Jitter_35px", "data_domain": "CONTROLLED_SIMULATION", "scenario_type": "spatial_jitter_px", "param_value": 35.0, "description": "Gaussian seat-anchor & ROI shift sigma = 35 px"},
    {"scenario_id": "S6_Spatial_Jitter_50px", "data_domain": "CONTROLLED_SIMULATION", "scenario_type": "spatial_jitter_px", "param_value": 50.0, "description": "Gaussian seat-anchor & ROI shift sigma = 50 px"},
    {"scenario_id": "S7_ROI_Shrink_75pct", "data_domain": "CONTROLLED_SIMULATION", "scenario_type": "roi_scale", "param_value": 0.75, "description": "Seat ROI box width & height contracted to 75%"},
    {"scenario_id": "S8_ROI_Shrink_50pct", "data_domain": "CONTROLLED_SIMULATION", "scenario_type": "roi_scale", "param_value": 0.50, "description": "Seat ROI box width & height contracted to 50%"},
    {"scenario_id": "S9_ROI_Expand_125pct", "data_domain": "CONTROLLED_SIMULATION", "scenario_type": "roi_scale", "param_value": 1.25, "description": "Seat ROI box width & height expanded to 125% (increased inter-seat overlap)"},
    {"scenario_id": "S10_Row_Pitch_Compression_15pct", "data_domain": "CONTROLLED_SIMULATION", "scenario_type": "row_pitch_compression", "param_value": 0.15, "description": "Classroom row vertical spacing compressed by 15%"},
    {"scenario_id": "S11_Col_Pitch_Compression_15pct", "data_domain": "CONTROLLED_SIMULATION", "scenario_type": "col_pitch_compression", "param_value": 0.15, "description": "Classroom column horizontal spacing compressed by 15%"},
    {"scenario_id": "S12_Adjacent_Seat_Swap_2Pairs", "data_domain": "CONTROLLED_SIMULATION", "scenario_type": "adjacent_seat_swap", "param_value": 2.0, "description": "Swaps coordinates of 2 adjacent seat pairs (S01<->S02, S03<->S04)"},
]


def run_controlled_seating_perturbation_suite(
    detections_df: pd.DataFrame,
    base_seats_df: pd.DataFrame,
    base_identity_module,
    mcdm_eval_fn=None,
    fps: float = 25.0,
    total_video_frames: int = 23471,
    seed: int = 42,
    enable_simulation: bool = False,
) -> pd.DataFrame:
    """
    Evaluates the impact of controlled seat-geometry and seating-configuration perturbations on:
    1. Seat assignment rate (track-level & frame-level)
    2. Pre-propagation seat purity
    3. Unassigned tracks and unassigned detections
    4. Identity mapping consistency (exact agreement with baseline unperturbed seat assignments)
    5. Primary assignment coverage across 23,471 frames
    6. Downstream AHP-SAW rank concordance (Spearman rho)

    NOTE: This is a controlled sensitivity simulation on fixed-seat geometry parameters, NOT a
    cross-camera or cross-classroom empirical validation.
    """
    if not enable_simulation:
        raise RuntimeError(
            "[SAFETY GUARD LOCKED] Controlled Seating Arrangement Perturbation simulation is locked. "
            "Set ENABLE_SEAT_PERTURBATION_SIMULATION = True in Cell 1 to execute."
        )
    icfg = base_identity_module.IdentityConfig.from_dict({"mode": "student_level"})
    criteria = ["Angkat Tangan", "Berdiri", "Diam", "Tidur"]
    student_ids = sorted(base_seats_df["student_id"].astype(str).unique().tolist())

    baseline_assigns = None
    baseline_saw_scores = None
    rows = []

    for sc in CONTROLLED_PERTURBATION_SCENARIOS:
        pert_seats = generate_perturbed_seat_map(
            base_seats_df=base_seats_df,
            scenario_type=sc["scenario_type"],
            param_value=float(sc["param_value"]),
            seed=seed,
        )
        assigns = base_identity_module.assign_track_to_seat(detections_df, pert_seats, icfg)
        val_rep = base_identity_module.validate_seat_assignment(
            assigns, pert_seats, icfg, total_video_frames=total_video_frames
        )
        t2s, _ = base_identity_module.merge_track_fragments(
            assigns, fps=fps, config=icfg, total_video_frames=total_video_frames, seats=pert_seats
        )
        n_total_tracks = int(len(t2s)) if not t2s.empty else int(val_rep["n_unique_track_ids"])
        n_assigned_tracks = int(t2s["student_id"].notna().sum()) if not t2s.empty else 0
        n_unassigned_tracks = n_total_tracks - n_assigned_tracks
        track_assign_rate_pct = (n_assigned_tracks / max(n_total_tracks, 1)) * 100.0

        cur_student_series = assigns["student_id"].fillna("UNASSIGNED").astype(str)
        if sc["scenario_id"] == "S0_Baseline_Original_Seats":
            baseline_assigns = cur_student_series
            mapping_consistency_pct = 100.0
        else:
            mapping_consistency_pct = float((cur_student_series == baseline_assigns).mean() * 100.0)

        rho_saw = None
        if mcdm_eval_fn is not None:
            dedup_det = base_identity_module.build_student_level_detections(assigns)
            freq_mat = (
                dedup_det.groupby(["student_id", "class_name"])["frame"]
                .nunique()
                .unstack(fill_value=0)
                .reindex(index=student_ids, columns=criteria, fill_value=0)
            )
            dur_mat = (freq_mat / float(fps)).reset_index().rename(columns={"student_id": "entity"})
            mcdm_df = mcdm_eval_fn(dur_mat).sort_values("student_id").reset_index(drop=True)
            cur_scores = mcdm_df["AHP_SAW_Score"].astype(float).to_numpy()
            if sc["scenario_id"] == "S0_Baseline_Original_Seats":
                baseline_saw_scores = cur_scores
                rho_saw = 1.0
            elif baseline_saw_scores is not None:
                rho_val, _ = stats.spearmanr(baseline_saw_scores, cur_scores)
                rho_saw = round(float(rho_val), 4)

        rows.append({
            "scenario_id": sc["scenario_id"],
            "data_domain": sc["data_domain"],
            "scenario_type": sc["scenario_type"],
            "param_value": sc["param_value"],
            "n_assigned_students": int(val_rep["n_assigned_students"]),
            "track_assignment_rate_pct": round(track_assign_rate_pct, 2),
            "n_assigned_tracks": n_assigned_tracks,
            "n_unassigned_tracks": n_unassigned_tracks,
            "primary_assignment_coverage_pct": round(float(val_rep["primary_assignment_rate"]) * 100.0, 2),
            "mean_student_frame_coverage_pct": round(float(val_rep["mean_student_frame_coverage"]) * 100.0, 2),
            "mean_seat_purity_preprop_pct": round(float(val_rep["mean_seat_purity"]) * 100.0, 2),
            "n_unassigned_detections": int(val_rep["n_unassigned_detections"]),
            "identity_mapping_consistency_pct": round(mapping_consistency_pct, 2),
            "ahp_saw_spearman_rho_vs_baseline": rho_saw,
            "description": sc["description"],
        })

    return pd.DataFrame(rows)
