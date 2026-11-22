"""
Unified Multi-Object Tracker Benchmark Adapters (Targeted Fix v2.1)
Primary Workspace: MyDrive/YOLO11_AHP_SAW_IJITCS_REVALIDATION_2026

Fixes & Methodological Disclosures Applied:
1. Fixed DataFrame indexing in `OfflineTrackerReplayer.run_on_cached_detections()` using
   `.sort_values(["frame", "confidence"], ascending=[True, False], kind="mergesort").reset_index(drop=True).copy()`
   so `grp.index` strictly matches 0-based positional indices in `out_tids`.
2. Fixed velocity EMA updates (`0.5 * old_vel + 0.5 * (new_bbox - old_bbox)`) and predicted box propagation
   across both Stage 1 (high-confidence) and Stage 2 (low-confidence) of `bytetrack`.
3. Fixed constructor and method invocation compatibility (`method="botsort_cached"` / `"botsort"`, `"sort"`, `"bytetrack"`,
   plus convenience method `run_tracker()`).
4. Connected tracker outputs directly to `src/identity.py` (`assign_track_to_seat`, `validate_seat_assignment`,
   `build_student_level_detections`), 25 FPS temporal duration aggregation, and `AHP-SAW` ranking concordances via
   `evaluate_tracker_pipeline_end_to_end()`.
5. Explicit Algorithmic Fidelity Disclosure:
   - `OfflineTrackerReplayer` (`sort` and `bytetrack` modes) implements a **Simplified Constant-Velocity EMA + Hungarian IoU
     Offline Replay Baseline** over the cached `detections.csv` (`497,848` detections) to isolate association logic without
     re-running video inference. It is NOT claimed as a full 8-dimensional Kalman Filter state-covariance implementation.
   - For full Kalman-Filter ByteTrack (`bytetrack.yaml`) or BoT-SORT (`botsort.yaml`) execution on `video_part001.mp4`,
     use `run_native_ultralytics_tracker()` (locked behind `ENABLE_FULL_VIDEO_INFERENCE = False`).
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy import stats

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("YOLO11_IJITCS_TrackerBenchmark")


def iou_batch(bboxes1: np.ndarray, bboxes2: np.ndarray) -> np.ndarray:
    """Computes pairwise IoU matrix between two sets of [x1, y1, x2, y2] boxes."""
    if len(bboxes1) == 0 or len(bboxes2) == 0:
        return np.zeros((len(bboxes1), len(bboxes2)), dtype=float)
    x1 = np.maximum(bboxes1[:, 0][:, None], bboxes2[:, 0][None, :])
    y1 = np.maximum(bboxes1[:, 1][:, None], bboxes2[:, 1][None, :])
    x2 = np.minimum(bboxes1[:, 2][:, None], bboxes2[:, 2][None, :])
    y2 = np.minimum(bboxes1[:, 3][:, None], bboxes2[:, 3][None, :])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
    area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])
    union = area1[:, None] + area2[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


class OfflineTrackerReplayer:
    """
    Replays cached frame-level detections (`detections.csv`) through:
    - 'botsort_cached' (or 'botsort'): Uses the existing Ultralytics BoT-SORT `track_id` column in `detections.csv`.
    - 'sort' (Simplified Constant-Velocity EMA + Hungarian IoU Offline Baseline):
      Single-stage matching on detections with `confidence >= 0.25`, `iou_thresh = 0.30`, `track_buffer = 30`.
    - 'bytetrack' (Simplified Two-Stage Constant-Velocity EMA + Hungarian IoU Offline Baseline):
      Stage 1 matches high-confidence detections (`>= 0.50`, `iou >= 0.30`); Stage 2 matches remaining active tracks
      against low-confidence detections (`[0.10, 0.50)`, `iou >= 0.20`) using velocity-predicted bounding boxes.
    """
    ALGORITHMIC_FIDELITY_NOTES = {
        "botsort_cached": "Exact Ultralytics BoT-SORT (with_reid=False, gmc=sparseOptFlow, Kalman filter + IoU) cached from video_part001.mp4",
        "sort": "Simplified Constant-Velocity EMA + Single-Stage Hungarian IoU Offline Replay Baseline (not 8-state Kalman Filter SORT)",
        "bytetrack": "Simplified Constant-Velocity EMA + Two-Stage High/Low Confidence Hungarian IoU Offline Replay Baseline (not 8-state Kalman Filter ByteTrack)",
    }

    def __init__(
        self,
        method: str = "botsort_cached",
        track_high_thresh: float = 0.50,
        track_low_thresh: float = 0.10,
        match_iou_thresh: float = 0.30,
        track_buffer: int = 30,
    ):
        norm_method = str(method).strip().lower()
        if norm_method == "botsort":
            norm_method = "botsort_cached"
        if norm_method not in ("botsort_cached", "sort", "bytetrack"):
            raise ValueError(f"Unsupported tracker method: {method!r}. Expected 'botsort_cached', 'sort', or 'bytetrack'.")
        self.method = norm_method
        self.track_high_thresh = float(track_high_thresh)
        self.track_low_thresh = float(track_low_thresh)
        self.match_iou_thresh = float(match_iou_thresh)
        self.track_buffer = int(track_buffer)

    def run_on_cached_detections(self, detections_df: pd.DataFrame, enable_experiment: bool = False) -> pd.DataFrame:
        if not enable_experiment:
            raise RuntimeError(
                f"[SAFETY GUARD LOCKED] Offline tracker replay ({self.method}) over 497,848 detections is locked. "
                "Set ENABLE_TRACKER_REPLAY = True in Cell 1 to execute."
            )
        if self.method == "botsort_cached":
            return detections_df.sort_values(["frame", "track_id"], kind="mergesort").reset_index(drop=True).copy()

        # Critical Fix (BUG-04): reset_index(drop=True) after sorting ensures grp.index is 0-based positional index
        df = detections_df.sort_values(["frame", "confidence"], ascending=[True, False], kind="mergesort").reset_index(drop=True).copy()
        active_tracks: Dict[int, Dict[str, Any]] = {}  # tid -> {"bbox": np.ndarray(4), "vel": np.ndarray(4), "last_frame": int}
        next_tid = 1
        out_tids = np.full(len(df), -1, dtype=int)

        for frame_idx, grp in df.groupby("frame", sort=True):
            row_indices = grp.index.to_numpy()
            boxes = grp[["x1", "y1", "x2", "y2"]].to_numpy(dtype=np.float64)
            confs = grp["confidence"].to_numpy(dtype=np.float64)

            # Prune expired tracks exceeding track_buffer frames
            expired = [tid for tid, st in active_tracks.items() if (frame_idx - st["last_frame"]) > self.track_buffer]
            for tid in expired:
                del active_tracks[tid]

            tids_list = list(active_tracks.keys())
            pred_boxes = (
                np.array([active_tracks[tid]["bbox"] + active_tracks[tid]["vel"] for tid in tids_list], dtype=np.float64)
                if tids_list else np.empty((0, 4), dtype=np.float64)
            )

            if self.method == "sort":
                valid_mask = confs >= 0.25
                v_idx = np.where(valid_mask)[0]
                if len(tids_list) > 0 and len(v_idx) > 0:
                    iou_mat = iou_batch(pred_boxes, boxes[v_idx])
                    r_ind, c_ind = linear_sum_assignment(-iou_mat)
                    matched_det = set()
                    for r, c in zip(r_ind, c_ind):
                        if iou_mat[r, c] >= self.match_iou_thresh:
                            tid = tids_list[r]
                            det_pos = v_idx[c]
                            out_tids[row_indices[det_pos]] = tid
                            active_tracks[tid]["vel"] = 0.5 * active_tracks[tid]["vel"] + 0.5 * (boxes[det_pos] - active_tracks[tid]["bbox"])
                            active_tracks[tid]["bbox"] = boxes[det_pos]
                            active_tracks[tid]["last_frame"] = int(frame_idx)
                            matched_det.add(det_pos)
                    for det_pos in v_idx:
                        if det_pos not in matched_det:
                            out_tids[row_indices[det_pos]] = next_tid
                            active_tracks[next_tid] = {"bbox": boxes[det_pos], "vel": np.zeros(4, dtype=np.float64), "last_frame": int(frame_idx)}
                            next_tid += 1
                else:
                    for det_pos in v_idx:
                        out_tids[row_indices[det_pos]] = next_tid
                        active_tracks[next_tid] = {"bbox": boxes[det_pos], "vel": np.zeros(4, dtype=np.float64), "last_frame": int(frame_idx)}
                        next_tid += 1

            elif self.method == "bytetrack":
                high_idx = np.where(confs >= self.track_high_thresh)[0]
                low_idx = np.where((confs >= self.track_low_thresh) & (confs < self.track_high_thresh))[0]
                unmatched_tracks = set(range(len(tids_list)))
                matched_high = set()

                # Stage 1: match active tracks with high-confidence detections
                if len(tids_list) > 0 and len(high_idx) > 0:
                    iou_mat = iou_batch(pred_boxes, boxes[high_idx])
                    r_ind, c_ind = linear_sum_assignment(-iou_mat)
                    for r, c in zip(r_ind, c_ind):
                        if iou_mat[r, c] >= self.match_iou_thresh:
                            tid = tids_list[r]
                            det_pos = high_idx[c]
                            out_tids[row_indices[det_pos]] = tid
                            # Critical Fix (BUG-05): update velocity EMA in ByteTrack Stage 1
                            active_tracks[tid]["vel"] = 0.5 * active_tracks[tid]["vel"] + 0.5 * (boxes[det_pos] - active_tracks[tid]["bbox"])
                            active_tracks[tid]["bbox"] = boxes[det_pos]
                            active_tracks[tid]["last_frame"] = int(frame_idx)
                            unmatched_tracks.discard(r)
                            matched_high.add(det_pos)

                # Stage 2: match remaining active tracks with low-confidence detections using velocity-predicted boxes
                rem_tids = [tids_list[r] for r in sorted(unmatched_tracks)]
                if len(rem_tids) > 0 and len(low_idx) > 0:
                    rem_pred_boxes = np.array(
                        [active_tracks[tid]["bbox"] + active_tracks[tid]["vel"] for tid in rem_tids], dtype=np.float64
                    )
                    iou_low = iou_batch(rem_pred_boxes, boxes[low_idx])
                    r2, c2 = linear_sum_assignment(-iou_low)
                    for r, c in zip(r2, c2):
                        if iou_low[r, c] >= 0.20:
                            tid = rem_tids[r]
                            det_pos = low_idx[c]
                            out_tids[row_indices[det_pos]] = tid
                            # Critical Fix (BUG-05): update velocity EMA in ByteTrack Stage 2
                            active_tracks[tid]["vel"] = 0.5 * active_tracks[tid]["vel"] + 0.5 * (boxes[det_pos] - active_tracks[tid]["bbox"])
                            active_tracks[tid]["bbox"] = boxes[det_pos]
                            active_tracks[tid]["last_frame"] = int(frame_idx)

                # Spawn new tracks from unmatched high-confidence detections
                for det_pos in high_idx:
                    if det_pos not in matched_high:
                        out_tids[row_indices[det_pos]] = next_tid
                        active_tracks[next_tid] = {"bbox": boxes[det_pos], "vel": np.zeros(4, dtype=np.float64), "last_frame": int(frame_idx)}
                        next_tid += 1

        df["track_id"] = out_tids
        df["tracker_implementation"] = self.ALGORITHMIC_FIDELITY_NOTES[self.method]
        return df[df["track_id"] > 0].reset_index(drop=True)

    def run_tracker(self, detections_df: pd.DataFrame, enable_execution: bool = False) -> pd.DataFrame:
        """Backward-compatible alias for `run_on_cached_detections()`."""
        return self.run_on_cached_detections(detections_df=detections_df, enable_experiment=enable_execution)


def run_native_ultralytics_tracker(
    video_path: Path,
    checkpoint_path: Path,
    output_csv: Path,
    tracker_yaml: str = "bytetrack.yaml",
    conf: float = 0.25,
    iou: float = 0.50,
    imgsz: int = 640,
    device: str = "0",
    enable_video_inference: bool = False,
) -> Path:
    """
    Runs full-video native Ultralytics tracking (`botsort.yaml` or `bytetrack.yaml` with full 8-state Kalman filter)
    directly on `video_part001.mp4`. Strictly locked behind `enable_video_inference=False`.
    """
    if not enable_video_inference:
        raise RuntimeError(
            f"[SAFETY GUARD LOCKED] Native Ultralytics {tracker_yaml} full-video inference is locked. "
            "Set ENABLE_FULL_VIDEO_INFERENCE = True in Cell 1 to execute on GPU."
        )
    from src.detect_track import run_detection_tracking
    run_detection_tracking(
        model_path=str(checkpoint_path),
        video_path=str(video_path),
        output_csv=str(output_csv),
        tracker=tracker_yaml,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device,
        save_video=False,
        with_reid=False,
    )
    return Path(output_csv)


def evaluate_tracker_pipeline_end_to_end(
    detections_df: pd.DataFrame,
    seats_df: pd.DataFrame,
    base_identity_module,
    mcdm_eval_fn,
    output_dir: Path,
    fps: float = 25.0,
    total_video_frames: int = 23471,
    enable_experiment: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Critical Fix (BUG-06): Connects `OfflineTrackerReplayer` outputs (`botsort_cached`, `sort`, `bytetrack`)
    directly into Fixed-Seat Identity Resolution (`assign_track_to_seat`, `validate_seat_assignment`,
    `build_student_level_detections`), 25 FPS duration matrix aggregation, and `AHP-SAW` ranking comparison.
    """
    if not enable_experiment:
        raise RuntimeError(
            "[SAFETY GUARD LOCKED] End-to-end tracker-to-identity-to-AHP-SAW replay is locked. "
            "Set ENABLE_TRACKER_REPLAY = True in Cell 1 to execute."
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    criteria = ["Angkat Tangan", "Berdiri", "Diam", "Tidur"]
    student_ids = sorted(seats_df["student_id"].astype(str).unique().tolist())

    summary_rows = []
    rankings_by_tracker: Dict[str, pd.DataFrame] = {}
    baseline_saw_scores: Optional[np.ndarray] = None
    baseline_saw_ranks: Optional[np.ndarray] = None

    for method_name in ["botsort_cached", "sort", "bytetrack"]:
        replayer = OfflineTrackerReplayer(method=method_name)
        df_tr = replayer.run_on_cached_detections(detections_df, enable_experiment=True)
        df_tr.to_csv(output_dir / f"tracks_{method_name}.csv", index=False)

        icfg = base_identity_module.IdentityConfig.from_dict({"mode": "student_level"})
        assigns = base_identity_module.assign_track_to_seat(df_tr, seats_df, icfg)
        val_rep = base_identity_module.validate_seat_assignment(
            assigns, seats_df, icfg, total_video_frames=total_video_frames
        )
        dedup_det = base_identity_module.build_student_level_detections(assigns)

        freq_mat = (
            dedup_det.groupby(["student_id", "class_name"])["frame"]
            .nunique()
            .unstack(fill_value=0)
            .reindex(index=student_ids, columns=criteria, fill_value=0)
        )
        dur_mat = (freq_mat / float(fps)).reset_index().rename(columns={"student_id": "entity"})
        dur_mat.to_csv(output_dir / f"duration_matrix_{method_name}.csv", index=False)

        mcdm_df = mcdm_eval_fn(dur_mat).sort_values("student_id").reset_index(drop=True)
        rankings_by_tracker[method_name] = mcdm_df
        cur_scores = mcdm_df["AHP_SAW_Score"].astype(float).to_numpy()
        cur_ranks = mcdm_df["AHP_SAW_Rank"].astype(int).to_numpy()

        if method_name == "botsort_cached":
            baseline_saw_scores = cur_scores
            baseline_saw_ranks = cur_ranks
            rho, p_rho, tau, p_tau, mean_shift, top5_ov = 1.0, 0.0, 1.0, 0.0, 0.0, 1.0
        else:
            rho, p_rho = stats.spearmanr(baseline_saw_scores, cur_scores)
            tau, p_tau = stats.kendalltau(baseline_saw_scores, cur_scores)
            mean_shift = float(np.mean(np.abs(baseline_saw_ranks - cur_ranks)))
            top5_base = set(mcdm_df["student_id"][baseline_saw_ranks <= 5])
            top5_cur = set(mcdm_df["student_id"][cur_ranks <= 5])
            top5_ov = len(top5_base & top5_cur) / 5.0

        summary_rows.append({
            "tracker_method": method_name,
            "algorithmic_scope": OfflineTrackerReplayer.ALGORITHMIC_FIDELITY_NOTES[method_name],
            "n_tracked_detections": int(len(df_tr)),
            "n_unique_tracks": int(val_rep["n_unique_track_ids"]),
            "n_primary_assigned": int(val_rep["n_primary_assigned"]),
            "primary_assignment_rate_pct": round(float(val_rep["primary_assignment_rate"]) * 100.0, 2),
            "n_assigned_students": int(val_rep["n_assigned_students"]),
            "mean_preprop_seat_purity_pct": round(float(val_rep["mean_seat_purity"]) * 100.0, 2),
            "mean_student_frame_coverage_pct": round(float(val_rep["mean_student_frame_coverage"]) * 100.0, 2),
            "n_long_gap_fragments": int(val_rep["n_long_gap_fragments"]),
            "ahp_saw_spearman_rho_vs_botsort": round(float(rho), 4),
            "ahp_saw_spearman_p_value": float(p_rho),
            "ahp_saw_kendall_tau_vs_botsort": round(float(tau), 4),
            "ahp_saw_mean_rank_shift": round(float(mean_shift), 2),
            "ahp_saw_top5_overlap_ratio": round(float(top5_ov), 2),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "tracker_baseline_end_to_end_comparison.csv", index=False)
    return summary_df, rankings_by_tracker
