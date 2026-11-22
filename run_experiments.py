"""
run_experiments.py -- orkestrator eksperimen berat (terpisah dari pipeline utama).

`run_pipeline.py` sengaja dijaga ringan dan reproducible. Eksperimen yang mahal
secara komputasi dijalankan dari sini:

    python run_experiments.py --config config.yaml
    python run_experiments.py --config config.yaml --only ablation,montecarlo

Eksperimen tersedia: temporal, ablation, montecarlo, wp, detector, errorprop.

Seluruh eksperimen memakai detections.csv dan asosiasi identitas yang SUDAH ADA
(hasil run_pipeline.py). Detector/tracker TIDAK dijalankan ulang.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

_REPO_DIR = Path(__file__).resolve().parent
if str(_REPO_DIR) not in sys.path:
    sys.path.insert(0, str(_REPO_DIR))

from run_pipeline import _cfg
from src.paths_config import resolve_project_path
from src.aggregate import build_both_representations
from src.ahp import ahp_weights
from src.identity import (IdentityConfig, assign_track_to_seat,
                          build_student_level_detections, load_seat_map,
                          merge_track_fragments, validate_seat_assignment)

ALL_EXPERIMENTS = ["temporal", "ablation", "montecarlo", "wp", "detector", "errorprop"]


def main() -> int:
    ap = argparse.ArgumentParser(description="Eksperimen tambahan YOLO11 classroom")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--only", help=f"subset dipisah koma dari: {','.join(ALL_EXPERIMENTS)}")
    a = ap.parse_args()

    cfg_file = Path(a.config) if Path(a.config).exists() else (_REPO_DIR / a.config)
    with open(cfg_file, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("source", {})
    cfg["source"]["model_path"] = resolve_project_path(_cfg(cfg, "source.model_path"), kind="model_path")
    cfg["source"]["video_path"] = resolve_project_path(_cfg(cfg, "source.video_path"), kind="video_path")
    cfg["source"]["data_yaml"] = resolve_project_path(_cfg(cfg, "source.data_yaml"), kind="data_yaml")
    cfg.setdefault("identity", {})
    cfg["identity"]["seats_csv"] = resolve_project_path(_cfg(cfg, "identity.seats_csv"), kind="seats_csv")
    out = Path(resolve_project_path(_cfg(cfg, "output_dir", "output"), kind="output_dir"))
    out.mkdir(parents=True, exist_ok=True)

    exp_key_map = {
        "temporal": "temporal_comparison",
        "ablation": "ablation",
        "montecarlo": "monte_carlo",
        "wp": "saw_vs_wp",
        "detector": "detector_evaluation",
        "errorprop": "error_propagation",
    }
    selected = (
        [s.strip() for s in a.only.split(",")]
        if a.only
        else [
            e
            for e in ALL_EXPERIMENTS
            if _cfg(
                cfg,
                f"experiments.{exp_key_map[e]}",
                True if e == "montecarlo" else False,
            )
            or e == "montecarlo"
        ]
    )
    unknown = set(selected) - set(ALL_EXPERIMENTS)
    if unknown:
        print(f"ERROR: eksperimen tidak dikenal: {sorted(unknown)}", file=sys.stderr)
        return 2

    det_csv = out / "detections.csv"
    if not det_csv.exists():
        print(f"ERROR:\n{det_csv} tidak ditemukan. Jalankan run_pipeline.py dahulu.",
              file=sys.stderr)
        return 2
    detections = pd.read_csv(det_csv)

    criteria = _cfg(cfg, "criteria", [])
    orientation = _cfg(cfg, "orientation", {})
    fps_cfg = _cfg(cfg, "temporal.fps", 25.0)
    if isinstance(fps_cfg, str) and fps_cfg.strip().lower() == "auto":
        from src.provenance import video_info
        vi = video_info(_cfg(cfg, "source.video_path"))
        fps = float(vi["video_fps"]) if vi.get("video_fps") not in (None, "NOT_AVAILABLE") else 25.0
    else:
        fps = float(fps_cfg)
    mtf = int(_cfg(cfg, "temporal.min_track_frames", 30))
    zeps = float(_cfg(cfg, "saw.zero_epsilon", 1e-6))
    zstr = _cfg(cfg, "saw.zero_strategy", "epsilon")

    ahp = ahp_weights(np.array(_cfg(cfg, "ahp.matrix")), criteria,
                      source=_cfg(cfg, "ahp.source", "predefined_model"),
                      expert_count=int(_cfg(cfg, "ahp.expert_count", 0)))
    weights = ahp["weights"]

    # --- rekonstruksi identitas ------------------------------------------
    id_raw = dict(_cfg(cfg, "identity", {}) or {})
    seats_csv = id_raw.pop("seats_csv", None)
    id_raw.pop("mode", None)
    icfg = IdentityConfig.from_dict({**id_raw, "mode": "student_level"})
    seats = load_seat_map(seats_csv, icfg.expected_students)
    assigns = assign_track_to_seat(detections, seats, icfg)
    report = validate_seat_assignment(assigns, seats, icfg)
    if not report["passed"]:
        print("ERROR:\nStudent-level experiments blocked because identity "
              "validation has not passed.", file=sys.stderr)
        return 1
    student_det = build_student_level_detections(assigns)
    total_frames = int(detections["frame"].max()) + 1 if not detections.empty else 0
    _, student_audit = merge_track_fragments(assigns, fps, icfg, total_video_frames=total_frames, seats=seats)

    matrices = build_both_representations(
        student_det, fps=fps, entity_column="student_id", classes=criteria,
        min_track_frames=mtf,
        strict_classes=bool(_cfg(cfg, "temporal.strict_classes", True)))
    main_rep = _cfg(cfg, "temporal.representation", "duration")
    matrix = matrices[main_rep]

    ran: list = []

    if "temporal" in selected:
        from src.temporal_analysis import (compare_representations,
                                           per_entity_rank_table)
        compare_representations(matrices, weights, orientation, zeps, zstr) \
            .to_csv(out / "temporal_comparison.csv", index=False)
        per_entity_rank_table(matrices, weights, orientation, zeps, zstr) \
            .to_csv(out / "temporal_comparison_per_student.csv", index=False)
        ran.append("temporal")

    if "ablation" in selected:
        from src.ablation import run_ablation
        run_ablation(detections, fps, criteria, mtf, student_detections=student_det,
                     identity_report=report, representation=main_rep) \
            .to_csv(out / "ablation_results.csv", index=False)
        ran.append("ablation")

    if "montecarlo" in selected:
        from src.monte_carlo import monte_carlo_sensitivity
        monte_carlo_sensitivity(
            matrix, weights, orientation,
            iterations=int(_cfg(cfg, "monte_carlo.iterations", 5000)),
            seed=int(_cfg(cfg, "monte_carlo.seed", 42)),
            method=_cfg(cfg, "monte_carlo.method", "dirichlet_around_base"),
            concentration=float(_cfg(cfg, "monte_carlo.concentration", 50.0)),
            zero_epsilon=zeps, zero_strategy=zstr) \
            .to_csv(out / "sensitivity_montecarlo.csv", index=False)
        ran.append("montecarlo")

    if "wp" in selected:
        from src.wp import compare_saw_wp
        compare_saw_wp(matrix, weights, orientation, zeps, zstr) \
            .to_csv(out / "ahp_saw_vs_wp.csv", index=False)
        ran.append("wp")

    if "errorprop" in selected:
        from src.error_propagation import build_propagation_table, qualitative_notes
        from src.saw import saw_score
        saw = saw_score(matrix, weights, orientation, zeps, zstr)
        t = build_propagation_table(student_audit, matrix, saw)
        if not t.empty:
            pd.concat([t, qualitative_notes()], ignore_index=True) \
              .to_csv(out / "error_propagation.csv", index=False)
        ran.append("errorprop")

    if "detector" in selected:
        from src.evaluate_detector import (DetectorEvalUnavailable,
                                           evaluate_detector)
        try:
            evaluate_detector(
                model_path=_cfg(cfg, "source.model_path"),
                data_yaml=_cfg(cfg, "source.data_yaml"),
                split=_cfg(cfg, "detector_evaluation.split", "test"),
                imgsz=_cfg(cfg, "detection.imgsz", 640),
                device=str(_cfg(cfg, "detection.device", "0")),
                batch=int(_cfg(cfg, "detector_evaluation.batch", 16)),
                output_json=str(out / "detection_metrics.json"),
                output_csv=str(out / "detection_metrics.csv"))
            ran.append("detector")
        except DetectorEvalUnavailable as e:
            print(e)

    print(f"\nSelesai. Eksperimen dijalankan: {ran or 'tidak ada'}")
    print(f"Keluaran -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
