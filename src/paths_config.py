"""
Centralized path configuration and dynamic path resolver for the research project:
YOLO11-Based Classroom Activity Assessment with Fixed-Seat Identity Resolution
and AHP-SAW Decision Support (IJITCS -- PAPER024568-1).

All Colab notebooks, CLI scripts (run_pipeline.py, run_experiments.py), and
src/* modules share this single canonical source of truth for directory paths.

Targeted Dataset Organization (Multi-Detector Separation):
  1. YOLO11 (Preserved Untouched):
     - 03_DATASET_AUDIT/YOLO11_DATASET_UNZIPPED/
     - 03_DATASET_AUDIT/_BACKUP_YOLO11_DATASET_UNZIPPED_PREV/
     - 03_DATASET_AUDIT/bima_lppm2_.v2i.yolov11.zip (1,373 train / 393 valid / 196 test)
  2. YOLOv8 (Separate YOLO TXT Dataset, 1,373 / 393 / 196 split parity):
     - 03_DATASET_AUDIT/DETECTOR_DATASETS/YOLOv8/
     - 03_DATASET_AUDIT/DETECTOR_DATASETS/YOLOv8/data.yaml
  3. RF-DETR (Separate COCO JSON Dataset, 1,373 / 393 / 196 split parity):
     - 03_DATASET_AUDIT/DETECTOR_DATASETS/RF_DETR/
     - 03_DATASET_AUDIT/DETECTOR_DATASETS/RF_DETR/{train,valid,test}/_annotations.coco.json
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

COLAB_MASTER_ROOT = Path("/content/drive/MyDrive/YOLO11_AHP_SAW_IJITCS_REVALIDATION_2026")
_THIS_FILE = Path(__file__).resolve()
_LOCAL_REPO_ROOT = _THIS_FILE.parent.parent


def get_master_root() -> Path:
    """Return the active MASTER_ROOT (Colab Drive path if mounted, else local project root)."""
    env_root = os.environ.get("YOLO11_MASTER_ROOT")
    if env_root:
        return Path(env_root)
    if COLAB_MASTER_ROOT.exists():
        return COLAB_MASTER_ROOT
    if _LOCAL_REPO_ROOT.parent.name == "05_GITHUB_CODE_AND_MODULES":
        return _LOCAL_REPO_ROOT.parent.parent
    return COLAB_MASTER_ROOT


MASTER_ROOT: Path = get_master_root()

DATASET_AUDIT_DIR: Path = MASTER_ROOT / "03_DATASET_AUDIT"
# Preserved YOLO11 dataset paths (DO NOT MOVE, DELETE, OR OVERWRITE)
DATASET_ROOT: Path = DATASET_AUDIT_DIR / "YOLO11_DATASET_UNZIPPED"
STAGING_DIR: Path = DATASET_AUDIT_DIR / "_STAGING_ROBOFLOW_YOLOV11"
BACKUP_DATASET_DIR: Path = DATASET_AUDIT_DIR / "_BACKUP_YOLO11_DATASET_UNZIPPED_PREV"
ZIP_ARCHIVE_PATH: Path = DATASET_AUDIT_DIR / "bima_lppm2_.v2i.yolov11.zip"

DATA_YAML_PATH: Path = DATASET_ROOT / "data_colab.yaml"
DATA_ROBOFLOW_ORIG_YAML_PATH: Path = DATASET_ROOT / "data_roboflow_original.yaml"

# Separate Detector Dataset Directories (Historical 1,373 / 393 / 196 Split Parity)
DETECTOR_DATASETS_DIR: Path = DATASET_AUDIT_DIR / "DETECTOR_DATASETS"
YOLOV8_DATASET_DIR: Path = DETECTOR_DATASETS_DIR / "YOLOv8"
YOLOV8_DATA_YAML_PATH: Path = YOLOV8_DATASET_DIR / "data.yaml"
RFDETR_DATASET_DIR: Path = DETECTOR_DATASETS_DIR / "RF_DETR"
RFDETR_MANIFEST_PATH: Path = RFDETR_DATASET_DIR / "rf_detr_dataset_manifest.json"

REPO_ROOT: Path = (
    MASTER_ROOT / "05_GITHUB_CODE_AND_MODULES" / "YOLO11_Repository"
    if (MASTER_ROOT / "05_GITHUB_CODE_AND_MODULES" / "YOLO11_Repository").exists()
    or not _LOCAL_REPO_ROOT.exists()
    else _LOCAL_REPO_ROOT
)
MODULES_DIR: Path = MASTER_ROOT / "05_GITHUB_CODE_AND_MODULES" / "Reconstructed_Research_Modules"

VIDEO_PATH: Path = MASTER_ROOT / "02_ORIGINAL_RESEARCH_ARCHIVE" / "Original_Videos" / "video_part001.mp4"
CHECKPOINT_DIR: Path = MASTER_ROOT / "04_MODEL_AND_CONFIGURATION" / "Checkpoints"
PRIMARY_CHECKPOINT_PATH: Path = CHECKPOINT_DIR / "best.pt"
OUTPUT_DIR: Path = MASTER_ROOT / "09_REVALIDATED_EXPERIMENT_OUTPUTS"
PIPELINE_OUTPUT_DIR: Path = OUTPUT_DIR / "Pipeline_Outputs"
DETECTOR_BENCHMARK_OUTPUT_DIR: Path = OUTPUT_DIR / "detector_benchmarks"
RFDETR_SMALL_BENCHMARK_DIR: Path = DETECTOR_BENCHMARK_OUTPUT_DIR / "rfdetr_small"
RFDETR_LARGE_BENCHMARK_DIR: Path = DETECTOR_BENCHMARK_OUTPUT_DIR / "rfdetr_large"
YOLOV8_BENCHMARK_DIR: Path = DETECTOR_BENCHMARK_OUTPUT_DIR / "yolov8m"
YOLO11_BENCHMARK_DIR: Path = DETECTOR_BENCHMARK_OUTPUT_DIR / "yolo11m"
MULTI_DETECTOR_PIPELINE_DIR: Path = OUTPUT_DIR / "multi_detector_pipeline"

CANONICAL_CLASSES: dict[int, str] = {
    0: "Angkat Tangan",
    1: "Berdiri",
    2: "Diam",
    3: "Tidur",
}


def resolve_project_path(raw_path: Optional[str | Path], kind: str = "generic") -> str:
    """
    Resolve a configured path against canonical MASTER_ROOT and REPO_ROOT locations.

    Parameters
    ----------
    raw_path : str | Path | None
        Path string from config.yaml or caller argument.
    kind : {"generic", "seats_csv", "data_yaml", "yolov8_data_yaml", "rfdetr_dataset_dir", "model_path", "video_path", "output_dir"}
        Semantic type of path for fallback resolution when a relative or legacy
        Windows path (e.g. E:/LanaNumpang/...) is encountered.
    """
    raw_str = str(raw_path).strip() if raw_path is not None else ""

    if raw_str and not raw_str.startswith(("E:/", "E:\\", "C:/", "C:\\")):
        p = Path(raw_str)
        if p.exists():
            if kind == "data_yaml" and p.name == "data.yaml" and p.parent == DATASET_ROOT:
                colab_sibling = p.parent / "data_colab.yaml"
                if colab_sibling.exists():
                    return str(colab_sibling)
            return str(p)
        for base in (_LOCAL_REPO_ROOT, REPO_ROOT, MASTER_ROOT):
            cand = base / p
            if cand.exists():
                return str(cand)

    if kind == "seats_csv":
        candidates = [
            _LOCAL_REPO_ROOT / "data" / "student_seats.csv",
            REPO_ROOT / "data" / "student_seats.csv",
            MASTER_ROOT / "05_GITHUB_CODE_AND_MODULES" / "YOLO11_Repository" / "data" / "student_seats.csv",
            MASTER_ROOT / "06_TRACKING_AND_IDENTITY_VALIDATION" / "student_seats.csv",
            MASTER_ROOT / "02_ORIGINAL_RESEARCH_ARCHIVE" / "Supporting_Configs" / "student_seats.csv",
            MASTER_ROOT / "04_MODEL_AND_CONFIGURATION" / "Seat_Map" / "student_seats.csv",
        ]
        for c in candidates:
            if c.exists():
                return str(c)
        return str(REPO_ROOT / "data" / "student_seats.csv")

    if kind == "data_yaml":
        candidates = [
            DATA_YAML_PATH,
            DATASET_ROOT / "data.yaml",
            _LOCAL_REPO_ROOT / "data_colab.yaml",
        ]
        for c in candidates:
            if c.exists():
                return str(c)
        return str(DATA_YAML_PATH)

    if kind == "yolov8_data_yaml":
        return str(YOLOV8_DATA_YAML_PATH)

    if kind == "rfdetr_dataset_dir":
        return str(RFDETR_DATASET_DIR)

    if kind == "model_path":
        if raw_str:
            fname = Path(raw_str.replace("\\", "/")).name
            for folder in (
                CHECKPOINT_DIR,
                MASTER_ROOT / "02_ORIGINAL_RESEARCH_ARCHIVE" / "Historical_Checkpoints_and_Logs",
                _LOCAL_REPO_ROOT / "weights",
            ):
                cand = folder / fname
                if cand.exists():
                    return str(cand)
        return str(PRIMARY_CHECKPOINT_PATH)

    if kind == "video_path":
        candidates = [
            VIDEO_PATH,
            MASTER_ROOT / "02_ORIGINAL_RESEARCH_ARCHIVE" / "Original_Videos" / "video_part001.mp4",
            _LOCAL_REPO_ROOT / "data" / "video_part001.mp4",
        ]
        for c in candidates:
            if c.exists():
                return str(c)
        return str(VIDEO_PATH)

    if kind == "output_dir":
        if not raw_str or raw_str == "output":
            if COLAB_MASTER_ROOT.exists():
                return str(PIPELINE_OUTPUT_DIR)
            return str(_LOCAL_REPO_ROOT / "output")
        return raw_str

    return raw_str


def get_detector_pipeline_paths(detector_key: str, master_root: Optional[Path] = None) -> dict[str, Path]:
    """
    Resolve input/output pipeline CSV paths (detections.csv, track_summary.csv,
    track_to_student_summary.csv, student_identity_audit.csv, identity_assignments.csv,
    student_activity_matrix_duration.csv) for any detector model:
      - "yolo11m": canonical 06_TRACKING_AND_IDENTITY_VALIDATION & 07_AHP_SAW_AND_STATISTICAL_VALIDATION
      - "yolov8m": 09_REVALIDATED_EXPERIMENT_OUTPUTS/multi_detector_pipeline/yolov8m
      - "rfdetr_small" (or "S"): 09_REVALIDATED_EXPERIMENT_OUTPUTS/multi_detector_pipeline/rfdetr_small
      - "rfdetr_large" (or "L"): 09_REVALIDATED_EXPERIMENT_OUTPUTS/multi_detector_pipeline/rfdetr_large
    """
    root = Path(master_root) if master_root is not None else MASTER_ROOT
    norm = str(detector_key).strip().lower()
    if norm in ("s", "small", "rfdetr_s", "rf-detr-small", "rfdetr_small"):
        norm = "rfdetr_small"
    elif norm in ("l", "large", "rfdetr_l", "rf-detr-large", "rfdetr_large"):
        norm = "rfdetr_large"
    elif norm in ("yolov8", "yolov8m", "v8"):
        norm = "yolov8m"
    else:
        norm = "yolo11m"

    out_root = root / "09_REVALIDATED_EXPERIMENT_OUTPUTS"
    ckpt_dir = root / "04_MODEL_AND_CONFIGURATION" / "Checkpoints"

    if norm == "yolo11m":
        trk_dir = root / "06_TRACKING_AND_IDENTITY_VALIDATION"
        ahp_dir = root / "07_AHP_SAW_AND_STATISTICAL_VALIDATION"
        return {
            "detector_key": "yolo11m",
            "pipeline_dir": out_root,
            "checkpoint_path": ckpt_dir / "yolo11m_baseline_best.pt",
            "detections_csv": trk_dir / "detections.csv",
            "track_summary_csv": trk_dir / "track_summary.csv",
            "track_to_student_csv": trk_dir / "track_to_student_summary.csv",
            "student_audit_csv": trk_dir / "student_identity_audit.csv",
            "identity_assignments_csv": trk_dir / "identity_assignments.csv",
            "duration_matrix_csv": ahp_dir / "student_activity_matrix_duration.csv",
        }

    pipe_dir = out_root / "multi_detector_pipeline" / norm
    if norm == "yolov8m":
        ckpt_cands = [
            ckpt_dir / "yolov8m_baseline_best.pt",
            out_root / "detector_benchmarks" / "yolov8m" / "yolov8m_baseline_seed42" / "weights" / "best.pt",
            ckpt_dir / "yolov8m.pt",
        ]
    elif norm == "rfdetr_small":
        ckpt_cands = [
            ckpt_dir / "rfdetr_small_best.pth",
            out_root / "detector_benchmarks" / "rfdetr_small" / "checkpoint_best_total.pth",
            out_root / "detector_benchmarks" / "rfdetr_small" / "checkpoint_best_ema.pth",
            ckpt_dir / "rf-detr-small.pth",
        ]
    else:
        ckpt_cands = [
            ckpt_dir / "rfdetr_large_best.pth",
            out_root / "detector_benchmarks" / "rfdetr_large" / "checkpoint_best_total.pth",
            out_root / "detector_benchmarks" / "rfdetr_large" / "checkpoint_best_ema.pth",
            ckpt_dir / "rf-detr-large.pth",
        ]
    resolved_ckpt = next((c for c in ckpt_cands if c.exists()), ckpt_cands[0])
    return {
        "detector_key": norm,
        "pipeline_dir": pipe_dir,
        "checkpoint_path": resolved_ckpt,
        "detections_csv": pipe_dir / "detections.csv",
        "track_summary_csv": pipe_dir / "track_summary.csv",
        "track_to_student_csv": pipe_dir / "track_to_student_summary.csv",
        "student_audit_csv": pipe_dir / "student_identity_audit.csv",
        "identity_assignments_csv": pipe_dir / "identity_assignments.csv",
        "duration_matrix_csv": pipe_dir / "student_activity_matrix_duration.csv",
    }
