"""
Aligned Detector Benchmark, Multi-Detector Video Pipeline & Checkpoint Evaluation Adapter Module (v2.3)
Primary Workspace: MyDrive/YOLO11_AHP_SAW_IJITCS_REVALIDATION_2026
Target Journal: IJITCS (Submission ID: PAPER024568-1)

Key Capabilities:
1. Auto-installs `rfdetr[train,loggers]` and `pytorch_lightning` so RF-DETR training never fails with
   `ModuleNotFoundError: No module named 'pytorch_lightning'`.
2. Supports both `RF-DETR-Small` (`RFDETR_MODEL_SIZE = "S"`) and `RF-DETR-Large` (`RFDETR_MODEL_SIZE = "L"`,
   or `"BOTH"`) with strictly separated output directories:
   - `09_REVALIDATED_EXPERIMENT_OUTPUTS/detector_benchmarks/rfdetr_small/`
   - `09_REVALIDATED_EXPERIMENT_OUTPUTS/detector_benchmarks/rfdetr_large/`
   - `09_REVALIDATED_EXPERIMENT_OUTPUTS/detector_benchmarks/yolov8m/`
   - `09_REVALIDATED_EXPERIMENT_OUTPUTS/detector_benchmarks/yolo11m/`
3. Provides `run_multi_detector_video_pipeline()` so Notebook 04 can generate `detections.csv`,
   `track_summary.csv`, `track_to_student_summary.csv`, `student_identity_audit.csv`,
   `identity_assignments.csv`, and `student_activity_matrix_duration.csv` for `yolov8m`,
   `rfdetr_small`, and `rfdetr_large` inside `09_REVALIDATED_EXPERIMENT_OUTPUTS/multi_detector_pipeline/<detector>/`,
   seamlessly feeding Notebook 08 (MCDM + Identity Ablation + Controlled Seating Perturbation)
   and Notebook 10 (Within-Session 3x5-Minute Temporal Robustness).
"""
from __future__ import annotations
import os
import sys
import json
import time
import shutil
import hashlib
import logging
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd


def _get_ultralytics_cls(cls_name: str):
    try:
        import ultralytics
    except ModuleNotFoundError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ultralytics"])
        import ultralytics
    return getattr(ultralytics, cls_name)


def ensure_rfdetr_train_dependencies() -> None:
    """
    Ensure `rfdetr[train,loggers]` and `pytorch_lightning` are installed in the active runtime
    so `RFDETRSmall.train()` and `RFDETRLarge.train()` do not raise ModuleNotFoundError.
    """
    missing_train_deps = False
    for mod_name in ("rfdetr", "pytorch_lightning", "pycocotools", "supervision"):
        try:
            __import__(mod_name)
        except ModuleNotFoundError:
            missing_train_deps = True
            break
    if not missing_train_deps:
        try:
            import rfdetr.training  # type: ignore  # noqa: F401
        except Exception:
            missing_train_deps = True
    if missing_train_deps:
        logger.info("Installing rfdetr[train,loggers] and pytorch_lightning via pip...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "rfdetr[train,loggers]", "pytorch_lightning", "pycocotools", "supervision"]
        )


logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")

logger = logging.getLogger("YOLO11_IJITCS_DetectorBenchmark")

EXPECTED_CLASS_MAPPING: Dict[int, str] = {
    0: "Angkat Tangan",
    1: "Berdiri",
    2: "Diam",
    3: "Tidur",
}

CONFIG_INPUT: Dict[str, Any] = {
    "master_drive_root": "/content/drive/MyDrive/YOLO11_AHP_SAW_IJITCS_REVALIDATION_2026",
    "dataset_zip_relpath": "03_DATASET_AUDIT/bima_lppm2_.v2i.yolov11.zip",
    "dataset_unzipped_relpath": "03_DATASET_AUDIT/YOLO11_DATASET_UNZIPPED",
    "data_yaml_relpath": "03_DATASET_AUDIT/YOLO11_DATASET_UNZIPPED/data_colab.yaml",
    "data_yaml_fallback_relpath": "03_DATASET_AUDIT/data_colab.yaml",
    "yolov8_dataset_relpath": "03_DATASET_AUDIT/DETECTOR_DATASETS/YOLOv8",
    "yolov8_data_yaml_relpath": "03_DATASET_AUDIT/DETECTOR_DATASETS/YOLOv8/data.yaml",
    "rfdetr_dataset_relpath": "03_DATASET_AUDIT/DETECTOR_DATASETS/RF_DETR",
    "rfdetr_manifest_relpath": "03_DATASET_AUDIT/DETECTOR_DATASETS/RF_DETR/rf_detr_dataset_manifest.json",
    "parity_report_relpath": "03_DATASET_AUDIT/DETECTOR_DATASETS/DETECTOR_DATASET_SPLIT_PARITY_REPORT.csv",
    "checkpoints_dir_relpath": "04_MODEL_AND_CONFIGURATION/Checkpoints",
    "output_dir_relpath": "09_REVALIDATED_EXPERIMENT_OUTPUTS/detector_benchmarks",
    "yolo11_output_dir_relpath": "09_REVALIDATED_EXPERIMENT_OUTPUTS/detector_benchmarks/yolo11m",
    "yolov8_output_dir_relpath": "09_REVALIDATED_EXPERIMENT_OUTPUTS/detector_benchmarks/yolov8m",
    "rfdetr_small_output_dir_relpath": "09_REVALIDATED_EXPERIMENT_OUTPUTS/detector_benchmarks/rfdetr_small",
    "rfdetr_large_output_dir_relpath": "09_REVALIDATED_EXPERIMENT_OUTPUTS/detector_benchmarks/rfdetr_large",
    "rtdetr_output_dir_relpath": "09_REVALIDATED_EXPERIMENT_OUTPUTS/detector_benchmarks/rtdetr_l",
    "multi_detector_pipeline_relpath": "09_REVALIDATED_EXPERIMENT_OUTPUTS/multi_detector_pipeline",
    "imgsz": 640,
    "epochs": 150,
    "patience_baseline_focal": 30,
    "patience_balanced": 40,
    "batch": 16,
    "conf_eval": 0.001,
    "iou_eval": 0.70,
    "conf_inference": 0.25,
    "iou_inference": 0.50,
    "seeds": [42, 100, 2026],
}



def stage_dataset_to_local_colab_ssd(
    drive_dataset_dir: Path,
    local_cache_root: Path = Path("/content/local_detector_datasets"),
    dataset_type: str = "rfdetr",
    historical_zip_path: Optional[Path] = None,
) -> Path:
    """
    Stage RF-DETR or YOLOv8 dataset to fast local Colab NVMe SSD (/content/local_detector_datasets/...)
    routing every image strictly by the exact `file_name` entries in `{split}/_annotations.coco.json`
    (for RF-DETR) or split membership manifest so 100% of files required by DataLoader exist at
    `local_target / {split} / <file_name>`.
    """
    import zipfile

    drive_dataset_dir = Path(drive_dataset_dir)
    if not Path("/content/drive").exists() and not drive_dataset_dir.exists():
        return drive_dataset_dir

    local_target = Path(local_cache_root) / drive_dataset_dir.name
    marker_file = local_target / ".local_ssd_staged_v3_verified"
    if marker_file.is_file():
        logger.info("Using verified local Colab SSD dataset cache at %s", local_target)
        return local_target

    logger.info("Staging %s dataset to fast local Colab SSD at %s...", dataset_type, local_target)
    if local_target.exists():
        shutil.rmtree(local_target, ignore_errors=True)
    local_target.mkdir(parents=True, exist_ok=True)

    if historical_zip_path is None:
        cand_zip = drive_dataset_dir.parent.parent / "bima_lppm2_.v2i.yolov11.zip"
        if cand_zip.is_file():
            historical_zip_path = cand_zip
        else:
            colab_zip = Path("/content/drive/MyDrive/YOLO11_AHP_SAW_IJITCS_REVALIDATION_2026/03_DATASET_AUDIT/bima_lppm2_.v2i.yolov11.zip")
            if colab_zip.is_file():
                historical_zip_path = colab_zip

    # 1. Create split directory structure and copy _annotations.coco.json
    expected_by_split: Dict[str, List[str]] = {"train": [], "valid": [], "test": []}
    for sp in ("train", "valid", "test"):
        (local_target / sp / "images").mkdir(parents=True, exist_ok=True)
        if dataset_type == "yolov8":
            (local_target / sp / "labels").mkdir(parents=True, exist_ok=True)
        src_coco = drive_dataset_dir / sp / "_annotations.coco.json"
        if src_coco.is_file():
            dst_coco = local_target / sp / "_annotations.coco.json"
            shutil.copyfile(src_coco, dst_coco)
            coco_obj = json.loads(dst_coco.read_text(encoding="utf-8"))
            expected_by_split[sp] = [str(im["file_name"]) for im in coco_obj.get("images", [])]
        else:
            # Also check sibling RF_DETR/_annotations.coco.json so YOLOv8 gets the exact same 1373/393/196 routing
            rf_coco = drive_dataset_dir.parent / "RF_DETR" / sp / "_annotations.coco.json"
            if rf_coco.is_file():
                coco_obj = json.loads(rf_coco.read_text(encoding="utf-8"))
                expected_by_split[sp] = [str(im["file_name"]) for im in coco_obj.get("images", [])]

    # 2. Index ZIP archive by basename AND by base frame stem (before .rf.)
    zip_img_by_name: Dict[str, str] = {}
    zip_img_by_stem: Dict[str, str] = {}
    zip_lbl_by_name: Dict[str, str] = {}
    zip_lbl_by_stem: Dict[str, str] = {}
    zf = None
    if historical_zip_path is not None and Path(historical_zip_path).is_file():
        zf = zipfile.ZipFile(historical_zip_path, "r")
        for member in zf.infolist():
            if member.is_dir():
                continue
            bname = Path(member.filename).name
            ext = Path(bname).suffix.lower()
            stem = bname.split(".rf.")[0]
            if ext in (".jpg", ".jpeg", ".png"):
                zip_img_by_name[bname] = member.filename
                zip_img_by_stem.setdefault(stem, member.filename)
            elif ext == ".txt" and bname != "classes.txt":
                zip_lbl_by_name[bname] = member.filename
                zip_lbl_by_stem.setdefault(stem, member.filename)

    staged_images = 0
    missing_images = []
    try:
        if any(expected_by_split.values()):
            for sp in ("train", "valid", "test"):
                for fname in expected_by_split[sp]:
                    dst_root = local_target / sp / fname
                    dst_sub = local_target / sp / "images" / fname
                    stem = fname.split(".rf.")[0]

                    # Copy image from ZIP (by exact name or stem) or from Drive split folders
                    if zf is not None and (fname in zip_img_by_name or stem in zip_img_by_stem):
                        zmember = zip_img_by_name.get(fname) or zip_img_by_stem[stem]
                        with zf.open(zmember) as src_f, open(dst_root, "wb") as dst_f:
                            shutil.copyfileobj(src_f, dst_f)
                    else:
                        # Fallback to Drive RF_DETR, YOLOv8, or _STAGING_ROBOFLOW_COCO
                        for cand_p in (
                            drive_dataset_dir / sp / fname,
                            drive_dataset_dir / sp / "images" / fname,
                            drive_dataset_dir.parent / "YOLOv8" / sp / "images" / fname,
                            drive_dataset_dir.parent / "YOLOv8" / "valid" / "images" / fname,
                            drive_dataset_dir.parent / "YOLOv8" / "train" / "images" / fname,
                            drive_dataset_dir.parent / "YOLOv8" / "test" / "images" / fname,
                        ):
                            if cand_p.is_file():
                                shutil.copyfile(cand_p, dst_root)
                                break

                    if dst_root.is_file():
                        staged_images += 1
                        if not dst_sub.exists():
                            try:
                                os.symlink(dst_root, dst_sub)
                            except Exception:
                                shutil.copyfile(dst_root, dst_sub)
                    else:
                        missing_images.append(f"{sp}/{fname}")

                    if dataset_type == "yolov8":
                        lbl_name = fname[:-4] + ".txt" if fname.endswith(".jpg") else fname + ".txt"
                        dst_lbl = local_target / sp / "labels" / lbl_name
                        if zf is not None and (lbl_name in zip_lbl_by_name or stem in zip_lbl_by_stem):
                            zmember_lbl = zip_lbl_by_name.get(lbl_name) or zip_lbl_by_stem[stem]
                            with zf.open(zmember_lbl) as src_f, open(dst_lbl, "wb") as dst_f:
                                shutil.copyfileobj(src_f, dst_f)
    finally:
        if zf is not None:
            zf.close()

    if missing_images:
        raise FileNotFoundError(
            f"Local SSD staging could not find {len(missing_images)} images (e.g. {missing_images[:3]})."
        )

    if dataset_type == "yolov8":
        local_yaml = local_target / "data.yaml"
        local_yaml.write_text(
            f"path: {local_target}\n"
            "train: train/images\n"
            "val: valid/images\n"
            "test: test/images\n\n"
            "nc: 4\n"
            "names:\n"
            "  0: Angkat Tangan\n"
            "  1: Berdiri\n"
            "  2: Diam\n"
            "  3: Tidur\n",
            encoding="utf-8",
        )

    marker_file.write_text(f"OK_VERIFIED_{staged_images}\n", encoding="utf-8")
    logger.info("Local Colab SSD staging verified: %d/1962 images staged at %s (0 missing).", staged_images, local_target)
    return local_target


def verify_environment_and_inputs(master_root: Path) -> Dict[str, Any]:
    """Verifies separate dataset availability for YOLO11, YOLOv8, and RF-DETR (S & L), checkpoints, and output isolation."""
    master_root = Path(master_root)
    dataset_zip = master_root / CONFIG_INPUT["dataset_zip_relpath"]
    yolo11_dir = master_root / CONFIG_INPUT["dataset_unzipped_relpath"]
    yolo11_yaml = master_root / CONFIG_INPUT["data_yaml_relpath"]
    if not yolo11_yaml.exists():
        yolo11_yaml = master_root / CONFIG_INPUT["data_yaml_fallback_relpath"]
    yolov8_dir = master_root / CONFIG_INPUT["yolov8_dataset_relpath"]
    yolov8_yaml = master_root / CONFIG_INPUT["yolov8_data_yaml_relpath"]
    rfdetr_dir = master_root / CONFIG_INPUT["rfdetr_dataset_relpath"]
    ckpt_dir = master_root / CONFIG_INPUT["checkpoints_dir_relpath"]

    report = {
        "master_root_exists": master_root.exists(),
        "dataset_zip_exists": dataset_zip.exists(),
        "yolo11_dataset_preserved_exists": yolo11_dir.exists(),
        "yolo11_data_colab_yaml_exists": yolo11_yaml.exists(),
        "resolved_yolo11_data_yaml": str(yolo11_yaml),
        "yolov8_separate_dataset_exists": yolov8_dir.exists(),
        "yolov8_data_yaml_exists": yolov8_yaml.exists(),
        "resolved_yolov8_data_yaml": str(yolov8_yaml),
        "rfdetr_separate_dataset_exists": rfdetr_dir.exists(),
        "rfdetr_coco_train_exists": (rfdetr_dir / "train" / "_annotations.coco.json").exists(),
        "rfdetr_coco_valid_exists": (rfdetr_dir / "valid" / "_annotations.coco.json").exists(),
        "rfdetr_coco_test_exists": (rfdetr_dir / "test" / "_annotations.coco.json").exists(),
        "resolved_rfdetr_dataset_dir": str(rfdetr_dir),
        "checkpoints_dir_exists": ckpt_dir.exists(),
        "class_mapping_valid": EXPECTED_CLASS_MAPPING == {0: "Angkat Tangan", 1: "Berdiri", 2: "Diam", 3: "Tidur"},
        "isolated_output_dirs": {
            "yolo11m": str(master_root / CONFIG_INPUT["yolo11_output_dir_relpath"]),
            "yolov8m": str(master_root / CONFIG_INPUT["yolov8_output_dir_relpath"]),
            "rfdetr_small": str(master_root / CONFIG_INPUT["rfdetr_small_output_dir_relpath"]),
            "rfdetr_large": str(master_root / CONFIG_INPUT["rfdetr_large_output_dir_relpath"]),
            "rtdetr_l": str(master_root / CONFIG_INPUT["rtdetr_output_dir_relpath"]),
        },
        "multi_detector_pipeline_dirs": {
            "yolov8m": str(master_root / CONFIG_INPUT["multi_detector_pipeline_relpath"] / "yolov8m"),
            "rfdetr_small": str(master_root / CONFIG_INPUT["multi_detector_pipeline_relpath"] / "rfdetr_small"),
            "rfdetr_large": str(master_root / CONFIG_INPUT["multi_detector_pipeline_relpath"] / "rfdetr_large"),
        },
    }
    logger.info("Environment & Multi-Detector Input Verification: %s", json.dumps(report))
    return report


def evaluate_checkpoint_on_split(
    ckpt_path: Path,
    data_yaml: Path,
    split_name: str,
    out_dir: Path,
    imgsz: int = 640,
    batch: int = 16,
    conf: float = 0.001,
    iou: float = 0.70,
    device: str = "0",
    enable_eval: bool = False,
) -> Dict[str, Any]:
    """
    Evaluates a verified YOLO checkpoint separately on 'val' (393 images, 8,189 instances)
    or 'test' (196 images, 4,088 instances) and exports per-class CSV + JSON metrics.
    """
    if not enable_eval:
        raise RuntimeError(
            f"[SAFETY GUARD LOCKED] Evaluation of {ckpt_path.name} on split '{split_name}' is locked. "
            "Set ENABLE_MODEL_EVALUATION = True in Colab to execute."
        )
    if split_name not in ("val", "test"):
        raise ValueError(f"split_name must be 'val' or 'test', got {split_name!r}")
    ckpt_path = Path(ckpt_path)
    data_yaml = Path(data_yaml)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    YOLO = _get_ultralytics_cls("YOLO")
    model = YOLO(str(ckpt_path))
    model_names = model.names if isinstance(model.names, dict) else {i: n for i, n in enumerate(model.names)}
    if list(model_names.values()) != list(EXPECTED_CLASS_MAPPING.values()):
        raise RuntimeError(f"Incompatible class mapping in {ckpt_path.name}: {model_names}")

    metrics = model.val(
        data=str(data_yaml),
        split=split_name,
        imgsz=imgsz,
        batch=batch,
        conf=conf,
        iou=iou,
        device=device,
        plots=True,
        save_json=False,
        project=str(out_dir),
        name=f"eval_{ckpt_path.stem}_{split_name}",
        exist_ok=True,
    )
    per_class_rows = []
    for cid, cname in EXPECTED_CLASS_MAPPING.items():
        p_c, r_c, ap50_c, ap_c = metrics.box.class_result(cid)
        f1_c = (2.0 * p_c * r_c / (p_c + r_c)) if (p_c + r_c) > 0 else 0.0
        per_class_rows.append({
            "checkpoint": ckpt_path.name,
            "data_yaml": str(data_yaml),
            "split": split_name,
            "class_id": cid,
            "class_name": cname,
            "precision": float(p_c),
            "recall": float(r_c),
            "f1_score": float(f1_c),
            "mAP50": float(ap50_c),
            "mAP50_95": float(ap_c),
        })
    df_cls = pd.DataFrame(per_class_rows)
    df_cls.to_csv(out_dir / f"metrics_{ckpt_path.stem}_{split_name}_per_class.csv", index=False)

    mp = float(metrics.box.mp)
    mr = float(metrics.box.mr)
    mf1 = (2.0 * mp * mr / (mp + mr)) if (mp + mr) > 0 else 0.0
    summary = {
        "checkpoint": ckpt_path.name,
        "data_yaml": str(data_yaml),
        "split": split_name,
        "overall_precision": mp,
        "overall_recall": mr,
        "overall_f1": mf1,
        "overall_mAP50": float(metrics.box.map50),
        "overall_mAP50_95": float(metrics.box.map),
        "per_class_ap50": {EXPECTED_CLASS_MAPPING[i]: float(metrics.box.ap50[i]) for i in EXPECTED_CLASS_MAPPING},
        "per_class_recall": {EXPECTED_CLASS_MAPPING[i]: float(metrics.box.r[i]) for i in EXPECTED_CLASS_MAPPING},
        "confusion_matrix": metrics.confusion_matrix.matrix.tolist() if hasattr(metrics, "confusion_matrix") else None,
    }
    (out_dir / f"reproducibility_{ckpt_path.stem}_{split_name}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def evaluate_all_yolo11_checkpoints(
    checkpoint_dir: Path,
    data_yaml: Path,
    out_dir: Path,
    splits: Tuple[str, ...] = ("val", "test"),
    device: str = "0",
    enable_eval: bool = False,
) -> pd.DataFrame:
    """
    Evaluates all three historical YOLO11m variants (`baseline`, `focal`, `balanced`)
    across both 'val' and 'test' splits when `enable_eval=True`.
    """
    if not enable_eval:
        raise RuntimeError(
            "[SAFETY GUARD LOCKED] Multi-checkpoint evaluation is locked (ENABLE_MODEL_EVALUATION = False)."
        )
    checkpoint_dir = Path(checkpoint_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    variants = [
        ("baseline", checkpoint_dir / "yolo11m_baseline_best.pt"),
        ("focal", checkpoint_dir / "yolo11m_focal_best.pt"),
        ("balanced", checkpoint_dir / "yolo11m_balanced_best.pt"),
    ]
    rows = []
    for var_name, ckpt_path in variants:
        if not ckpt_path.exists() and var_name == "baseline":
            ckpt_path = checkpoint_dir / "best.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint for variant '{var_name}' not found at {ckpt_path}")
        for sp in splits:
            res = evaluate_checkpoint_on_split(
                ckpt_path=ckpt_path,
                data_yaml=data_yaml,
                split_name=sp,
                out_dir=out_dir,
                device=device,
                enable_eval=True,
            )
            rows.append({
                "variant": var_name,
                "checkpoint": ckpt_path.name,
                "dataset_yaml": str(data_yaml),
                "split": sp,
                "precision": res["overall_precision"],
                "recall": res["overall_recall"],
                "f1_score": res["overall_f1"],
                "mAP50": res["overall_mAP50"],
                "mAP50_95": res["overall_mAP50_95"],
                "ap50_angkat_tangan": res["per_class_ap50"]["Angkat Tangan"],
                "ap50_berdiri": res["per_class_ap50"]["Berdiri"],
                "ap50_diam": res["per_class_ap50"]["Diam"],
                "ap50_tidur": res["per_class_ap50"]["Tidur"],
            })
    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_dir / "yolo11_all_checkpoints_val_vs_test_summary.csv", index=False)
    return df_out


class BaseDetectorAdapter:
    """Base adapter with explicit execution safety guards for YOLO11m, YOLOv8m, RT-DETR-L, RF-DETR-Small, and RF-DETR-Large."""
    def __init__(self, model_name: str, weights_path: str, imgsz: int = 640, conf: float = 0.25, device: str = "0"):
        self.model_name = model_name
        self.weights_path = str(weights_path)
        self.imgsz = imgsz
        self.conf = conf
        self.device = device

    @staticmethod
    def check_guard(flag: bool, action_name: str) -> None:
        if not flag:
            raise RuntimeError(
                f"[SAFETY GUARD LOCKED] {action_name} is disabled by default. "
                "Set the corresponding ENABLE_* flag to True in Cell 1 when ready to run on Colab GPU."
            )


class YOLO11Adapter(BaseDetectorAdapter):
    """Primary detector adapter using Ultralytics YOLO11m (`yolo11m.pt`) on preserved YOLO11 dataset."""
    def load_model(self):
        YOLO = _get_ultralytics_cls("YOLO")
        return YOLO(self.weights_path)

    def train(self, data_yaml: str, out_dir: str, epochs: int = 150, patience: int = 30, batch: int = 16, seed: int = 42, enable_training: bool = False):
        self.check_guard(enable_training, "YOLO11m Training")
        model = self.load_model()
        return model.train(
            data=str(data_yaml),
            epochs=epochs,
            patience=patience,
            imgsz=self.imgsz,
            batch=batch,
            optimizer="auto",
            lr0=0.01,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3.0,
            cos_lr=True,
            close_mosaic=10,
            seed=seed,
            deterministic=True,
            device=self.device,
            project=str(out_dir),
            name=f"yolo11m_seed{seed}",
            exist_ok=True,
        )

    def evaluate_split(self, data_yaml: str, out_dir: str, split: str = "test", batch: int = 16, enable_eval: bool = False) -> Dict[str, Any]:
        return evaluate_checkpoint_on_split(
            ckpt_path=Path(self.weights_path),
            data_yaml=Path(data_yaml),
            split_name=split,
            out_dir=Path(out_dir),
            imgsz=self.imgsz,
            batch=batch,
            device=self.device,
            enable_eval=enable_eval,
        )


class YOLOv8mAdapter(BaseDetectorAdapter):
    """
    Reviewer Priority 3 predecessor baseline using Ultralytics YOLOv8m (`yolov8m.pt`)
    reading from its own separate dataset folder:
        03_DATASET_AUDIT/DETECTOR_DATASETS/YOLOv8/data.yaml
    and saving outputs to:
        09_REVALIDATED_EXPERIMENT_OUTPUTS/detector_benchmarks/yolov8m/
    """
    def load_model(self):
        YOLO = _get_ultralytics_cls("YOLO")
        if self.weights_path and Path(self.weights_path).exists():
            return YOLO(self.weights_path)
        return YOLO("yolov8m.pt")

    def train(self, data_yaml: str, out_dir: str, epochs: int = 150, patience: int = 30, batch: int = 16, seed: int = 42, use_local_ssd_cache: bool = True, enable_training: bool = False):
        self.check_guard(enable_training, "YOLOv8m Baseline Training")
        yaml_p = Path(data_yaml)
        if use_local_ssd_cache and yaml_p.parent.exists():
            local_v8_dir = stage_dataset_to_local_colab_ssd(yaml_p.parent, dataset_type="yolov8")
            if (local_v8_dir / "data.yaml").is_file():
                yaml_p = local_v8_dir / "data.yaml"
        model = self.load_model()
        res = model.train(
            data=str(yaml_p),
            cache="ram",
            epochs=epochs,
            patience=patience,
            imgsz=self.imgsz,
            batch=batch,
            optimizer="auto",
            lr0=0.01,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3.0,
            cos_lr=True,
            close_mosaic=10,
            seed=seed,
            deterministic=True,
            device=self.device,
            project=str(out_dir),
            name=f"yolov8m_baseline_seed{seed}",
            exist_ok=True,
        )
        best_pt = Path(out_dir) / f"yolov8m_baseline_seed{seed}" / "weights" / "best.pt"
        if best_pt.exists():
            self.weights_path = str(best_pt)
        return res

    def evaluate_split(self, data_yaml: str, out_dir: str, split: str = "test", batch: int = 16, enable_eval: bool = False) -> Dict[str, Any]:
        return evaluate_checkpoint_on_split(
            ckpt_path=Path(self.weights_path),
            data_yaml=Path(data_yaml),
            split_name=split,
            out_dir=Path(out_dir),
            imgsz=self.imgsz,
            batch=batch,
            device=self.device,
            enable_eval=enable_eval,
        )


class RTDETRLAdapter(BaseDetectorAdapter):
    """
    Reviewer Priority 3 Transformer detector baseline using Ultralytics `RTDETR("rtdetr-l.pt")`.
    Note: Sets `deterministic=False` on CUDA because deformable attention uses `F.grid_sample`.
    """
    def load_model(self):
        RTDETR = _get_ultralytics_cls("RTDETR")
        if self.weights_path and Path(self.weights_path).exists():
            return RTDETR(self.weights_path)
        return RTDETR("rtdetr-l.pt")

    def train(self, data_yaml: str, out_dir: str, epochs: int = 150, patience: int = 30, batch: int = 16, seed: int = 42, enable_training: bool = False):
        self.check_guard(enable_training, "RT-DETR-L Baseline Training")
        model = self.load_model()
        return model.train(
            data=str(data_yaml),
            epochs=epochs,
            patience=patience,
            imgsz=self.imgsz,
            batch=batch,
            optimizer="AdamW",
            lr0=0.0001,
            seed=seed,
            deterministic=False,
            device=self.device,
            project=str(out_dir),
            name=f"rtdetr_l_baseline_seed{seed}",
            exist_ok=True,
        )

    def evaluate_split(self, data_yaml: str, out_dir: str, split: str = "test", batch: int = 16, enable_eval: bool = False) -> Dict[str, Any]:
        self.check_guard(enable_eval, f"RT-DETR-L {split} Evaluation")
        model = self.load_model()
        metrics = model.val(
            data=str(data_yaml),
            split=split,
            imgsz=self.imgsz,
            batch=batch,
            conf=0.001,
            iou=0.70,
            device=self.device,
            project=str(out_dir),
            name=f"eval_rtdetr_l_{split}",
            exist_ok=True,
        )
        mp, mr = float(metrics.box.mp), float(metrics.box.mr)
        return {
            "checkpoint": self.weights_path,
            "data_yaml": str(data_yaml),
            "split": split,
            "overall_precision": mp,
            "overall_recall": mr,
            "overall_f1": (2.0 * mp * mr / (mp + mr)) if (mp + mr) > 0 else 0.0,
            "overall_mAP50": float(metrics.box.map50),
            "overall_mAP50_95": float(metrics.box.map),
            "per_class_ap50": {EXPECTED_CLASS_MAPPING[i]: float(metrics.box.ap50[i]) for i in EXPECTED_CLASS_MAPPING},
            "per_class_recall": {EXPECTED_CLASS_MAPPING[i]: float(metrics.box.r[i]) for i in EXPECTED_CLASS_MAPPING},
        }


class RFDETRAdapter(BaseDetectorAdapter):
    """
    DINOv2-backbone transformer detector adapter supporting both:
      - `model_size="S"` (`rfdetr.RFDETRSmall`, saved to `detector_benchmarks/rfdetr_small/`)
      - `model_size="L"` (`rfdetr.RFDETRLarge`, saved to `detector_benchmarks/rfdetr_large/`)
    Automatically installs `rfdetr[train,loggers]` and `pytorch_lightning` before loading/training.
    """
    def __init__(
        self,
        model_size: str = "S",
        weights_path: str = "",
        imgsz: int = 640,
        conf: float = 0.25,
        device: str = "0",
    ):
        norm_size = str(model_size).strip().upper()
        if norm_size not in ("S", "SMALL", "L", "LARGE"):
            norm_size = "S"
        self.model_size = "L" if norm_size in ("L", "LARGE") else "S"
        self.variant_slug = "rfdetr_large" if self.model_size == "L" else "rfdetr_small"
        default_w = "rf-detr-large.pth" if self.model_size == "L" else "rf-detr-small.pth"
        super().__init__(
            model_name=f"RF_DETR_{'Large' if self.model_size == 'L' else 'Small'}",
            weights_path=weights_path or default_w,
            imgsz=imgsz,
            conf=conf,
            device=device,
        )

    def load_model(self):
        ensure_rfdetr_train_dependencies()
        import rfdetr
        if self.model_size == "L":
            cls = getattr(rfdetr, "RFDETRLarge", getattr(rfdetr, "RFDETRBase", None))
        else:
            cls = getattr(rfdetr, "RFDETRSmall", getattr(rfdetr, "RFDETRBase", None))
        if cls is None:
            raise ImportError(f"Could not resolve RFDETR class for size={self.model_size} in rfdetr package.")
        if self.weights_path and Path(self.weights_path).exists():
            return cls(pretrain_weights=str(self.weights_path))
        return cls()

    def resolve_best_checkpoint(self, out_dir: str | Path) -> Optional[Path]:
        out_p = Path(out_dir)
        for cand_name in ("checkpoint_best_total.pth", "checkpoint_best_ema.pth", "checkpoint_best_regular.pth", "checkpoint.pth"):
            cand = out_p / cand_name
            if cand.is_file():
                return cand
        if self.weights_path and Path(self.weights_path).is_file():
            return Path(self.weights_path)
        return None

    def train(
        self,
        rfdetr_dataset_dir: str,
        out_dir: str,
        epochs: int = 35,
        batch: int = 8,
        grad_accum_steps: int = 2,
        lr: float = 1e-4,
        early_stopping: bool = True,
        early_stopping_patience: int = 8,
        num_workers: int = 2,
        use_local_ssd_cache: bool = True,
        enable_training: bool = False,
        checkpoint_export_dir: Optional[Path] = None,
    ):
        self.check_guard(enable_training, f"{self.model_name} Baseline Training")
        ensure_rfdetr_train_dependencies()
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        active_dataset_dir = (
            stage_dataset_to_local_colab_ssd(Path(rfdetr_dataset_dir), dataset_type="rfdetr")
            if use_local_ssd_cache
            else Path(rfdetr_dataset_dir)
        )

        model = self.load_model()
        import inspect
        train_kwargs: Dict[str, Any] = {
            "dataset_dir": str(active_dataset_dir),
            "epochs": int(epochs),
            "batch_size": int(batch),
            "grad_accum_steps": int(grad_accum_steps),
            "lr": float(lr),
            "output_dir": str(out_path),
        }
        try:
            sig = inspect.signature(model.train)
            if "early_stopping" in sig.parameters or any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            ):
                train_kwargs["early_stopping"] = bool(early_stopping)
                train_kwargs["early_stopping_patience"] = int(early_stopping_patience)
                train_kwargs["num_workers"] = int(num_workers)
        except Exception:
            pass

        res = model.train(**train_kwargs)
        best_ckpt = self.resolve_best_checkpoint(out_path)
        if best_ckpt is not None:
            self.weights_path = str(best_ckpt)
            if checkpoint_export_dir is not None:
                Path(checkpoint_export_dir).mkdir(parents=True, exist_ok=True)
                shutil.copy2(best_ckpt, Path(checkpoint_export_dir) / f"{self.variant_slug}_best.pth")
        return res

    def evaluate_coco_split(
        self,
        rfdetr_dataset_dir: str | Path,
        out_dir: str | Path,
        split_name: str = "test",
        conf_threshold: float = 0.001,
        enable_eval: bool = False,
    ) -> Dict[str, Any]:
        """
        Evaluate a trained RF-DETR-Small or RF-DETR-Large checkpoint on 'valid' (393 images)
        or 'test' (196 images) using pycocotools COCOeval and export per-class CSV + JSON metrics.
        """
        self.check_guard(enable_eval, f"{self.model_name} {split_name} Evaluation")
        ensure_rfdetr_train_dependencies()
        from PIL import Image
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        split_folder = "valid" if split_name in ("val", "valid") else "test"
        canonical_split_label = "val" if split_folder == "valid" else "test"
        ds_root = stage_dataset_to_local_colab_ssd(Path(rfdetr_dataset_dir), dataset_type='rfdetr')
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        coco_json_path = ds_root / split_folder / "_annotations.coco.json"
        if not coco_json_path.is_file():
            raise FileNotFoundError(f"COCO annotation file not found: {coco_json_path}")

        best_ckpt = self.resolve_best_checkpoint(out_path)
        if best_ckpt is not None:
            self.weights_path = str(best_ckpt)
        model = self.load_model()

        coco_gt = COCO(str(coco_json_path))
        coco_preds = []
        for img_id, img_info in coco_gt.imgs.items():
            fname = img_info["file_name"]
            img_p = ds_root / split_folder / "images" / fname
            if not img_p.is_file():
                img_p = ds_root / split_folder / fname
            if not img_p.is_file():
                continue
            im = Image.open(img_p).convert("RGB")
            dets = model.predict(im, threshold=conf_threshold)
            xyxy = np.asarray(getattr(dets, "xyxy", np.empty((0, 4))), dtype=float)
            confs = np.asarray(getattr(dets, "confidence", np.empty((0,))), dtype=float)
            cids = np.asarray(getattr(dets, "class_id", np.empty((0,))), dtype=int)
            for box, sc, cid in zip(xyxy, confs, cids):
                x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
                coco_preds.append({
                    "image_id": int(img_id),
                    "category_id": int(cid),
                    "bbox": [round(x1, 3), round(y1, 3), round(max(0.0, x2 - x1), 3), round(max(0.0, y2 - y1), 3)],
                    "score": float(sc),
                })

        if coco_preds:
            coco_dt = coco_gt.loadRes(coco_preds)
            coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            map50_95 = float(coco_eval.stats[0])
            map50 = float(coco_eval.stats[1])
            precisions = coco_eval.eval["precision"]  # [T, R, K, A, M]
            recalls = coco_eval.eval["recall"]        # [T, K, A, M]
        else:
            map50_95, map50 = 0.0, 0.0
            precisions, recalls = None, None

        per_class_rows = []
        for cid, cname in EXPECTED_CLASS_MAPPING.items():
            if precisions is not None and recalls is not None:
                p_slice50 = precisions[0, :, cid, 0, -1]
                p_slice50 = p_slice50[p_slice50 > -1]
                ap50_c = float(np.mean(p_slice50)) if len(p_slice50) else 0.0
                p_slice_all = precisions[:, :, cid, 0, -1]
                p_slice_all = p_slice_all[p_slice_all > -1]
                ap_c = float(np.mean(p_slice_all)) if len(p_slice_all) else 0.0
                r_val = float(recalls[0, cid, 0, -1]) if recalls[0, cid, 0, -1] > -1 else 0.0
                p_val = float(ap50_c)
            else:
                p_val, r_val, ap50_c, ap_c = 0.0, 0.0, 0.0, 0.0
            f1_c = (2.0 * p_val * r_val / (p_val + r_val)) if (p_val + r_val) > 0 else 0.0
            per_class_rows.append({
                "checkpoint": Path(self.weights_path).name,
                "model_variant": self.variant_slug,
                "split": canonical_split_label,
                "class_id": cid,
                "class_name": cname,
                "precision": p_val,
                "recall": r_val,
                "f1_score": f1_c,
                "mAP50": ap50_c,
                "mAP50_95": ap_c,
            })

        df_cls = pd.DataFrame(per_class_rows)
        df_cls.to_csv(out_path / f"metrics_{self.variant_slug}_{canonical_split_label}_per_class.csv", index=False)
        mp = float(df_cls["precision"].mean())
        mr = float(df_cls["recall"].mean())
        mf1 = (2.0 * mp * mr / (mp + mr)) if (mp + mr) > 0 else 0.0
        summary = {
            "checkpoint": str(self.weights_path),
            "model_variant": self.variant_slug,
            "split": canonical_split_label,
            "overall_precision": mp,
            "overall_recall": mr,
            "overall_f1": mf1,
            "overall_mAP50": map50,
            "overall_mAP50_95": map50_95,
            "per_class_ap50": {r["class_name"]: r["mAP50"] for r in per_class_rows},
            "per_class_recall": {r["class_name"]: r["recall"] for r in per_class_rows},
        }
        (out_path / f"reproducibility_{self.variant_slug}_{canonical_split_label}.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        return summary


class RFDETRSmallAdapter(RFDETRAdapter):
    """Backward-compatible alias for RF-DETR-Small (`model_size='S'`)."""
    def __init__(self, model_name: str = "RF_DETR_Small_Baseline", weights_path: str = "rf-detr-small.pth", imgsz: int = 640, conf: float = 0.25, device: str = "0"):
        super().__init__(model_size="S", weights_path=weights_path, imgsz=imgsz, conf=conf, device=device)


class RFDETRLargeAdapter(RFDETRAdapter):
    """Explicit adapter for RF-DETR-Large (`model_size='L'`)."""
    def __init__(self, model_name: str = "RF_DETR_Large_Baseline", weights_path: str = "rf-detr-large.pth", imgsz: int = 640, conf: float = 0.25, device: str = "0"):
        super().__init__(model_size="L", weights_path=weights_path, imgsz=imgsz, conf=conf, device=device)


def run_multi_detector_video_pipeline(
    detector_key: str,
    master_root: Path,
    video_path: Path,
    seats_csv_path: Path,
    base_identity_module: Any,
    tracker_replayer_cls: Any,
    conf_thresh: float = 0.25,
    fps: float = 25.0,
    total_video_frames: int = 23471,
    enable_video_inference: bool = False,
) -> Dict[str, Any]:
    """
    End-to-End Video Inference -> Multi-Object Tracking -> Fixed-Seat Identity Resolution
    -> 25 FPS Activity Duration Matrix generator for `yolov8m`, `rfdetr_small` ("S"), and `rfdetr_large` ("L").

    Saves all 6 downstream CSVs into:
        09_REVALIDATED_EXPERIMENT_OUTPUTS/multi_detector_pipeline/<detector_key>/
      1. detections.csv
      2. track_summary.csv
      3. track_to_student_summary.csv
      4. student_identity_audit.csv
      5. identity_assignments.csv
      6. student_activity_matrix_duration.csv
    so Notebook 08 (MCDM + Seating Perturbation) and Notebook 10 (Temporal Robustness)
    can run directly on YOLOv8m, RF-DETR-Small, and RF-DETR-Large!
    """
    try:
        from .paths_config import get_detector_pipeline_paths
    except ImportError:
        from paths_config import get_detector_pipeline_paths

    paths = get_detector_pipeline_paths(detector_key, master_root=master_root)
    norm_key = paths["detector_key"]
    pipe_dir = paths["pipeline_dir"]
    pipe_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = paths["checkpoint_path"]

    if not enable_video_inference and not paths["detections_csv"].exists():
        raise RuntimeError(
            f"[SAFETY GUARD LOCKED] Full video inference for detector '{norm_key}' is locked and "
            f"{paths['detections_csv']} does not exist yet. Set ENABLE_FULL_VIDEO_INFERENCE = True in Notebook 04."
        )

    if enable_video_inference and not paths["detections_csv"].exists():
        import cv2
        from PIL import Image

        if not Path(video_path).exists():
            raise FileNotFoundError(f"Classroom video not found at {video_path}")

        active_video_path = Path(video_path)
        if Path("/content/drive").exists():
            local_vid = Path("/content/local_video_part001.mp4")
            if not local_vid.is_file() or local_vid.stat().st_size != active_video_path.stat().st_size:
                logger.info("Staging classroom video to local Colab SSD (%s)...", local_vid)
                shutil.copyfile(active_video_path, local_vid)
            active_video_path = local_vid

        logger.info("Running frame-by-frame video detection on %s using detector '%s' (%s)...", active_video_path, norm_key, ckpt_path)
        raw_rows = []
        t0_vid = time.time()

        if norm_key in ("yolov8m", "yolo11m"):
            YOLO = _get_ultralytics_cls("YOLO")
            model = YOLO(str(ckpt_path))
            results_gen = model.track(
                source=str(active_video_path),
                conf=conf_thresh,
                iou=0.50,
                imgsz=640,
                tracker="botsort.yaml",
                stream=True,
                verbose=False,
            )
            for frame_idx, r in enumerate(results_gen):
                if frame_idx > 0 and frame_idx % 2000 == 0:
                    elapsed = max(1e-3, time.time() - t0_vid)
                    logger.info("[%s Video Inference] Frame %d / %d (%.1f%% | %.1f FPS)", norm_key, frame_idx, total_video_frames, 100.0 * frame_idx / total_video_frames, frame_idx / elapsed)
                if r.boxes is None or len(r.boxes) == 0:
                    continue
                xyxy = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                cids = r.boxes.cls.cpu().numpy().astype(int)
                tids = (
                    r.boxes.id.cpu().numpy().astype(int)
                    if r.boxes.id is not None
                    else np.arange(1, len(xyxy) + 1, dtype=int)
                )
                for box, sc, cid, tid in zip(xyxy, confs, cids, tids):
                    x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
                    raw_rows.append({
                        "frame": int(frame_idx),
                        "track_id": int(tid),
                        "class_id": int(cid),
                        "class_name": EXPECTED_CLASS_MAPPING.get(int(cid), str(cid)),
                        "confidence": round(float(sc), 6),
                        "x1": round(x1, 2),
                        "y1": round(y1, 2),
                        "x2": round(x2, 2),
                        "y2": round(y2, 2),
                        "cx": round(0.5 * (x1 + x2), 2),
                        "cy": round(0.5 * (y1 + y2), 2),
                        "w": round(max(0.0, x2 - x1), 2),
                        "h": round(max(0.0, y2 - y1), 2),
                    })
            det_df = pd.DataFrame(raw_rows)
        else:
            rf_size = "L" if norm_key == "rfdetr_large" else "S"
            rf_adapter = RFDETRAdapter(model_size=rf_size, weights_path=str(ckpt_path), conf=conf_thresh)
            rf_model = rf_adapter.load_model()
            if hasattr(rf_model, "optimize_for_inference"):
                try:
                    rf_model.optimize_for_inference()
                except Exception:
                    pass
            cap = cv2.VideoCapture(str(active_video_path))
            frame_idx = 0
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break
                if frame_idx > 0 and frame_idx % 1000 == 0:
                    elapsed = max(1e-3, time.time() - t0_vid)
                    logger.info("[%s Video Inference] Frame %d / %d (%.1f%% | %.1f FPS)", norm_key, frame_idx, total_video_frames, 100.0 * frame_idx / total_video_frames, frame_idx / elapsed)
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                pil_im = Image.fromarray(frame_rgb)
                dets = rf_model.predict(pil_im, threshold=conf_thresh)
                xyxy = np.asarray(getattr(dets, "xyxy", np.empty((0, 4))), dtype=float)
                confs = np.asarray(getattr(dets, "confidence", np.empty((0,))), dtype=float)
                cids = np.asarray(getattr(dets, "class_id", np.empty((0,))), dtype=int)
                for local_idx, (box, sc, cid) in enumerate(zip(xyxy, confs, cids), start=1):
                    x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
                    raw_rows.append({
                        "frame": int(frame_idx),
                        "track_id": int(local_idx),
                        "class_id": int(cid),
                        "class_name": EXPECTED_CLASS_MAPPING.get(int(cid), str(cid)),
                        "confidence": round(float(sc), 6),
                        "x1": round(x1, 2),
                        "y1": round(y1, 2),
                        "x2": round(x2, 2),
                        "y2": round(y2, 2),
                        "cx": round(0.5 * (x1 + x2), 2),
                        "cy": round(0.5 * (y1 + y2), 2),
                        "w": round(max(0.0, x2 - x1), 2),
                        "h": round(max(0.0, y2 - y1), 2),
                    })
                frame_idx += 1
            cap.release()
            untracked_df = pd.DataFrame(raw_rows)
            replayer = tracker_replayer_cls(method="bytetrack")
            det_df = replayer.run_on_cached_detections(untracked_df, enable_experiment=True)

        det_df.to_csv(paths["detections_csv"], index=False)
        logger.info("Saved %d frame detections for '%s' to %s", len(det_df), norm_key, paths["detections_csv"])
    else:
        det_df = pd.read_csv(paths["detections_csv"])

    # Build Track Summary, Fixed-Seat Identity Resolution & 25 FPS Duration Matrix
    from .detect_track import summarise_tracks
    track_sum_df = summarise_tracks(detections_csv=None, fps=float(fps), detections_df=det_df)
    track_sum_df.to_csv(paths["track_summary_csv"], index=False)

    seats_df = base_identity_module.load_seat_map(str(seats_csv_path), expected_students=21)
    icfg = base_identity_module.IdentityConfig.from_dict({"mode": "student_level"})
    id_assign_df = base_identity_module.assign_track_to_seat(det_df, seats_df, icfg)
    t2s_df, student_audit_df = base_identity_module.merge_track_fragments(
        id_assign_df, fps=float(fps), config=icfg, total_video_frames=int(total_video_frames), seats=seats_df
    )
    val_summary = base_identity_module.validate_seat_assignment(
        id_assign_df, seats_df, icfg, total_video_frames=int(total_video_frames)
    )
    student_det_df = base_identity_module.build_student_level_detections(id_assign_df)

    t2s_df.to_csv(paths["track_to_student_csv"], index=False)
    student_audit_df.to_csv(paths["student_audit_csv"], index=False)
    id_assign_df.to_csv(paths["identity_assignments_csv"], index=False)

    # Aggregate 21 x 4 Duration Matrix at 25 FPS
    all_students = sorted(seats_df["student_id"].astype(str).unique().tolist())
    activity_cols = ["Diam", "Tidur", "Angkat Tangan", "Berdiri"]
    dedup_df = (
        student_det_df.sort_values(["frame", "student_id", "confidence"], ascending=[True, True, False], kind="mergesort")
        .drop_duplicates(subset=["frame", "student_id"], keep="first")
    )
    counts = dedup_df.groupby(["student_id", "class_name"]).size().unstack(fill_value=0)
    dur_rows = []
    for sid in all_students:
        row_dict = {"entity": sid}
        for act in activity_cols:
            n_f = int(counts.loc[sid, act]) if (sid in counts.index and act in counts.columns) else 0
            row_dict[act] = round(n_f / float(fps), 2)
        dur_rows.append(row_dict)
    dur_df = pd.DataFrame(dur_rows)
    dur_df.to_csv(paths["duration_matrix_csv"], index=False)

    return {
        "detector_key": norm_key,
        "pipeline_dir": str(pipe_dir),
        "total_detections": len(det_df),
        "total_tracks": len(track_sum_df),
        "identity_validation": val_summary,
        "duration_matrix_path": str(paths["duration_matrix_csv"]),
    }




def ensure_detector_downstream_csvs_ready(
    detector_key: str,
    master_root: Path,
    seats_csv_path: Path,
    video_path: Optional[Path] = None,
    fps: float = 25.0,
    total_video_frames: int = 23471,
    auto_run_video_inference: bool = False,
) -> Dict[str, Any]:
    """
    Bulletproof helper for Notebook 04, Notebook 08, and Notebook 10:
      - Checks if all 6 downstream CSVs (`detections.csv`, `track_summary.csv`,
        `track_to_student_summary.csv`, `student_identity_audit.csv`,
        `identity_assignments.csv`, `student_activity_matrix_duration.csv`) exist for `detector_key`.
      - If `detections.csv` already exists (e.g. `yolov8m/detections.csv` was generated in Notebook 04)
        but any of the 5 derived CSVs (`student_activity_matrix_duration.csv`, `track_summary.csv`, etc.)
        is missing, it IMMEDIATELY builds all 5 derived CSVs from `detections.csv` on the spot!
      - If `detections.csv` does not exist yet and `auto_run_video_inference=True` (and the trained
        checkpoint exists), it runs full-video inference on `video_part001.mp4` and builds all 6 CSVs.
    """
    try:
        from .paths_config import get_detector_pipeline_paths
        from . import identity as base_identity_module
        from .tracker_adapters_aligned import OfflineTrackerReplayer
    except ImportError:
        from paths_config import get_detector_pipeline_paths
        import identity as base_identity_module
        from tracker_adapters_aligned import OfflineTrackerReplayer

    paths = get_detector_pipeline_paths(detector_key, master_root=master_root)
    norm_key = paths["detector_key"]
    required_keys = (
        "detections_csv",
        "track_summary_csv",
        "track_to_student_csv",
        "student_audit_csv",
        "identity_assignments_csv",
        "duration_matrix_csv",
    )
    all_exist = all(paths[k].is_file() for k in required_keys)
    if all_exist:
        return {"ready": True, "status": "ALL_CSVS_READY", "paths": paths}

    # Case 1: detections.csv already exists, but derived CSVs (duration_matrix_csv, etc.) are missing
    if paths["detections_csv"].is_file():
        logger.info(
            "Found existing detections.csv for '%s' at %s; auto-building missing duration matrix & identity CSVs...",
            norm_key,
            paths["detections_csv"],
        )
        run_multi_detector_video_pipeline(
            detector_key=norm_key,
            master_root=master_root,
            video_path=video_path or (Path(master_root) / "02_ORIGINAL_RESEARCH_ARCHIVE/Original_Videos/video_part001.mp4"),
            seats_csv_path=seats_csv_path,
            base_identity_module=base_identity_module,
            tracker_replayer_cls=OfflineTrackerReplayer,
            conf_thresh=0.25,
            fps=fps,
            total_video_frames=total_video_frames,
            enable_video_inference=False,
        )
        all_exist_after = all(paths[k].is_file() for k in required_keys)
        return {"ready": all_exist_after, "status": "AUTO_BUILT_FROM_DETECTIONS_CSV", "paths": paths}

    # Case 2: detections.csv does not exist yet, and auto_run_video_inference=True
    if auto_run_video_inference and paths["checkpoint_path"].is_file():
        vpath = video_path or (Path(master_root) / "02_ORIGINAL_RESEARCH_ARCHIVE/Original_Videos/video_part001.mp4")
        if Path(vpath).is_file():
            logger.info(
                "Auto-running video inference and downstream CSV generation for '%s' using checkpoint %s...",
                norm_key,
                paths["checkpoint_path"],
            )
            run_multi_detector_video_pipeline(
                detector_key=norm_key,
                master_root=master_root,
                video_path=Path(vpath),
                seats_csv_path=seats_csv_path,
                base_identity_module=base_identity_module,
                tracker_replayer_cls=OfflineTrackerReplayer,
                conf_thresh=0.25,
                fps=fps,
                total_video_frames=total_video_frames,
                enable_video_inference=True,
            )
            all_exist_after = all(paths[k].is_file() for k in required_keys)
            return {"ready": all_exist_after, "status": "AUTO_BUILT_FROM_VIDEO_INFERENCE", "paths": paths}

    return {
        "ready": False,
        "status": "AWAITING_VIDEO_INFERENCE_IN_NOTEBOOK_04",
        "paths": paths,
    }


class UnifiedDetectorBenchmarkSuite:
    """
    Orchestrates YOLO11m checkpoint evaluation ('val' & 'test') and YOLOv8m / RF-DETR-Small ('S') /
    RF-DETR-Large ('L') / RT-DETR-L baselines using strictly separated dataset and output directories.
    """

    def __init__(self, master_root: Path):
        self.master_root = Path(master_root)
        self.output_dir = self.master_root / CONFIG_INPUT["output_dir_relpath"]
        self.yolo11_output_dir = self.master_root / CONFIG_INPUT["yolo11_output_dir_relpath"]
        self.yolov8_output_dir = self.master_root / CONFIG_INPUT["yolov8_output_dir_relpath"]
        self.rfdetr_small_output_dir = self.master_root / CONFIG_INPUT["rfdetr_small_output_dir_relpath"]
        self.rfdetr_large_output_dir = self.master_root / CONFIG_INPUT["rfdetr_large_output_dir_relpath"]
        self.rtdetr_output_dir = self.master_root / CONFIG_INPUT["rtdetr_output_dir_relpath"]
        self.checkpoint_dir = self.master_root / CONFIG_INPUT["checkpoints_dir_relpath"]
        self.yolo11_data_yaml = self.master_root / CONFIG_INPUT["data_yaml_relpath"]
        self.yolov8_data_yaml = self.master_root / CONFIG_INPUT["yolov8_data_yaml_relpath"]
        self.rfdetr_dataset_dir = self.master_root / CONFIG_INPUT["rfdetr_dataset_relpath"]

    def get_model_specs(self, include_optional_rfdetr: bool = True) -> List[Dict[str, Any]]:
        specs = [
            {
                "model_id": "YOLO11m_Primary",
                "role": "PRIMARY_MODEL",
                "dataset_path": str(self.master_root / CONFIG_INPUT["dataset_unzipped_relpath"]),
                "config_or_annotation_path": str(self.yolo11_data_yaml),
                "weights": str(self.checkpoint_dir / "yolo11m_baseline_best.pt"),
                "output_dir": str(self.yolo11_output_dir),
                "api_class": "YOLO11Adapter (ultralytics.YOLO)",
                "mandatory_for_reviewer": True,
                "notes": "Primary classroom detector; evaluated on preserved YOLO11 dataset across baseline, focal, and balanced.",
            },
            {
                "model_id": "YOLOv8m_Baseline",
                "role": "REVIEWER_PRIORITY_3_BASELINE",
                "dataset_path": str(self.master_root / CONFIG_INPUT["yolov8_dataset_relpath"]),
                "config_or_annotation_path": str(self.yolov8_data_yaml),
                "weights": str(self.checkpoint_dir / "yolov8m.pt"),
                "output_dir": str(self.yolov8_output_dir),
                "api_class": "YOLOv8mAdapter (ultralytics.YOLO)",
                "mandatory_for_reviewer": True,
                "notes": "Direct predecessor YOLO baseline; uses separate 03_DATASET_AUDIT/DETECTOR_DATASETS/YOLOv8/data.yaml.",
            },
            {
                "model_id": "RT_DETR_L_Baseline",
                "role": "REVIEWER_PRIORITY_3_BASELINE",
                "dataset_path": str(self.master_root / CONFIG_INPUT["yolov8_dataset_relpath"]),
                "config_or_annotation_path": str(self.yolov8_data_yaml),
                "weights": str(self.checkpoint_dir / "rtdetr-l.pt"),
                "output_dir": str(self.rtdetr_output_dir),
                "api_class": "RTDETRLAdapter (ultralytics.RTDETR)",
                "mandatory_for_reviewer": True,
                "notes": "Real-Time DETR baseline; uses AdamW lr0=1e-4, 150 epochs, patience=30, and deterministic=False on CUDA.",
            },
        ]
        if include_optional_rfdetr:
            specs.extend([
                {
                    "model_id": "RF_DETR_Small_Baseline",
                    "role": "SEPARATE_COCO_DETR_BASELINE_S",
                    "dataset_path": str(self.rfdetr_dataset_dir),
                    "config_or_annotation_path": str(self.rfdetr_dataset_dir / "rf_detr_dataset_manifest.json"),
                    "weights": str(self.checkpoint_dir / "rf-detr-small.pth"),
                    "output_dir": str(self.rfdetr_small_output_dir),
                    "api_class": "RFDETRSmallAdapter (rfdetr.RFDETRSmall, RFDETR_MODEL_SIZE='S')",
                    "mandatory_for_reviewer": False,
                    "notes": "DINOv2-backbone RF-DETR-Small ('S'); saves to detector_benchmarks/rfdetr_small/.",
                },
                {
                    "model_id": "RF_DETR_Large_Baseline",
                    "role": "SEPARATE_COCO_DETR_BASELINE_L",
                    "dataset_path": str(self.rfdetr_dataset_dir),
                    "config_or_annotation_path": str(self.rfdetr_dataset_dir / "rf_detr_dataset_manifest.json"),
                    "weights": str(self.checkpoint_dir / "rf-detr-large.pth"),
                    "output_dir": str(self.rfdetr_large_output_dir),
                    "api_class": "RFDETRLargeAdapter (rfdetr.RFDETRLarge, RFDETR_MODEL_SIZE='L')",
                    "mandatory_for_reviewer": False,
                    "notes": "DINOv2-backbone RF-DETR-Large ('L'); saves to detector_benchmarks/rfdetr_large/.",
                },
            ])
        return specs

    def run_guarded_benchmark(
        self,
        data_yaml: Optional[Path] = None,
        yolov8_data_yaml: Optional[Path] = None,
        rfdetr_dataset_dir: Optional[Path] = None,
        rfdetr_model_size: str = "S",
        yolov8_epochs: int = 150,
        rfdetr_epochs: int = 35,
        rfdetr_early_stopping: bool = True,
        rfdetr_patience: int = 8,
        use_local_ssd_cache: bool = True,
        seed: int = 42,
        enable_eval: bool = False,
        enable_training: bool = False,
        enable_rfdetr_training: bool = False,
        device: str = "0",
    ) -> Dict[str, Any]:
        yolo11_yaml = Path(data_yaml) if data_yaml is not None else self.yolo11_data_yaml
        v8_yaml = Path(yolov8_data_yaml) if yolov8_data_yaml is not None else self.yolov8_data_yaml
        rf_dir = Path(rfdetr_dataset_dir) if rfdetr_dataset_dir is not None else self.rfdetr_dataset_dir
        norm_rf_size = str(rfdetr_model_size).strip().upper()

        if not enable_eval and not enable_training and not enable_rfdetr_training:
            logger.info(
                "[GUARDED] ENABLE_MODEL_EVALUATION, ENABLE_TRAINING, and ENABLE_RFDETR_TRAINING are False. Returning readiness manifest."
            )
            return {
                "status": "CODE_READY_LOCKED",
                "seed": seed,
                "rfdetr_model_size_selected": norm_rf_size,
                "yolo11_data_yaml": str(yolo11_yaml),
                "yolov8_data_yaml": str(v8_yaml),
                "rfdetr_dataset_dir": str(rf_dir),
                "models_configured": self.get_model_specs(include_optional_rfdetr=True),
                "yolo11_checkpoints_configured": ["yolo11m_baseline_best.pt", "yolo11m_focal_best.pt", "yolo11m_balanced_best.pt"],
                "isolated_output_dirs": {
                    "yolo11m": str(self.yolo11_output_dir),
                    "yolov8m": str(self.yolov8_output_dir),
                    "rfdetr_small": str(self.rfdetr_small_output_dir),
                    "rfdetr_large": str(self.rfdetr_large_output_dir),
                    "rtdetr_l": str(self.rtdetr_output_dir),
                },
            }

        self.output_dir.mkdir(parents=True, exist_ok=True)
        results_payload: Dict[str, Any] = {
            "seed": seed,
            "rfdetr_model_size_selected": norm_rf_size,
            "yolo11_data_yaml": str(yolo11_yaml),
            "yolov8_data_yaml": str(v8_yaml),
            "rfdetr_dataset_dir": str(rf_dir),
        }

        if enable_eval:
            logger.info("Executing separate 'val' and 'test' split evaluation on baseline, focal, and balanced YOLO11m checkpoints...")
            df_yolo11 = evaluate_all_yolo11_checkpoints(
                checkpoint_dir=self.checkpoint_dir,
                data_yaml=yolo11_yaml,
                out_dir=self.yolo11_output_dir,
                splits=("val", "test"),
                device=device,
                enable_eval=True,
            )
            results_payload["yolo11_val_test_summary"] = df_yolo11.to_dict(orient="records")

        if enable_training:
            logger.info("Executing YOLOv8m Training & Evaluation using separate dataset %s...", v8_yaml)
            v8_adapter = YOLOv8mAdapter("YOLOv8m_Baseline", str(self.checkpoint_dir / "yolov8m.pt"), imgsz=640, device=device)
            v8_adapter.train(str(v8_yaml), str(self.yolov8_output_dir), epochs=yolov8_epochs, patience=30, batch=16, seed=seed, use_local_ssd_cache=use_local_ssd_cache, enable_training=True)
            v8_best = self.yolov8_output_dir / f"yolov8m_baseline_seed{seed}" / "weights" / "best.pt"
            if v8_best.exists():
                v8_adapter.weights_path = str(v8_best)
                self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(v8_best, self.checkpoint_dir / "yolov8m_baseline_best.pt")
            v8_val = v8_adapter.evaluate_split(str(v8_yaml), str(self.yolov8_output_dir), split="val", enable_eval=True)
            v8_test = v8_adapter.evaluate_split(str(v8_yaml), str(self.yolov8_output_dir), split="test", enable_eval=True)
            results_payload["yolov8m"] = {"val": v8_val, "test": v8_test}

        if enable_rfdetr_training:
            sizes_to_run = ["S", "L"] if norm_rf_size == "BOTH" else (["L"] if norm_rf_size in ("L", "LARGE") else ["S"])
            for sz in sizes_to_run:
                target_out = self.rfdetr_large_output_dir if sz == "L" else self.rfdetr_small_output_dir
                default_ckpt = self.checkpoint_dir / ("rf-detr-large.pth" if sz == "L" else "rf-detr-small.pth")
                batch_sz = 4 if sz == "L" else 8
                grad_accum = 4 if sz == "L" else 2
                logger.info("Executing RF-DETR-%s Training using separate COCO JSON dataset %s -> %s...", sz, rf_dir, target_out)
                rf_adapter = RFDETRAdapter(model_size=sz, weights_path=str(default_ckpt), imgsz=640, device=device)
                rf_adapter.train(
                    rfdetr_dataset_dir=str(rf_dir),
                    out_dir=str(target_out),
                    epochs=rfdetr_epochs,
                    batch=batch_sz,
                    grad_accum_steps=grad_accum,
                    lr=1e-4,
                    early_stopping=rfdetr_early_stopping,
                    early_stopping_patience=rfdetr_patience,
                    use_local_ssd_cache=use_local_ssd_cache,
                    enable_training=True,
                    checkpoint_export_dir=self.checkpoint_dir,
                )
                rf_val = rf_adapter.evaluate_coco_split(rf_dir, target_out, split_name="valid", enable_eval=True)
                rf_test = rf_adapter.evaluate_coco_split(rf_dir, target_out, split_name="test", enable_eval=True)
                results_payload[rf_adapter.variant_slug] = {
                    "status": "TRAINED_AND_EVALUATED",
                    "output_dir": str(target_out),
                    "best_checkpoint": str(rf_adapter.weights_path),
                    "val": rf_val,
                    "test": rf_test,
                }

        results_payload["status"] = "EXECUTED"
        return results_payload


def verify_yolo11_checkpoints(checkpoint_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Verify historical YOLO11m checkpoints (`best.pt`, `yolo11m_baseline_best.pt`,
    `yolo11m_focal_best.pt`, `yolo11m_balanced_best.pt`) by byte size and SHA-256 hash.
    """
    try:
        from .paths_config import CHECKPOINT_DIR as DEFAULT_CKPT_DIR
    except ImportError:
        from paths_config import CHECKPOINT_DIR as DEFAULT_CKPT_DIR

    ckpt_dir = Path(checkpoint_dir) if checkpoint_dir is not None else DEFAULT_CKPT_DIR
    candidates = [
        ("PRIMARY_BASELINE", ckpt_dir / "best.pt", 40508069),
        ("BASELINE_BEST", ckpt_dir / "yolo11m_baseline_best.pt", 40508069),
        ("FOCAL_BEST", ckpt_dir / "yolo11m_focal_best.pt", 40508069),
        ("BALANCED_BEST", ckpt_dir / "yolo11m_balanced_best.pt", 40497957),
    ]
    report: Dict[str, Any] = {}
    selected_primary: Optional[str] = None
    for label, ckpt_path, expected_size in candidates:
        exists = ckpt_path.exists()
        size = ckpt_path.stat().st_size if exists else None
        is_yolo11m_size = (size == expected_size) if size is not None else False
        sha256_hex = None
        if exists and is_yolo11m_size:
            h = hashlib.sha256()
            with open(ckpt_path, "rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
            sha256_hex = h.hexdigest()
            if selected_primary is None and label in ("PRIMARY_BASELINE", "BASELINE_BEST"):
                selected_primary = str(ckpt_path)
        report[label] = {
            "path": str(ckpt_path),
            "exists": exists,
            "size_bytes": size,
            "expected_bytes": expected_size,
            "verified_yolo11m": is_yolo11m_size,
            "sha256": sha256_hex,
            "status": "FOUND_AND_VERIFIED" if (exists and is_yolo11m_size) else "NOT_FOUND",
        }
    decision = "READY_TO_USE" if selected_primary else "RETRAINING_REQUIRED"
    return {
        "decision": decision,
        "selected_primary_checkpoint": selected_primary,
        "checkpoints": report,
    }


HISTORICAL_CHECKPOINT_RECONCILIATION: List[Dict[str, Any]] = [
    {"Variant": "baseline", "Checkpoint_File": "yolo11m_baseline_best.pt", "Size_Bytes": 40508069, "Loss": "Standard BCE", "Oversampling": "1x (1,373 imgs)", "Epochs_Run": 150, "Best_Epoch": 144, "Ep1_Cls_Loss": 1.02261, "Val_Precision": 0.93938, "Val_Recall": 0.94250, "Val_mAP50": 0.97569, "Val_mAP50_95": 0.82514, "Role": "Primary video_part001.mp4 inference model"},
    {"Variant": "focal", "Checkpoint_File": "yolo11m_focal_best.pt", "Size_Bytes": 40508069, "Loss": "ElementwiseFocalLoss (g=2.0, a=0.25)", "Oversampling": "1x (1,373 imgs)", "Epochs_Run": 150, "Best_Epoch": 129, "Ep1_Cls_Loss": 0.12432, "Val_Precision": 0.93840, "Val_Recall": 0.92749, "Val_mAP50": 0.97705, "Val_mAP50_95": 0.82008, "Role": "Pure Focal Loss ablation"},
    {"Variant": "balanced", "Checkpoint_File": "yolo11m_balanced_best.pt", "Size_Bytes": 40497957, "Loss": "ElementwiseFocalLoss (g=2.0, a=0.25)", "Oversampling": "3x Class 0 & 1", "Epochs_Run": 71, "Best_Epoch": 31, "Ep1_Cls_Loss": 0.10905, "Val_Precision": 0.94120, "Val_Recall": 0.93580, "Val_mAP50": 0.98018, "Val_mAP50_95": 0.81951, "Role": "Evaluated in historical yolo11_test_metrics.json (Test mAP50=0.97888)"},
]
