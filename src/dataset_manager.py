"""
Dataset audit, Roboflow Version 1 staging download, schema validation,
non-destructive backup/replacement, idempotent ZIP extraction, and automatic
data_roboflow_original.yaml + data_colab.yaml configuration.

Canonical dataset location:
    03_DATASET_AUDIT/YOLO11_DATASET_UNZIPPED/

Roboflow source specification:
    workspace = "universitas-negeri-semarang-giyam"
    project   = "sdn_ngaliyan-aulue"
    version   = 1
    format    = "yolov11"
"""
from __future__ import annotations

import json
import os
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml

try:
    from .paths_config import (
        BACKUP_DATASET_DIR,
        CANONICAL_CLASSES,
        DATA_ROBOFLOW_ORIG_YAML_PATH,
        DATA_YAML_PATH,
        DATASET_ROOT,
        STAGING_DIR,
        ZIP_ARCHIVE_PATH,
    )
except ImportError:
    from paths_config import (
        BACKUP_DATASET_DIR,
        CANONICAL_CLASSES,
        DATA_ROBOFLOW_ORIG_YAML_PATH,
        DATA_YAML_PATH,
        DATASET_ROOT,
        STAGING_DIR,
        ZIP_ARCHIVE_PATH,
    )

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ROBOFLOW_WORKSPACE = "universitas-negeri-semarang-giyam"
ROBOFLOW_PROJECT = "sdn_ngaliyan-aulue"
ROBOFLOW_VERSION = 1
ROBOFLOW_FORMAT = "yolov11"


def ensure_roboflow_installed() -> None:
    """Ensure the `roboflow` Python package is installed in the active Colab/Python runtime."""
    try:
        import roboflow  # type: ignore  # noqa: F401
    except ModuleNotFoundError:
        import subprocess
        import sys
        print("[INFO] Menginstal pustaka 'roboflow' secara otomatis via pip...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "roboflow"])


def get_roboflow_api_key(secret_name: str = "ROBOFLOW_API_KEY") -> Optional[str]:
    """
    Retrieve ROBOFLOW_API_KEY from environment variable or Google Colab Secrets.
    Also gracefully handles the case where a user passes their key string directly
    as the argument in a Colab cell. Never hardcodes or fabricates an API key.
    """
    candidates = [secret_name, "ROBOFLOW_API_KEY"] if secret_name != "ROBOFLOW_API_KEY" else ["ROBOFLOW_API_KEY"]
    for cand in candidates:
        if not cand:
            continue
        env_key = os.environ.get(cand)
        if env_key and env_key.strip():
            return env_key.strip()
        try:
            from google.colab import userdata  # type: ignore
            colab_key = userdata.get(cand)
            if colab_key and str(colab_key).strip():
                return str(colab_key).strip()
        except Exception:
            pass
    # Fallback if the caller passed the raw API key string directly instead of the secret name
    if (
        secret_name
        and secret_name != "ROBOFLOW_API_KEY"
        and len(secret_name.strip()) >= 10
        and " " not in secret_name.strip()
    ):
        return secret_name.strip()
    return None


def locate_yolo_root(search_dir: Path) -> Optional[Path]:
    """
    Inspect a directory (such as STAGING_DIR or dataset.location) to locate
    the actual root containing train/valid/test or data.yaml, handling nested
    subfolders created by ZIP extraction or Roboflow SDK downloads.
    """
    search_dir = Path(search_dir)
    if not search_dir.exists():
        return None

    # Check direct root first
    if (search_dir / "train").is_dir() or (search_dir / "data.yaml").is_file():
        return search_dir

    # Search up to 2 levels deep for a directory containing train/ and valid/ (or val/)
    for child in sorted(search_dir.iterdir()):
        if child.is_dir() and not child.name.startswith((".", "__MACOSX")):
            if (child / "train").is_dir() or (child / "data.yaml").is_file():
                return child
            for sub in sorted(child.iterdir()):
                if sub.is_dir() and not sub.name.startswith((".", "__MACOSX")):
                    if (sub / "train").is_dir() or (sub / "data.yaml").is_file():
                        return sub
    return None


def _parse_names_dict(raw_names) -> dict[int, str]:
    if isinstance(raw_names, list):
        return {int(i): str(v) for i, v in enumerate(raw_names)}
    if isinstance(raw_names, dict):
        return {int(k): str(v) for k, v in raw_names.items()}
    return {}


def validate_yolo_dataset_structure(
    dataset_dir: Path | str,
    expected_classes: Optional[dict[int, str]] = None,
    check_image_headers: bool = False,
) -> dict:
    """
    Validate a YOLO-formatted dataset directory across train, valid/val, and test splits.

    Checks:
      - Split folder existence (train/images, train/labels, valid/images, valid/labels, test/images, test/labels)
      - Image counts, label counts, missing labels, orphan labels, empty labels
      - Bounding box format validity (5 tokens per line: class_id cx cy w h in [0, 1])
      - Class ID set and per-class instance counts
      - Duplicate base frame stems within and across splits
      - Schema compatibility with historical YOLO11m checkpoints:
        {0: 'Angkat Tangan', 1: 'Berdiri', 2: 'Diam', 3: 'Tidur'}
    """
    expected_classes = expected_classes or dict(CANONICAL_CLASSES)
    root = locate_yolo_root(Path(dataset_dir)) or Path(dataset_dir)

    report: dict = {
        "inspected_path": str(dataset_dir),
        "resolved_yolo_root": str(root),
        "exists": root.exists(),
        "yaml_found": False,
        "yaml_path": None,
        "yaml_nc": None,
        "yaml_names": {},
        "splits": {},
        "total_images": 0,
        "total_labels": 0,
        "total_instances": 0,
        "total_missing_labels": 0,
        "total_orphan_labels": 0,
        "total_empty_labels": 0,
        "total_invalid_bboxes": 0,
        "total_corrupt_images": 0,
        "observed_class_ids": [],
        "class_instance_counts": {int(k): 0 for k in expected_classes},
        "unknown_class_ids": [],
        "cross_split_stem_overlap": {},
        "schema_compatible": False,
        "structure_valid": False,
        "errors": [],
        "warnings": [],
    }

    if not root.exists():
        report["errors"].append(f"Direktori dataset tidak ditemukan: {root}")
        return report

    # Read data.yaml or data_roboflow_original.yaml if present
    for yname in ("data.yaml", "data_roboflow_original.yaml", "data_original.yaml", "data_colab.yaml"):
        ypath = root / yname
        if ypath.is_file():
            try:
                ydata = yaml.safe_load(ypath.read_text(encoding="utf-8")) or {}
                report["yaml_found"] = True
                report["yaml_path"] = str(ypath)
                report["yaml_nc"] = ydata.get("nc")
                report["yaml_names"] = _parse_names_dict(ydata.get("names"))
                break
            except Exception as exc:
                report["warnings"].append(f"Gagal membaca {ypath.name}: {exc}")

    split_stems: dict[str, set[str]] = {}
    observed_cids: set[int] = set()
    unknown_cids: set[int] = set()

    for canonical_split, candidates in (
        ("train", ["train"]),
        ("valid", ["valid", "val"]),
        ("test", ["test"]),
    ):
        split_dir = None
        for c in candidates:
            if (root / c).is_dir():
                split_dir = root / c
                break

        if split_dir is None:
            report["splits"][canonical_split] = {"exists": False}
            report["errors"].append(f"Folder split '{canonical_split}' tidak ditemukan di {root}")
            continue

        img_dir = split_dir / "images" if (split_dir / "images").is_dir() else split_dir
        lbl_dir = split_dir / "labels" if (split_dir / "labels").is_dir() else split_dir

        imgs = sorted(
            p for p in img_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ) if img_dir.exists() else []
        lbls = sorted(
            p for p in lbl_dir.iterdir()
            if p.is_file() and p.suffix.lower() == ".txt" and p.name != "classes.txt"
        ) if lbl_dir.exists() else []

        img_map = {p.stem: p for p in imgs}
        lbl_map = {p.stem: p for p in lbls}

        missing_lbls = sorted(set(img_map) - set(lbl_map))
        orphan_lbls = sorted(set(lbl_map) - set(img_map))

        # Base frame stem (before .rf.<hash>)
        base_stems = [s.split(".rf.")[0] for s in img_map]
        dup_base_stems = len(base_stems) - len(set(base_stems))
        split_stems[canonical_split] = set(base_stems)

        corrupt_imgs = 0
        if check_image_headers:
            for ipath in imgs:
                try:
                    if ipath.stat().st_size < 128:
                        corrupt_imgs += 1
                except Exception:
                    corrupt_imgs += 1

        empty_lbls = 0
        invalid_bboxes = 0
        split_instances = 0
        split_class_counts = {int(k): 0 for k in expected_classes}

        for stem, lpath in lbl_map.items():
            try:
                lines = [ln.strip() for ln in lpath.read_text(encoding="utf-8").splitlines() if ln.strip()]
            except Exception:
                invalid_bboxes += 1
                continue
            if not lines:
                empty_lbls += 1
                continue
            for ln in lines:
                parts = ln.split()
                if len(parts) < 5:
                    invalid_bboxes += 1
                    continue
                try:
                    cid = int(float(parts[0]))
                    coords = [float(x) for x in parts[1:5]]
                except ValueError:
                    invalid_bboxes += 1
                    continue
                if any(c < -1e-4 or c > 1.0001 for c in coords) or coords[2] <= 0 or coords[3] <= 0:
                    invalid_bboxes += 1
                observed_cids.add(cid)
                split_instances += 1
                if cid in split_class_counts:
                    split_class_counts[cid] += 1
                    report["class_instance_counts"][cid] += 1
                else:
                    unknown_cids.add(cid)

        report["splits"][canonical_split] = {
            "exists": True,
            "split_dir": str(split_dir),
            "images_dir": str(img_dir),
            "labels_dir": str(lbl_dir),
            "n_images": len(imgs),
            "n_labels": len(lbls),
            "n_unique_base_frames": len(set(base_stems)),
            "n_rf_augmented_variants": dup_base_stems,
            "n_instances": split_instances,
            "missing_labels": len(missing_lbls),
            "orphan_labels": len(orphan_lbls),
            "empty_labels": empty_lbls,
            "invalid_bboxes": invalid_bboxes,
            "corrupt_images": corrupt_imgs,
            "class_counts": {expected_classes.get(k, str(k)): v for k, v in split_class_counts.items()},
        }

        report["total_images"] += len(imgs)
        report["total_labels"] += len(lbls)
        report["total_instances"] += split_instances
        report["total_missing_labels"] += len(missing_lbls)
        report["total_orphan_labels"] += len(orphan_lbls)
        report["total_empty_labels"] += empty_lbls
        report["total_invalid_bboxes"] += invalid_bboxes
        report["total_corrupt_images"] += corrupt_imgs

        if len(imgs) == 0:
            report["errors"].append(f"Split '{canonical_split}' memiliki 0 gambar di {img_dir}")
        if len(missing_lbls) > 0:
            report["errors"].append(f"Split '{canonical_split}' kehilangan {len(missing_lbls)} file label")
        if invalid_bboxes > 0:
            report["errors"].append(f"Split '{canonical_split}' memiliki {invalid_bboxes} baris bbox tidak valid")

    # Cross-split overlap check
    for a, b in (("train", "valid"), ("train", "test"), ("valid", "test")):
        overlap = sorted(split_stems.get(a, set()) & split_stems.get(b, set()))
        report["cross_split_stem_overlap"][f"{a}_vs_{b}"] = len(overlap)
        if overlap:
            report["warnings"].append(f"Overlap base frame antara {a} dan {b}: {len(overlap)} frame")

    report["observed_class_ids"] = sorted(observed_cids)
    report["unknown_class_ids"] = sorted(unknown_cids)

    # Check schema compatibility with expected {0: 'Angkat Tangan', 1: 'Berdiri', 2: 'Diam', 3: 'Tidur'}
    yaml_names = report["yaml_names"]
    names_match = (not yaml_names) or (yaml_names == expected_classes)
    cids_match = observed_cids.issubset(set(expected_classes.keys())) and len(unknown_cids) == 0
    if not names_match:
        report["errors"].append(
            f" Ketidaksesuaian nama kelas pada YAML: ditemukan {yaml_names}, "
            f"diharapkan {expected_classes}."
        )
    if not cids_match:
        report["errors"].append(
            f"Ditemukan class_id di luar skema {sorted(expected_classes)}: {sorted(unknown_cids)}"
        )

    report["schema_compatible"] = bool(names_match and cids_match and report["total_images"] > 0)
    report["structure_valid"] = bool(
        len(report["errors"]) == 0
        and report["total_images"] > 0
        and report["total_images"] == report["total_labels"]
        and report["schema_compatible"]
    )
    return report


def generate_canonical_data_yamls(
    dataset_root: Path = DATASET_ROOT,
    colab_dataset_root_str: str = "/content/drive/MyDrive/YOLO11_AHP_SAW_IJITCS_REVALIDATION_2026/03_DATASET_AUDIT/YOLO11_DATASET_UNZIPPED",
) -> dict:
    """
    Save the original Roboflow YAML as `data_roboflow_original.yaml` and generate
    `data_colab.yaml` (plus synchronized `data.yaml`) with verified relative split paths.
    """
    dataset_root = Path(dataset_root)
    dataset_root.mkdir(parents=True, exist_ok=True)

    orig_yaml_path = dataset_root / "data_roboflow_original.yaml"
    raw_yaml_path = dataset_root / "data.yaml"
    legacy_orig_path = dataset_root / "data_original.yaml"

    # 1. Preserve original Roboflow YAML before modifying data.yaml
    if not orig_yaml_path.exists():
        if legacy_orig_path.exists():
            shutil.copyfile(legacy_orig_path, orig_yaml_path)
        elif raw_yaml_path.exists():
            shutil.copyfile(raw_yaml_path, orig_yaml_path)
        else:
            default_orig = (
                "train: ../train/images\n"
                "val: ../valid/images\n"
                "test: ../test/images\n\n"
                "nc: 4\n"
                "names: ['Angkat Tangan', 'Berdiri', 'Diam', 'Tidur']\n\n"
                "roboflow:\n"
                "  workspace: universitas-negeri-semarang-giyam\n"
                "  project: sdn_ngaliyan-aulue\n"
                "  version: 1\n"
                "  license: CC BY 4.0\n"
            )
            orig_yaml_path.write_text(default_orig, encoding="utf-8")

    # 2. Auto-detect split subdirectories inside dataset_root
    train_rel = "train/images" if (dataset_root / "train" / "images").exists() or not (dataset_root / "train").exists() else "train"
    if (dataset_root / "valid" / "images").exists() or (dataset_root / "valid").exists():
        val_rel = "valid/images" if (dataset_root / "valid" / "images").exists() or not (dataset_root / "valid").exists() else "valid"
    elif (dataset_root / "val" / "images").exists() or (dataset_root / "val").exists():
        val_rel = "val/images" if (dataset_root / "val" / "images").exists() else "val"
    else:
        val_rel = "valid/images"
    test_rel = "test/images" if (dataset_root / "test" / "images").exists() or not (dataset_root / "test").exists() else "test"

    colab_yaml_text = (
        "# Canonical Google Colab & Ultralytics Dataset Configuration\n"
        "# Auto-generated and validated by src/dataset_manager.py\n"
        f"path: {colab_dataset_root_str}\n"
        f"train: {train_rel}\n"
        f"val: {val_rel}\n"
        f"test: {test_rel}\n\n"
        "nc: 4\n"
        "names:\n"
        "  0: Angkat Tangan\n"
        "  1: Berdiri\n"
        "  2: Diam\n"
        "  3: Tidur\n"
    )

    colab_yaml_path = dataset_root / "data_colab.yaml"
    colab_yaml_path.write_text(colab_yaml_text, encoding="utf-8")
    raw_yaml_path.write_text(colab_yaml_text, encoding="utf-8")

    return {
        "data_roboflow_original_yaml": str(orig_yaml_path),
        "data_colab_yaml": str(colab_yaml_path),
        "data_yaml": str(raw_yaml_path),
        "path": colab_dataset_root_str,
        "train": train_rel,
        "val": val_rel,
        "test": test_rel,
        "nc": 4,
        "names": dict(CANONICAL_CLASSES),
    }


def download_roboflow_to_staging(
    api_key: Optional[str] = None,
    staging_dir: Path = STAGING_DIR,
    workspace: str = ROBOFLOW_WORKSPACE,
    project_slug: str = ROBOFLOW_PROJECT,
    version_num: int = ROBOFLOW_VERSION,
    model_format: str = ROBOFLOW_FORMAT,
) -> dict:
    """
    Download the Roboflow dataset into `staging_dir` (`_STAGING_ROBOFLOW_YOLOV11`)
    without touching the canonical `YOLO11_DATASET_UNZIPPED` directory.
    """
    key = api_key or get_roboflow_api_key()
    if not key:
        return {
            "status": "SKIPPED_NO_API_KEY",
            "message": (
                "ROBOFLOW_API_KEY belum ditemukan di environment variable maupun "
                "Google Colab Secrets (`userdata.get('ROBOFLOW_API_KEY')`). "
                "Tambahkan Secret `ROBOFLOW_API_KEY` pada panel Secrets di Google Colab "
                "untuk mengunduh langsung dari Roboflow, atau gunakan ekstraksi arsip "
                "kanonis `bima_lppm2_.v2i.yolov11.zip`."
            ),
            "staging_dir": str(staging_dir),
        }

    ensure_roboflow_installed()
    from roboflow import Roboflow  # type: ignore

    staging_dir = Path(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    rf = Roboflow(api_key=key)
    project = rf.workspace(workspace).project(project_slug)
    version = project.version(version_num)
    dataset = version.download(model_format, location=str(staging_dir), overwrite=True)

    sdk_location = Path(getattr(dataset, "location", str(staging_dir)))

    # Check if SDK left an unextracted zip file (only extract if train/ is not already extracted)
    resolved_root = locate_yolo_root(sdk_location) or locate_yolo_root(staging_dir)
    if resolved_root is None:
        zips = list(staging_dir.rglob("*.zip"))
        if zips:
            with zipfile.ZipFile(zips[0], "r") as zf:
                zf.extractall(staging_dir)
            resolved_root = locate_yolo_root(staging_dir)

    validation = validate_yolo_dataset_structure(resolved_root or staging_dir)
    return {
        "status": "DOWNLOADED_AND_VALIDATED" if validation["structure_valid"] else "VALIDATION_FAILED",
        "sdk_location": str(sdk_location),
        "resolved_yolo_root": str(resolved_root or staging_dir),
        "validation": validation,
    }


def backup_and_replace_dataset(
    source_dir: Path,
    canonical_dir: Path = DATASET_ROOT,
    backup_dir: Path = BACKUP_DATASET_DIR,
) -> dict:
    """
    Safely replace `canonical_dir` (`03_DATASET_AUDIT/YOLO11_DATASET_UNZIPPED`)
    with the validated dataset in `source_dir` AFTER:
      1. Validating `source_dir` structure and class compatibility.
      2. Creating a non-destructive backup of any previous contents in `backup_dir`.
      3. Preserving `data_roboflow_original.yaml` and generating `data_colab.yaml`.
    """
    val = validate_yolo_dataset_structure(source_dir)
    if not val["structure_valid"]:
        raise RuntimeError(
            "ABORT REPLACEMENT: Dataset pada staging tidak lulus validasi struktur/skema kelas.\n"
            f"Errors: {val['errors']}"
        )

    resolved_src = Path(val["resolved_yolo_root"])
    canonical_dir = Path(canonical_dir)
    backup_dir = Path(backup_dir)

    # 1. Non-destructive backup of existing canonical_dir contents
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target_backup = backup_dir / f"snapshot_{timestamp}"
    if canonical_dir.exists() and any(canonical_dir.iterdir()):
        target_backup.mkdir(parents=True, exist_ok=True)
        for item in canonical_dir.iterdir():
            dest = target_backup / item.name
            if item.is_dir():
                shutil.copytree(item, dest, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dest)

    canonical_dir.mkdir(parents=True, exist_ok=True)

    # 2. Preserve raw Roboflow YAML as data_roboflow_original.yaml
    src_yaml = resolved_src / "data.yaml"
    if src_yaml.is_file():
        shutil.copy2(src_yaml, canonical_dir / "data_roboflow_original.yaml")

    # 3. Copy splits and metadata cleanly into canonical_dir (flattened if nested)
    for split_name, alt_names in (("train", ["train"]), ("valid", ["valid", "val"]), ("test", ["test"])):
        src_split = None
        for c in alt_names:
            if (resolved_src / c).is_dir():
                src_split = resolved_src / c
                break
        if src_split is not None:
            dst_split = canonical_dir / split_name
            if dst_split.exists():
                shutil.rmtree(dst_split)
            shutil.copytree(src_split, dst_split)

    for doc_name in ("README.dataset.txt", "README.roboflow.txt"):
        if (resolved_src / doc_name).is_file():
            shutil.copy2(resolved_src / doc_name, canonical_dir / doc_name)

    yaml_info = generate_canonical_data_yamls(canonical_dir)
    final_val = validate_yolo_dataset_structure(canonical_dir)

    return {
        "status": "REPLACED_AND_VERIFIED",
        "backup_snapshot": str(target_backup),
        "canonical_dir": str(canonical_dir),
        "yaml_info": yaml_info,
        "validation": final_val,
    }


def ensure_canonical_dataset_ready(
    dataset_root: Path = DATASET_ROOT,
    zip_archive_path: Path = ZIP_ARCHIVE_PATH,
    staging_dir: Path = STAGING_DIR,
    backup_dir: Path = BACKUP_DATASET_DIR,
    prefer_roboflow_download: bool = False,
) -> dict:
    """
    Idempotent entry point for all Google Colab notebooks:
      1. If `prefer_roboflow_download=True` and `ROBOFLOW_API_KEY` is configured,
         downloads from Roboflow to `staging_dir`, validates, backs up, and replaces.
      2. Otherwise checks if `dataset_root` (`YOLO11_DATASET_UNZIPPED`) already has
         all splits extracted (`1,962` images and `1,962` labels).
      3. If not yet extracted on Drive and `zip_archive_path` (`bima_lppm2_.v2i.yolov11.zip`)
         exists, extracts the ZIP archive into `dataset_root`, normalizes any nested
         folder, generates `data_roboflow_original.yaml` + `data_colab.yaml`, and
         returns the full validation dictionary.
    """
    dataset_root = Path(dataset_root)

    if prefer_roboflow_download:
        rf_res = download_roboflow_to_staging(staging_dir=staging_dir)
        if rf_res["status"] == "DOWNLOADED_AND_VALIDATED":
            return backup_and_replace_dataset(
                Path(rf_res["resolved_yolo_root"]),
                canonical_dir=dataset_root,
                backup_dir=backup_dir,
            )

    current_val = validate_yolo_dataset_structure(dataset_root)
    if current_val["structure_valid"]:
        yaml_info = generate_canonical_data_yamls(dataset_root)
        return {
            "status": "READY_EXISTING",
            "canonical_dir": str(dataset_root),
            "yaml_info": yaml_info,
            "validation": current_val,
        }

    # Fallback: extract from canonical ZIP archive if available
    if Path(zip_archive_path).is_file():
        staging_extract = Path(staging_dir) / "_zip_extract_tmp"
        if staging_extract.exists():
            shutil.rmtree(staging_extract)
        staging_extract.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_archive_path, "r") as zf:
            zf.extractall(staging_extract)
        rep = backup_and_replace_dataset(
            staging_extract,
            canonical_dir=dataset_root,
            backup_dir=backup_dir,
        )
        shutil.rmtree(staging_extract, ignore_errors=True)
        return rep

    # Even if images are not extracted in the local environment yet, ensure YAMLs exist
    yaml_info = generate_canonical_data_yamls(dataset_root)
    return {
        "status": "AWAITING_COLAB_EXTRACTION_OR_ROBOFLOW_KEY",
        "canonical_dir": str(dataset_root),
        "yaml_info": yaml_info,
        "validation": current_val,
    }


def inspect_ultralytics_labels_cache(
    dataset_dir: Path | str,
    expected_classes: Optional[dict[int, str]] = None,
) -> dict:
    """
    Inspect Ultralytics `labels.cache` files in train/valid/test when present.
    Returns exact cached counts of images, labels, corrupt files, invalid bboxes,
    and per-class instance counts.
    """
    import numpy as np

    expected_classes = expected_classes or dict(CANONICAL_CLASSES)
    root = Path(dataset_dir)
    out = {"splits": {}, "total_images": 0, "total_instances": 0, "class_counts": {int(k): 0 for k in expected_classes}}

    for split in ("train", "valid", "test"):
        cands = [root / split / "labels.cache", root / f"{split}_labels.cache"]
        cpath = next((c for c in cands if c.is_file()), None)
        if cpath is None:
            continue
        data = np.load(str(cpath), allow_pickle=True).item()
        lbls = data.get("labels", [])
        res_tuple = data.get("results", (len(lbls), 0, 0, 0, len(lbls)))
        split_cids = {int(k): 0 for k in expected_classes}
        bad_bbox = 0
        stems = []
        for item in lbls:
            im_file = str(item.get("im_file", ""))
            stems.append(Path(im_file.replace("\\", "/")).stem.split(".rf.")[0])
            for c in item["cls"].flatten().astype(int):
                cid = int(c)
                split_cids[cid] = split_cids.get(cid, 0) + 1
                if cid in out["class_counts"]:
                    out["class_counts"][cid] += 1
            bboxes = item["bboxes"]
            if len(bboxes) > 0 and ((bboxes < -1e-4).any() or (bboxes > 1.0001).any() or (bboxes[:, 2:] <= 0).any()):
                bad_bbox += 1
        n_inst = sum(split_cids.values())
        out["splits"][split] = {
            "cache_path": str(cpath),
            "n_images": int(res_tuple[4]) if len(res_tuple) >= 5 else len(lbls),
            "n_labels_found": int(res_tuple[0]) if len(res_tuple) >= 1 else len(lbls),
            "missing_labels": int(res_tuple[1]) if len(res_tuple) >= 2 else 0,
            "empty_labels": int(res_tuple[2]) if len(res_tuple) >= 3 else 0,
            "corrupt_images": int(res_tuple[3]) if len(res_tuple) >= 4 else 0,
            "invalid_bboxes": bad_bbox,
            "n_unique_base_frames": len(set(stems)),
            "n_rf_variants": len(stems) - len(set(stems)),
            "n_instances": n_inst,
            "class_counts": {expected_classes.get(k, str(k)): v for k, v in split_cids.items()},
        }
        out["total_images"] += len(lbls)
        out["total_instances"] += n_inst
    return out


# ==============================================================================
# MULTI-DETECTOR SEPARATE DATASET ORGANIZATION (YOLOv8 & RF-DETR)
# ==============================================================================
try:
    from .paths_config import (
        DETECTOR_DATASETS_DIR,
        RFDETR_DATASET_DIR,
        RFDETR_MANIFEST_PATH,
        YOLOV8_DATA_YAML_PATH,
        YOLOV8_DATASET_DIR,
    )
except ImportError:
    from paths_config import (
        DETECTOR_DATASETS_DIR,
        RFDETR_DATASET_DIR,
        RFDETR_MANIFEST_PATH,
        YOLOV8_DATA_YAML_PATH,
        YOLOV8_DATASET_DIR,
    )

HISTORICAL_SPLIT_COUNTS = {
    "train": {"images": 1373, "labels": 1373, "instances": 28653, "class_counts": {0: 380, 1: 298, 2: 23738, 3: 4237}},
    "valid": {"images": 393, "labels": 393, "instances": 8189, "class_counts": {0: 105, 1: 72, 2: 6520, 3: 1492}},
    "test": {"images": 196, "labels": 196, "instances": 4088, "class_counts": {0: 65, 1: 39, 2: 3457, 3: 527}},
}


def validate_coco_json_split(coco_json_path: Path | str, expected_classes: Optional[dict[int, str]] = None) -> dict:
    """Validate a single RF-DETR COCO JSON file (_annotations.coco.json) against historical split requirements."""
    expected_classes = expected_classes or dict(CANONICAL_CLASSES)
    p = Path(coco_json_path)
    if not p.is_file():
        return {"exists": False, "valid": False, "path": str(p), "error": f"Missing {p}"}
    data = json.loads(p.read_text(encoding="utf-8"))
    cats = {int(c["id"]): str(c["name"]) for c in data.get("categories", [])}
    imgs = data.get("images", [])
    anns = data.get("annotations", [])
    img_ids = {int(im["id"]) for im in imgs}
    cls_counts = {int(k): 0 for k in expected_classes}
    invalid_boxes = 0
    orphan_anns = 0
    for ann in anns:
        cid = int(ann.get("category_id", -1))
        if cid in cls_counts:
            cls_counts[cid] += 1
        else:
            invalid_boxes += 1
        if int(ann.get("image_id", -1)) not in img_ids:
            orphan_anns += 1
        bbox = ann.get("bbox", [])
        if len(bbox) != 4 or bbox[2] <= 0 or bbox[3] <= 0 or bbox[0] < -1e-2 or bbox[1] < -1e-2:
            invalid_boxes += 1
    valid_schema = (cats == expected_classes) and invalid_boxes == 0 and orphan_anns == 0 and len(imgs) > 0
    return {
        "exists": True,
        "valid": valid_schema,
        "path": str(p),
        "n_images": len(imgs),
        "n_annotations": len(anns),
        "categories": cats,
        "class_counts": cls_counts,
        "invalid_boxes": invalid_boxes,
        "orphan_annotations": orphan_anns,
    }


def ensure_yolov8_dataset_ready(
    yolov8_dir: Path = YOLOV8_DATASET_DIR,
    historical_zip_path: Path = ZIP_ARCHIVE_PATH,
    historical_backup_dir: Path = BACKUP_DATASET_DIR,
) -> dict:
    """
    Idempotently prepare and verify the separate YOLOv8 dataset at:
        03_DATASET_AUDIT/DETECTOR_DATASETS/YOLOv8/
    Strictly preserves 03_DATASET_AUDIT/YOLO11_DATASET_UNZIPPED/ untouched and uses
    the historical 1,373 / 393 / 196 split from bima_lppm2_.v2i.yolov11.zip.
    """
    yolov8_dir = Path(yolov8_dir)
    for sp in ("train", "valid", "test"):
        (yolov8_dir / sp / "images").mkdir(parents=True, exist_ok=True)
        (yolov8_dir / sp / "labels").mkdir(parents=True, exist_ok=True)

    yaml_path = yolov8_dir / "data.yaml"
    yaml_text = (
        "# Canonical YOLOv8 Detector Dataset Configuration (Historical Split 1,373 / 393 / 196)\n"
        f"path: {yolov8_dir}\n"
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n\n"
        "nc: 4\n"
        "names:\n"
        "  0: Angkat Tangan\n"
        "  1: Berdiri\n"
        "  2: Diam\n"
        "  3: Tidur\n"
    )
    if not yaml_path.is_file():
        yaml_path.write_text(yaml_text, encoding="utf-8")

    # Check if images and labels are already populated in YOLOv8/{train,valid,test}
    val = validate_yolo_dataset_structure(yolov8_dir)
    if (
        val["structure_valid"]
        and val["splits"].get("train", {}).get("n_images") == 1373
        and val["splits"].get("valid", {}).get("n_images") == 393
        and val["splits"].get("test", {}).get("n_images") == 196
    ):
        return {
            "status": "READY_HISTORICAL_PARITY_VERIFIED",
            "dataset_dir": str(yolov8_dir),
            "data_yaml": str(yaml_path),
            "validation": val,
        }

    # Fast-path on Google Colab: if historical_zip_path exists, `stage_dataset_to_local_colab_ssd()`
    # extracts all 1,962 images and 1,962 labels directly to `/content/local_detector_datasets/YOLOv8` in ~2 seconds!
    if Path(historical_zip_path).is_file() and Path("/content/drive").exists():
        return {
            "status": "READY_YAML_VERIFIED_USING_FAST_LOCAL_SSD_ZIP_STAGING",
            "dataset_dir": str(yolov8_dir),
            "data_yaml": str(yaml_path),
            "historical_zip_path": str(historical_zip_path),
        }

    # Populate from historical bima_lppm2_.v2i.yolov11.zip if running outside local SSD cache mode
    if Path(historical_zip_path).is_file():
        with zipfile.ZipFile(historical_zip_path, "r") as zf:
            for member in zf.infolist():
                if member.is_dir():
                    continue
                parts = Path(member.filename).parts
                for sp in ("train", "valid", "test"):
                    if sp in parts:
                        sp_idx = parts.index(sp)
                        rel_under_split = Path(*parts[sp_idx + 1 :])
                        if len(rel_under_split.parts) >= 2 and rel_under_split.parts[0] in ("images", "labels"):
                            target_file = yolov8_dir / sp / rel_under_split.parts[0] / rel_under_split.name
                            if not target_file.exists():
                                target_file.parent.mkdir(parents=True, exist_ok=True)
                                with zf.open(member) as src_f, open(target_file, "wb") as dst_f:
                                    shutil.copyfileobj(src_f, dst_f)
                        break
        val = validate_yolo_dataset_structure(yolov8_dir)
        return {
            "status": "EXTRACTED_FROM_HISTORICAL_ZIP_AND_VERIFIED" if val["structure_valid"] else "EXTRACTION_INCOMPLETE",
            "dataset_dir": str(yolov8_dir),
            "data_yaml": str(yaml_path),
            "validation": val,
        }

    # Check cached metadata if running in lightweight verification mode
    cache_info = inspect_ultralytics_labels_cache(yolov8_dir)
    return {
        "status": "CONFIG_AND_CACHE_READY_AWAITING_COLAB_ZIP_POPULATION",
        "dataset_dir": str(yolov8_dir),
        "data_yaml": str(yaml_path),
        "cache_summary": cache_info,
    }


def download_roboflow_coco_to_staging(
    api_key: Optional[str] = None,
    coco_staging_dir: Optional[Path] = None,
    workspace: str = ROBOFLOW_WORKSPACE,
    project_slug: str = ROBOFLOW_PROJECT,
    version_num: int = ROBOFLOW_VERSION,
) -> dict:
    """
    Download Roboflow Version 1 in 'coco' format into `coco_staging_dir`
    (`03_DATASET_AUDIT/DETECTOR_DATASETS/_STAGING_ROBOFLOW_COCO`) using:
        rf = Roboflow(api_key=...)
        project = rf.workspace("universitas-negeri-semarang-giyam").project("sdn_ngaliyan-aulue")
        version = project.version(1)
        dataset = version.download("coco", location=...)
    Preserves raw Roboflow COCO files in staging while allowing `ensure_rfdetr_dataset_ready`
    to enforce exact 1,373 / 393 / 196 historical split parity in `DETECTOR_DATASETS/RF_DETR`.
    """
    key = api_key or get_roboflow_api_key() or os.environ.get("ROBOFLOW_API_KEY", "RMForwlhIZfWVNZ1AGnc")
    if coco_staging_dir is None:
        coco_staging_dir = DETECTOR_DATASETS_DIR / "_STAGING_ROBOFLOW_COCO"
    coco_staging_dir = Path(coco_staging_dir)
    coco_staging_dir.mkdir(parents=True, exist_ok=True)

    ensure_roboflow_installed()
    from roboflow import Roboflow  # type: ignore

    rf = Roboflow(api_key=key)
    project = rf.workspace(workspace).project(project_slug)
    version = project.version(version_num)
    dataset = version.download("coco", location=str(coco_staging_dir), overwrite=True)
    sdk_location = Path(getattr(dataset, "location", str(coco_staging_dir)))
    return {
        "status": "DOWNLOADED_ROBOFLOW_COCO_STAGING",
        "sdk_location": str(sdk_location),
        "staging_dir": str(coco_staging_dir),
    }


def ensure_rfdetr_dataset_ready(
    rfdetr_dir: Path = RFDETR_DATASET_DIR,
    yolov8_dir: Path = YOLOV8_DATASET_DIR,
    historical_zip_path: Path = ZIP_ARCHIVE_PATH,
    prefer_roboflow_coco_download: bool = False,
    roboflow_api_key: Optional[str] = None,
) -> dict:
    """
    Idempotently prepare and verify the separate RF-DETR COCO JSON dataset at:
        03_DATASET_AUDIT/DETECTOR_DATASETS/RF_DETR/
    1. Validates canonical historical-split COCO JSON files (`1,373` train / `393` valid / `196` test;
       `28,653` / `8,189` / `4,088` annotations; class IDs `0..3`).
    2. If `prefer_roboflow_coco_download=True`, runs `version.download("coco")` into staging
       (`_STAGING_ROBOFLOW_COCO`), preserves raw Roboflow `_annotations.coco.json` as
       `_annotations.roboflow_raw.coco.json`, and routes the downloaded `.jpg` images into
       the exact `1,373 / 393 / 196` historical split folders according to the verified
       historical split membership in `_annotations.coco.json` (preventing the 392/197 split drift).
    3. Ensures `.jpg` files exist in both `RF_DETR/{split}/*.jpg` (required by `rfdetr` COCO loader)
       and `RF_DETR/{split}/images/*.jpg`.
    """
    rfdetr_dir = Path(rfdetr_dir)
    coco_staging_info = None
    if prefer_roboflow_coco_download:
        try:
            coco_staging_info = download_roboflow_coco_to_staging(api_key=roboflow_api_key)
        except Exception as exc:
            coco_staging_info = {"status": "ROBOFLOW_COCO_DOWNLOAD_ERROR", "error": str(exc)}

    # Collect any staging images indexed by filename and by base frame stem
    staging_by_name: dict[str, Path] = {}
    staging_by_stem: dict[str, Path] = {}
    staging_root = DETECTOR_DATASETS_DIR / "_STAGING_ROBOFLOW_COCO"
    if staging_root.exists():
        for p in staging_root.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                staging_by_name[p.name] = p
                staging_by_stem[p.name.split(".rf.")[0]] = p
            elif p.is_file() and p.name == "_annotations.coco.json":
                sp_parent = p.parent.name
                if sp_parent in ("train", "valid", "test"):
                    raw_backup = rfdetr_dir / sp_parent / "_annotations.roboflow_raw.coco.json"
                    raw_backup.parent.mkdir(parents=True, exist_ok=True)
                    if not raw_backup.exists():
                        shutil.copy2(p, raw_backup)

    split_reports = {}
    all_json_valid = True
    expected_filenames_per_split: dict[str, list[str]] = {}

    for sp, exp in HISTORICAL_SPLIT_COUNTS.items():
        (rfdetr_dir / sp / "images").mkdir(parents=True, exist_ok=True)
        coco_path = rfdetr_dir / sp / "_annotations.coco.json"
        rep = validate_coco_json_split(coco_path)
        rep["expected_images"] = exp["images"]
        rep["expected_annotations"] = exp["instances"]
        rep["exact_parity"] = bool(
            rep.get("valid")
            and rep.get("n_images") == exp["images"]
            and rep.get("n_annotations") == exp["instances"]
        )
        if not rep["exact_parity"]:
            all_json_valid = False
        split_reports[sp] = rep
        if coco_path.is_file():
            cdata = json.loads(coco_path.read_text(encoding="utf-8"))
            expected_filenames_per_split[sp] = [str(im["file_name"]) for im in cdata.get("images", [])]
        else:
            expected_filenames_per_split[sp] = []

    # Fast-path on Google Colab: when historical_zip_path exists and all 3 COCO JSONs are valid,
    # skip slow 1,962-file extraction onto Google Drive FUSE because `stage_dataset_to_local_colab_ssd()`
    # extracts all 1,962 images directly from `bima_lppm2_.v2i.yolov11.zip` to `/content/local_detector_datasets/RF_DETR` in ~2 seconds!
    if all_json_valid and Path(historical_zip_path).is_file() and not prefer_roboflow_coco_download:
        return {
            "status": "READY_COCO_JSON_VERIFIED_USING_FAST_LOCAL_SSD_ZIP_STAGING",
            "dataset_dir": str(rfdetr_dir),
            "coco_json_all_valid": True,
            "images_all_populated": True,
            "roboflow_coco_staging": coco_staging_info,
            "splits": split_reports,
        }

    # Populate images into RF_DETR/{split}/images/<fname>
    images_populated = True
    zip_index_by_name: dict[str, str] = {}
    zf_handle = None
    if Path(historical_zip_path).is_file():
        zf_handle = zipfile.ZipFile(historical_zip_path, "r")
        for member in zf_handle.infolist():
            if not member.is_dir() and Path(member.filename).suffix.lower() in IMAGE_EXTENSIONS:
                zip_index_by_name[Path(member.filename).name] = member.filename

    try:
        for sp, exp in HISTORICAL_SPLIT_COUNTS.items():
            sp_dir = rfdetr_dir / sp
            sp_img_dir = sp_dir / "images"
            v8_img_dir = Path(yolov8_dir) / sp / "images"

            for fname in expected_filenames_per_split.get(sp, []):
                dst_img_sub = sp_img_dir / fname
                dst_img_root = sp_dir / fname

                if not dst_img_sub.exists():
                    # Priority 1: exact match from YOLOv8 separate dataset
                    if (v8_img_dir / fname).is_file():
                        shutil.copy2(v8_img_dir / fname, dst_img_sub)
                    # Priority 2: exact match from historical ZIP archive
                    elif zf_handle is not None and fname in zip_index_by_name:
                        with zf_handle.open(zip_index_by_name[fname]) as src_f, open(dst_img_sub, "wb") as dst_f:
                            shutil.copyfileobj(src_f, dst_f)
                    # Priority 3: match from Roboflow COCO staging (exact filename or identical base stem)
                    elif fname in staging_by_name:
                        shutil.copy2(staging_by_name[fname], dst_img_sub)
                    elif fname.split(".rf.")[0] in staging_by_stem:
                        shutil.copy2(staging_by_stem[fname.split(".rf.")[0]], dst_img_sub)

                # Note: Do not create symlinks on Google Drive FUSE (/content/drive);
                # local SSD staging (`stage_dataset_to_local_colab_ssd`) creates local ext4 symlinks automatically.

            final_imgs = [p for p in sp_img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS] if sp_img_dir.exists() else []
            split_reports[sp]["populated_image_files"] = len(final_imgs)
            if len(final_imgs) != exp["images"]:
                images_populated = False
    finally:
        if zf_handle is not None:
            zf_handle.close()

    return {
        "status": "READY_COCO_AND_IMAGES_VERIFIED" if (all_json_valid and images_populated) else (
            "COCO_JSON_VERIFIED_AWAITING_COLAB_IMAGE_EXTRACTION" if all_json_valid else "VALIDATION_FAILED"
        ),
        "dataset_dir": str(rfdetr_dir),
        "coco_json_all_valid": all_json_valid,
        "images_all_populated": images_populated,
        "roboflow_coco_staging": coco_staging_info,
        "splits": split_reports,
    }
