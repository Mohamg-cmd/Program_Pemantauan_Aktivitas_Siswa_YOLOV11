"""
YOLO11 Classroom Activity Assessment -- Research Pipeline Package (src).
"""
from .paths_config import (
    BACKUP_DATASET_DIR,
    CANONICAL_CLASSES,
    CHECKPOINT_DIR,
    DATA_ROBOFLOW_ORIG_YAML_PATH,
    DATA_YAML_PATH,
    DATASET_AUDIT_DIR,
    DATASET_ROOT,
    MASTER_ROOT,
    MODULES_DIR,
    OUTPUT_DIR,
    PIPELINE_OUTPUT_DIR,
    PRIMARY_CHECKPOINT_PATH,
    REPO_ROOT,
    STAGING_DIR,
    VIDEO_PATH,
    ZIP_ARCHIVE_PATH,
    resolve_project_path,
)

__all__ = [
    "MASTER_ROOT",
    "DATASET_AUDIT_DIR",
    "DATASET_ROOT",
    "STAGING_DIR",
    "BACKUP_DATASET_DIR",
    "ZIP_ARCHIVE_PATH",
    "DATA_YAML_PATH",
    "DATA_ROBOFLOW_ORIG_YAML_PATH",
    "REPO_ROOT",
    "MODULES_DIR",
    "VIDEO_PATH",
    "CHECKPOINT_DIR",
    "PRIMARY_CHECKPOINT_PATH",
    "OUTPUT_DIR",
    "PIPELINE_OUTPUT_DIR",
    "CANONICAL_CLASSES",
    "resolve_project_path",
]
