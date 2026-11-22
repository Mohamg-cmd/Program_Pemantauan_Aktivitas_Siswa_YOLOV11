# YOLO11-Based Classroom Activity Assessment with Fixed-Seat Identity Resolution and AHP-SAW Decision Support

**Empirical Revalidation, Multi-Detector Benchmark (`YOLO11m`, `YOLOv8m`, `RF-DETR-Small`, `RF-DETR-Large`), Multi-Object Tracker Replay (`BoT-SORT`, `SORT`, `ByteTrack`), 7-Variant Identity Ablation, 13-Scenario Seating Perturbation, Multi-Criteria Decision-Making (`AHP-SAW`, `AHP-TOPSIS`, `AHP-ELECTRE I`, `AHP-WP`), and Within-Session Temporal Robustness Suite**

* **Target Journal:** *International Journal of Information Technology and Computer Science (IJITCS)*
* **Manuscript ID:** `PAPER024568-1`
* **Master Google Drive Research Archive:** [YOLO11_AHP_SAW_IJITCS_REVALIDATION_2026](https://drive.google.com/drive/folders/1ZjjAggTePrkOOoFJcMGL9dbttWHpKXFW)

## 1. Overview and Research Objectives

This repository contains the complete, reproducible source code, Google Colab execution notebooks, dataset parity configurations, and verified empirical outputs for our study on automated classroom student activity assessment from single-camera fixed-viewpoint video (`23,471` frames, `938.84 s` at `25.0 FPS`, `640x480` resolution, `21` seated students `S01` to `S21`).

The framework integrates four sequential computational stages:
1. **Multi-Class Student Posture and Activity Detection:** Fine-grained frame-level detection across four classroom behavioral states:
   * `Class 0`: **Angkat Tangan** (Hand Raising, Active Participation)
   * `Class 1`: **Berdiri** (Standing / Presentation Activity)
   * `Class 2`: **Diam** (Seated Attentive Listening)
   * `Class 3`: **Tidur** (Head-Down / Sleeping / Disengagement)
2. **Multi-Object Trajectory Association:** Frame-to-frame bounding-box tracking via Ultralytics **BoT-SORT** (`with_reid=False, gmc=sparseOptFlow`, Kalman filter + IoU association), benchmarked directly against offline trajectory replay baselines (**SORT** and **ByteTrack**).
3. **Fixed-Seat Spatial-Temporal Identity Resolution (`src/identity.py` and `src/identity_ablation_aligned.py`):** A 7-component deterministic assignment pipeline that resolves fragmented tracker IDs (`290` to `1,399` raw tracks) into exactly `21` persistent student identities (`S01` to `S21`) without biometric facial recognition, preserving student privacy while achieving `93.64%` to `97.24%` primary assignment rates and `98.34%` to `98.77%` seat purity.
4. **Multi-Criteria Decision-Making (`AHP-SAW`) and Statistical Concordance:** Analytic Hierarchy Process (**AHP**) priority weighting combined with Simple Additive Weighting (**SAW**) preference aggregation, cross-validated against **AHP-TOPSIS**, **AHP-ELECTRE I**, **AHP-WP**, and **Equal-Weight SAW** with Fisher-$z$ `95%` confidence intervals and Holm-Bonferroni family-wise error correction.

## 2. Repository Directory Structure

```text
YOLO11_AHP_SAW_IJITCS_REVALIDATION/
├── README.md                                      # Complete architecture, benchmarks, and usage guide
├── config.yaml                                    # Master pipeline hyperparameters and AHP matrix configuration
├── requirements.txt                               # Pinned Python package dependencies
├── .gitignore                                     # Standard Git ignore rules for weights and caches
├── run_pipeline.py                                # End-to-end CLI pipeline runner (Stages 1 to 10)
├── run_experiments.py                             # Automated ablation, sensitivity, and MCDM experiment runner
├── focal_trainer.py                               # Custom ElementwiseFocalLoss trainer for Ultralytics YOLO11
├── train_yolo11_baseline.py                       # Training script: YOLO11m Standard BCE (150 epochs, 1x sampling)
├── train_yolo11_focal.py                          # Training script: YOLO11m Focal Loss (gamma=2.0, alpha=0.25)
├── train_yolo11_balanced.py                       # Training script: YOLO11m Focal Loss + 3x minority oversampling
├── data/
│   ├── student_seats.csv                          # Calibrated 21-seat anchor coordinates and radius thresholds
│   ├── data_colab.yaml                            # Canonical YOLO11 dataset split configuration (1,373/393/196)
│   └── DETECTOR_DATASET_SPLIT_PARITY_REPORT.csv   # Verified split & class parity across YOLO11, YOLOv8, RF-DETR
├── notebooks/
│   ├── 02_YOLO11_RETRAINING_IF_CHECKPOINT_MISSING.ipynb
│   ├── 03_Detector_Evaluation_WORKING_COPY.ipynb
│   ├── 04_Video_Inference_Tracking_WORKING_COPY.ipynb
│   ├── 08_Baseline_Ablation_Analysis_WORKING_COPY.ipynb
│   └── 10_Within_Session_Temporal_Robustness_WORKING_COPY.ipynb
├── src/
│   ├── __init__.py                                # Package exports
│   ├── paths_config.py                            # Centralized path resolver & multi-detector registry
│   ├── dataset_manager.py                         # Roboflow/COCO dataset validator, auto-repair & parity engine
│   ├── detector_adapters_v2_aligned.py            # Multi-detector benchmark & 23,471-frame video inference suite
│   ├── tracker_adapters_aligned.py                # Offline SORT & ByteTrack trajectory replay engines
│   ├── mcdm_topsis_electre_adapters.py            # AHP-TOPSIS, AHP-ELECTRE I, AHP-WP & Equal-SAW engines
│   ├── identity_ablation_aligned.py               # 7-variant identity ablation & 13-scenario seating perturbation
│   ├── temporal_consistency_aligned.py            # Within-session 3x300s direct frame-counting & IPF comparison
│   ├── detect_track.py                            # Ultralytics BoT-SORT inference & track summary builder
│   ├── identity.py                                # Core Fixed-Seat Spatial-Temporal Identity Resolution engine
│   ├── aggregate.py                               # 25 FPS student activity duration & frequency matrix builder
│   ├── ahp.py                                     # Saaty AHP eigenvector weight derivation & CR verification
│   ├── saw.py                                     # Simple Additive Weighting (Benefit/Cost normalization)
│   ├── wp.py                                      # Weighted Product (WP) multiplicative MCDM baseline
│   ├── sensitivity.py                             # One-at-a-Time (OAT) weight perturbation analysis
│   ├── monte_carlo.py                             # 1,000-iteration Dirichlet Monte Carlo rank stability simulator
│   ├── ablation.py                                # Single-detector component ablation runner
│   ├── temporal_analysis.py                       # Segment-level temporal correlation utilities
│   ├── error_propagation.py                       # Track-to-seat error propagation audit
│   ├── evaluate_detector.py                       # Per-class Precision, Recall, F1, mAP50, mAP50-95 evaluator
│   ├── mot_eval.py                                # Multi-object tracking fragmentation & coverage metrics
│   ├── provenance.py                              # SHA-256 checkpoint & dataset provenance verifier
│   ├── rank_metrics.py                            # Spearman rho, Kendall tau, Fisher-z CI & Holm-Bonferroni tests
│   ├── report.py                                  # Automated Markdown & CSV report generator
│   └── visual_audit.py                            # Annotated classroom seat-map & trajectory overlay plotter
└── results/
    ├── static_benchmarks/                         # Validation (393) & Test (196) metrics across all 4 detectors
    ├── tracker_replay/                            # 23,471-frame BoT-SORT, SORT & ByteTrack comparisons
    ├── mcdm_and_ablation/                         # MCDM concordance, 7-variant ablation & 13 seating scenarios
    └── temporal_robustness/                       # Within-session (5m/10m/15m/Full) rankings & IPF comparisons
```

## 3. Dataset Split Parity and Class Distribution

To guarantee strict experimental fairness across CNN (`YOLO11m`, `YOLOv8m`) and Vision Transformer (`RF-DETR-Small`, `RF-DETR-Large`) architectures, `src/dataset_manager.py` enforces identical image-level and instance-level partitions across all three dataset formats (`YOLO11` Ultralytics TXT, `YOLOv8` Ultralytics TXT, and `RF-DETR` MS-COCO JSON):

| Partition Split | Image Count | Total Instances | Class 0 (`Angkat Tangan`) | Class 1 (`Berdiri`) | Class 2 (`Diam`) | Class 3 (`Tidur`) | SHA-256 Parity |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train (`70.0%`)** | `1,373` | `28,653` | `380` | `298` | `23,738` | `4,237` | `100% Verified` |
| **Validation (`20.0%`)** | `393` | `8,189` | `105` | `72` | `6,520` | `1,492` | `100% Verified` |
| **Held-Out Test (`10.0%`)** | `196` | `4,088` | `65` | `39` | `3,457` | `527` | `100% Verified` |
| **Total (`100.0%`)** | **`1,962`** | **`40,930`** | **`550`** | **`409`** | **`33,715`** | **`6,256`** | **`100% Verified`** |

## 4. Key Empirical Results Across All Four Detectors

### 4.1 Static Detector Benchmark (Notebook 03: Validation `393` Images and Test `196` Images)

| Detector Variant | Loss & Sampling Strategy | Split | Precision | Recall | F1-Score | mAP@50 | mAP@50-95 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **YOLO11m Baseline** | Standard BCE (`1x` sampling, `150` ep) | Val (`393`) | `0.9394` | `0.9425` | `0.9409` | `0.9757` | `0.8251` |
| **YOLO11m Baseline** | Standard BCE (`1x` sampling, `150` ep) | **Test (`196`)** | **`0.9600`** | **`0.9384`** | **`0.9491`** | **`0.9774`** | **`0.8294`** |
| **YOLO11m Focal** | `ElementwiseFocalLoss(gamma=2.0, alpha=0.25)` | Val (`393`) | `0.9384` | `0.9275` | `0.9329` | `0.9771` | `0.8201` |
| **YOLO11m Focal** | `ElementwiseFocalLoss(gamma=2.0, alpha=0.25)` | **Test (`196`)** | **`0.9473`** | **`0.9422`** | **`0.9447`** | **`0.9766`** | **`0.8259`** |
| **YOLO11m Balanced** | Focal Loss + `3x` Minority Oversampling (`71` ep) | Val (`393`) | `0.9412` | `0.9358` | `0.9385` | `0.9802` | `0.8195` |
| **YOLO11m Balanced** | Focal Loss + `3x` Minority Oversampling (`71` ep) | **Test (`196`)** | **`0.9557`** | **`0.9445`** | **`0.9501`** | **`0.9789`** | **`0.8286`** |
| **YOLOv8m Baseline** | Standard BCE (`1x` sampling, `50` ep) | Val (`393`) | `0.9509` | `0.9439` | `0.9474` | `0.9805` | `0.8372` |
| **YOLOv8m Baseline** | Standard BCE (`1x` sampling, `50` ep) | **Test (`196`)** | **`0.9615`** | **`0.9345`** | **`0.9478`** | **`0.9737`** | **`0.8336`** |
| **RF-DETR-Small (`S`)** | DINOv2 + DETR Set-Prediction (`20` ep) | Val (`393`) | `0.9699` | `0.9939` | `0.9817` | `0.9699` | `0.7953` |
| **RF-DETR-Small (`S`)** | DINOv2 + DETR Set-Prediction (`20` ep) | **Test (`196`)** | **`0.9798`** | **`0.9978`** | **`0.9887`** | **`0.9798`** | **`0.8130`** |
| **RF-DETR-Large (`L`)** | DINOv2 + DETR Set-Prediction (`20` ep) | Val (`393`) | `0.9771` | `0.9979` | `0.9874` | `0.9771` | `0.7942` |
| **RF-DETR-Large (`L`)** | DINOv2 + DETR Set-Prediction (`20` ep) | **Test (`196`)** | **`0.9792`** | **`0.9977`** | **`0.9883`** | **`0.9792`** | **`0.8126`** |

### 4.2 Full-Video (`23,471` Frames) Multi-Object Tracker Replay (Notebook 04)

| Detector Model | Tracker Method | Tracked Detections | Unique Tracks | Primary Assigned (`%`) | Seat Purity (`%`) | Frame Coverage (`%`) | Long-Gap Fragments | Spearman $ho$ vs BoT-SORT | Kendall $	au$ | Top-5 Overlap |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **yolo11m** | `botsort_cached` | `497,848` | `1,399` | `484,093 (97.24%)` | `98.77%` | `98.22%` | `12` | `1.0000` | `1.0000` | `1.00` |
| **yolo11m** | `sort` | `497,848` | `720` | `481,596 (96.74%)` | `99.07%` | `97.71%` | `15` | `0.9662` | `0.8857` | `1.00` |
| **yolo11m** | `bytetrack` | `491,694` | `417` | `483,660 (98.37%)` | `98.76%` | `98.13%` | `18` | `0.9714` | `0.8857` | `1.00` |
| **yolov8m** | `botsort_cached` | `506,311` | `1,056` | `486,554 (96.10%)` | `98.71%` | `98.71%` | `9` | `1.0000` | `1.0000` | `1.00` |
| **yolov8m** | `sort` | `506,311` | `613` | `486,572 (96.10%)` | `98.71%` | `98.72%` | `9` | `0.9909` | `0.9619` | `1.00` |
| **yolov8m** | `bytetrack` | `496,593` | `305` | `486,301 (97.93%)` | `98.71%` | `98.66%` | `7` | `0.9948` | `0.9619` | `1.00` |
| **rfdetr_small** | `botsort_cached` | `512,578` | `292` | `487,702 (95.15%)` | `98.34%` | `98.95%` | `5` | `1.0000` | `1.0000` | `1.00` |
| **rfdetr_small** | `sort` | `512,578` | `311` | `490,393 (95.67%)` | `98.76%` | `99.49%` | `3` | `0.9831` | `0.9238` | `1.00` |
| **rfdetr_small** | `bytetrack` | `512,578` | `292` | `487,702 (95.15%)` | `98.34%` | `98.95%` | `5` | `1.0000` | `1.0000` | `1.00` |
| **rfdetr_large** | `botsort_cached` | `519,753` | `290` | `486,707 (93.64%)` | `98.77%` | `98.75%` | `0` | `1.0000` | `1.0000` | `1.00` |
| **rfdetr_large** | `sort` | `519,753` | `308` | `489,279 (94.14%)` | `99.15%` | `99.27%` | `0` | `0.9896` | `0.9429` | `1.00` |
| **rfdetr_large** | `bytetrack` | `519,753` | `290` | `486,707 (93.64%)` | `98.77%` | `98.75%` | `0` | `1.0000` | `1.0000` | `1.00` |

### 4.3 MCDM Comparative Concordance Across All Four Detectors (Notebook 08)

Using the verified AHP priority vector $w = [0.5115, 0.2427, 0.1494, 0.0964]^T$ ($\lambda_{\max} = 4.1016, 	ext{CI} = 0.0339, 	ext{CR} = 0.0376 < 0.10$), `AHP-SAW` exhibits high statistical concordance with `AHP-TOPSIS` and `AHP-ELECTRE I` across all four detectors:

| Detector Model | vs `AHP-TOPSIS` ($ho / 	au$) | vs `AHP-ELECTRE I` ($ho / 	au$) | vs `AHP-WP` ($ho / 	au$) | vs `Equal-SAW` ($ho / 	au$) | Top-5 Ranked Students (`AHP-SAW`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **yolo11m** | `0.9597 / 0.8667` | `0.9013 / 0.7810` | `0.8597 / 0.6952` | `0.7364 / 0.5905` | `S03, S21, S15, S11, S19` |
| **yolov8m** | `0.9623 / 0.8571` | `0.9052 / 0.7714` | `0.7610 / 0.6190` | `0.8701 / 0.7333` | `S15, S03, S11, S05, S04` |
| **rfdetr_small** | `0.9844 / 0.9143` | `0.8740 / 0.7524` | `0.8610 / 0.7619` | `0.9143 / 0.7619` | `S11, S15, S19, S03, S04` |
| **rfdetr_large** | `0.9948 / 0.9714` | `0.9571 / 0.8476` | `0.7117 / 0.6095` | `0.7792 / 0.6476` | `S11, S03, S15, S12, S19` |

### 4.4 Within-Session Temporal Robustness (`3 x 300s + 38.84s` Residual, Notebook 10)

| Detector Model | `Cum 5m vs 10m` ($ho$, $p_{	ext{Holm}}$) | `Cum 10m vs 15m` ($ho$, $p_{	ext{Holm}}$) | `Cum 15m vs Full` ($ho$, $p_{	ext{Holm}}$) | `Cum 15m vs Full` ($	au$) | Mean / Max Rank Shift (`15m vs Full`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **yolo11m** | `0.7857` ($1.70 	imes 10^{-4}$) | `0.9052` ($1.37 	imes 10^{-7}$) | **`0.9935`** ($1.90 	imes 10^{-18}$) | `0.9619` | `0.38 / 1` |
| **yolov8m** | `0.7208` ($1.59 	imes 10^{-3}$) | `0.9416` ($1.60 	imes 10^{-9}$) | **`0.9909`** ($4.60 	imes 10^{-17}$) | `0.9524` | `0.48 / 2` |
| **rfdetr_small** | `0.6896` ($3.80 	imes 10^{-3}$) | `0.9117` ($7.19 	imes 10^{-8}$) | **`0.9948`** ($2.29 	imes 10^{-19}$) | `0.9714` | `0.29 / 1` |
| **rfdetr_large** | `0.5610` ($4.89 	imes 10^{-2}$) | `0.8104` ($6.75 	imes 10^{-5}$) | **`0.9961`** ($1.50 	imes 10^{-20}$) | `0.9714` | `0.29 / 1` |

## 5. Execution Guide (Google Colab & Local CLI)

### 5.1 Google Colab Execution Order
1. **`notebooks/02_YOLO11_RETRAINING_IF_CHECKPOINT_MISSING.ipynb`**: Verifies SHA-256 integrity of historical `YOLO11m` checkpoints (`baseline`, `focal`, `balanced`) and extracts `YOLO11_DATASET_UNZIPPED`.
2. **`notebooks/03_Detector_Evaluation_WORKING_COPY.ipynb`**: Evaluates `YOLO11m` (`3` variants), trains/evaluates `YOLOv8m`, `RF-DETR-Small`, and `RF-DETR-Large` on the identical `1,373 / 393 / 196` split (`ACTIVE_BENCHMARK_MODELS = "all"`).
3. **`notebooks/04_Video_Inference_Tracking_WORKING_COPY.ipynb`**: Runs `23,471`-frame video inference and offline `BoT-SORT`, `SORT`, and `ByteTrack` trajectory replay (`ACTIVE_DETECTOR_MODEL = "all_available"`).
4. **`notebooks/08_Baseline_Ablation_Analysis_WORKING_COPY.ipynb`**: Executes `AHP-TOPSIS` / `AHP-ELECTRE I` concordance, `7`-variant identity resolution ablation, and `13`-scenario seating arrangement perturbation (`ACTIVE_DETECTOR_MODEL = "all_available"`).
5. **`notebooks/10_Within_Session_Temporal_Robustness_WORKING_COPY.ipynb`**: Computes direct frame-counted (`3 x 300s + 38.84s`) within-session temporal convergence and IPF error analysis (`ACTIVE_DETECTOR_MODEL = "all_available"`).

### 5.2 Command-Line Interface (CLI) Execution
```bash
pip install -r requirements.txt
python run_pipeline.py --config config.yaml --stage all
python run_experiments.py --config config.yaml --output-dir results/
```
