# Multi-User HAR Framework

This repository supports the paper:

**Hybrid Classification for Complex and Composite Activity Recognition in Multi-User Environments**

The project provides a reproducible pipeline for working with a synthetic multi-user wearable Human Activity Recognition (HAR) dataset and baseline models for atomic activity recognition, composite activity recognition, interaction recognition, and user attribution.

## Overview

Real-world human activity recognition often involves multiple users, concurrent activities, ambiguous user attribution, sensor dropout, synchronization drift, and hierarchical/composite activities. This repository provides:

- Dataset loading utilities
- Preprocessing and sliding-window segmentation
- Baseline models
- Evaluation metrics
- Train/validation/test split handling
- Example scripts for reproducible experiments

## Repository Structure

```text
multi-user-har-framework/
│
├── data/
│   └── README.md
│
├── src/
│   ├── config.py
│   ├── dataset.py
│   ├── preprocessing.py
│   ├── features.py
│   ├── baseline_models.py
│   ├── train_baseline.py
│   ├── evaluate.py
│   └── utils.py
│
├── notebooks/
│   └── quick_start.ipynb
│
├── figures/
│   └── README.md
│
├── paper/
│   └── citation.bib
│
├── results/
│   └── README.md
│
├── requirements.txt
├── LICENSE
├── CITATION.cff
└── README.md
```

## Dataset

The dataset is distributed separately as a ZIP package:

**Multi-User Wearable HAR Dataset for Complex and Composite Activity Recognition**

Expected dataset files:

```text
raw_imu_samples.csv
session_metadata.csv
activity_segments.csv
session_splits.csv
label_dictionary.json
preview_first_2000_rows.csv
README.md
```

Place the dataset files inside the `data/` folder before running experiments.

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Place dataset files

Copy the generated dataset CSV files into:

```bash
data/
```

### 3. Train baseline model

```bash
python src/train_baseline.py --data_dir data --task atomic_activity

# For a quick smoke test using preview_first_2000_rows.csv renamed as raw_imu_samples.csv:
python src/train_baseline.py --data_dir data --task atomic_activity --quick --window_size 50 --step 25
```

### 4. Evaluate

```bash
python src/evaluate.py --data_dir data --predictions results/predictions_atomic_activity.csv
```

## Supported Tasks

The code supports the following label targets:

- `atomic_activity`
- `composite_activity`
- `interaction_type`
- `user_id`

Example:

```bash
python src/train_baseline.py --data_dir data --task composite_activity
python src/train_baseline.py --data_dir data --task interaction_type
```

## Baseline Model

The included baseline is a classical machine-learning pipeline:

1. Sliding-window segmentation
2. Statistical feature extraction
3. Random Forest classifier
4. Macro-F1, weighted-F1, accuracy, and classification report

This baseline is intentionally lightweight and reproducible. It is useful for verifying dataset integrity and establishing a benchmark before implementing deep models such as CNN--BiLSTM, Transformers, GNNs, or neuro-symbolic pipelines.



## Transformer Baseline

A PyTorch Transformer encoder baseline is included for sequence-level HAR classification.

Quick smoke test using a preview dataset:

```bash
python src/train_transformer.py --data_dir data --task atomic_activity --quick --epochs 2 --window_size 50 --step 25
```

Full dataset training:

```bash
python src/train_transformer.py --data_dir data --task atomic_activity --epochs 10
python src/train_transformer.py --data_dir data --task composite_activity --epochs 10
python src/train_transformer.py --data_dir data --task interaction_type --epochs 10
```

Transformer outputs are saved to `results/`:

```text
metrics_transformer_<task>.json
predictions_transformer_<task>.csv
classification_report_transformer_<task>.txt
transformer_<task>.pt
transformer_scaler_<task>.joblib
transformer_label_encoder_<task>.joblib
```

## Paper Method Summary

The paper proposes a hybrid HAR framework integrating:

- CNN--BiLSTM temporal representation learning
- Attention-based sensor fusion
- Graph Neural Network interaction modeling
- Probabilistic symbolic reasoning for composite activities

## Reproducibility Notes

- The dataset uses session-level splits to avoid leakage.
- Train/validation/test sessions are defined in `session_splits.csv`.
- Baselines aggregate raw IMU samples into fixed-size windows.
- Missing values caused by simulated dropout are imputed during preprocessing.

## Citation

If you use this dataset or code, please cite:

```bibtex
@misc{dursunoglu2026multiuserhar,
  author = {Dursunoglu, Halil Ibrahim},
  title = {Synthetic Multi-User Wearable HAR Dataset and Hybrid Classification Framework},
  year = {2026},
  publisher = {Zenodo},
  doi = {TO_BE_ADDED}
}
```

## License

This repository is released under the MIT License. The dataset may be released under CC BY 4.0.

## Contact

Halil Ibrahim Dursunoglu  
Department of Computer Science  
Western Michigan University  
Email: halilibrahim.dursunoglu@wmich.edu
