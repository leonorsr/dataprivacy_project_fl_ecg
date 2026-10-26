# Data Privacy Project — Federated Learning for ECG Arrhythmia Detection

This repository contains notebooks for exploratory data analysis (EDA) and data preparation for the PhysioNet Challenge 2017 and Icentia11k ECG datasets.

## Repository contents

- `DataPreparation/dp_physionet2017.ipynb` — prepares PhysioNet 2017 recordings, normalizes each recording, and creates simulated federated-client train/dev/test partitions.
- `DataPreparation/dp_incetia11k.ipynb` — documents and runs the Icentia11k preprocessing pipeline, including the 16/2/2 patient-level train/dev/test split and conversion to 200 Hz.
- `DataPreparation/FeatureExtraction/dp_feaatureextraction.ipynb` — explores candidate ECG features separately for both prepared datasets. After the initial pilot, it checks natural class prevalence using a random Icentia training sample and complete dev, measures each feature group's incremental contribution, and records proposed feature sets. Test is held out; outputs are saved under `datasets/feature_engineering/`.
- `DataPreparation/FeatureExtraction/build_featured_datasets.py` — materializes all prepared examples as NPZ shards containing aligned ECG signals, selected features, labels and IDs. Outputs are `icentia11k_prepared_featured/` and `physionet2027_prepared_featured/` in that folder; the latter contains PhysioNet **2017**, retaining the requested output name. Original clients and train/dev/test splits are preserved.
- `DataPreparation/FeatureExtraction/featured_dataset.py` — loads the featured datasets by split/client and yields batches without fitting preprocessing on dev/test. Generated featured datasets are excluded from Git.
- `EDA/Physionet2017/EDA_Physionet2017.ipynb` — explores the PhysioNet 2017 dataset.
- `EDA/Icentia11k/EDA_Icentia11k.ipynb` — downloads and explores a reproducible sample of Icentia11k patients.

## Datasets and generated data

Datasets and processed ECG data are **not included** in this repository. The `datasets/` directory and generated PhysioNet and Icentia11k preparation outputs are excluded by `.gitignore`. The preparation and feature-engineering notebooks retain analysis tables and plots; generated signal arrays remain local.

Obtain each dataset directly from its official source, and review and follow its current access, license, citation, and use conditions:

- [PhysioNet Challenge 2017](https://physionet.org/content/challenge-2017/1.0.0/)
- [Icentia11k Continuous ECG](https://physionet.org/content/icentia11k-continuous-ecg/1.0/)

Place downloaded source data under `datasets/` using the paths expected by the relevant notebook. The Icentia11k notebook starts its download only after its explicit confirmation flag is changed; review its disk-space estimate first. Prepared outputs are written locally and remain ignored by Git.

## Setup

Python 3.9 or newer is recommended. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
jupyter lab
```

On Windows, activate the environment with `.venv\Scripts\activate`.

Run notebooks from the repository root so their relative-path discovery can locate `datasets/`.
