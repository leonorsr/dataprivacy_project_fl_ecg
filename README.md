# Data Privacy Project — Federated Learning for ECG Arrhythmia Detection

This repository contains notebooks for exploratory data analysis (EDA) and data preparation for the PhysioNet Challenge 2017 and Icentia11k ECG datasets.

## Repository contents

- `DataPreparation/dp_physionet2017.ipynb` — prepares PhysioNet 2017 recordings, normalizes each recording, and creates simulated federated-client train/dev/test partitions.
- `EDA/Physionet2017/EDA_Physionet2017.ipynb` — explores the PhysioNet 2017 dataset.
- `EDA/Icentia11k/EDA_Icentia11k.ipynb` — downloads and explores a reproducible sample of Icentia11k patients.

## Datasets and generated data

Datasets and processed ECG data are **not included** in this repository. The `datasets/` directory and generated PhysioNet preparation outputs are excluded by `.gitignore`. Notebook outputs are cleared to avoid publishing signal plots or patient-derived results.

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
