"""Empacota os sinais preparados e as features selecionadas, sem refazer os splits.

Executar da raiz: .venv/bin/python DataPreparation/FeatureExtraction/build_featured_datasets.py
Os NPZ são particionados por registo (Icentia) ou cliente (PhysioNet).
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import time

# Cada processo usa um único thread numérico.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import pandas as pd
from scipy import signal as sp_signal
from scipy.stats import skew, kurtosis
from importlib.metadata import version

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ICENTIA = ROOT / "datasets" / "icentia11_prepared"
PHYSIO = ROOT / "datasets" / "physionet2017_prepared_normalized"
PROPOSALS = ROOT / "datasets" / "feature_engineering" / "natural_prevalence" / "proposed_feature_sets.json"
FS = 200
SCHEMA = 1
BASE = ["mean", "std", "peak_to_peak", "rms", "iqr"]
SHAPE = ["skewness", "kurtosis", "mean_abs_difference", "max_abs_slope", "total_variation_per_second"]
SPECTRAL = ["spectral_entropy", "dominant_frequency_hz", "power_fraction_0_5_5", "power_fraction_5_15", "power_fraction_15_40"]
AMPLITUDE = ["source_peak_to_peak", "source_max_abs"]
RR = ["rr_median_ms", "rr_iqr_ms", "rr_rmssd_ms", "rr_interval_count"]
LABELS = {"icentia11k": {"N": 0, "S": 1, "V": 2, "Q": 3},
          "physionet2017": {"N": 0, "A": 1, "O": 2, "~": 3}}
OUTPUTS = {"icentia11k": HERE / "icentia11k_prepared_featured",
           "physionet2017": HERE / "physionet2027_prepared_featured"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, data):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def batch_signal_features(signals, fs=FS):
    """Mesmas definições do notebook, vetorizadas por linha para os batimentos."""
    x = np.asarray(signals, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] < 4 or not np.isfinite(x).all():
        raise ValueError("Sinais inválidos para a extração de features")
    std = x.std(axis=1)
    diff = np.diff(x, axis=1)
    quantiles = np.quantile(x, [.25, .75], axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        sk = skew(x, axis=1)
        ku = kurtosis(x, axis=1, fisher=False)
    sk[std <= 1e-12] = np.nan
    ku[std <= 1e-12] = np.nan
    out = {
        "mean": x.mean(axis=1), "std": std, "peak_to_peak": np.ptp(x, axis=1),
        "rms": np.sqrt(np.mean(x*x, axis=1)), "iqr": quantiles[1]-quantiles[0],
        "skewness": sk, "kurtosis": ku,
        "mean_abs_difference": np.abs(diff).mean(axis=1),
        "max_abs_slope": np.abs(diff).max(axis=1)*fs,
        "total_variation_per_second": np.abs(diff).sum(axis=1)/(diff.shape[1]/fs),
    }
    frequencies, psd = sp_signal.welch(x, fs=fs, nperseg=min(256, x.shape[1]), detrend="constant", axis=1)
    valid = (frequencies >= .5) & (frequencies <= 40)
    f, power = frequencies[valid], psd[:, valid]
    total = power.sum(axis=1)
    good = total > np.finfo(float).eps
    probabilities = np.divide(power, total[:, None], out=np.zeros_like(power), where=good[:, None])
    entropy = -np.sum(probabilities*np.log(np.clip(probabilities, np.finfo(float).tiny, None)), axis=1)/np.log(len(f))
    out["spectral_entropy"] = np.where(good, entropy, np.nan)
    out["dominant_frequency_hz"] = np.where(good, f[np.argmax(power, axis=1)], np.nan)
    for name, low, high in [("0_5_5", .5, 5), ("5_15", 5, 15), ("15_40", 15, 40.0001)]:
        numerator = power[:, (f >= low) & (f < high)].sum(axis=1)
        out[f"power_fraction_{name}"] = np.divide(numerator, total, out=np.full(len(x), np.nan), where=good)
    return out


def scalar_features(values, fs=FS, include_rr=False):
    """Extrator do piloto; preserva também os RR heurísticos selecionados."""
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    out = {key: float(value[0]) for key, value in batch_signal_features(x[None, :], fs).items()}
    if include_rr:
        filtered = sp_signal.sosfiltfilt(sp_signal.butter(3, [5, 20], fs=fs, btype="bandpass", output="sos"), x)
        envelope = np.abs(filtered)
        peaks, _ = sp_signal.find_peaks(envelope, distance=int(.3*fs), prominence=max(.5*envelope.std(), 1e-9))
        rr = np.diff(peaks)/fs*1000
        out.update({"rr_median_ms": float(np.median(rr)) if len(rr) else np.nan,
                    "rr_iqr_ms": float(np.quantile(rr, .75)-np.quantile(rr, .25)) if len(rr) else np.nan,
                    "rr_rmssd_ms": float(np.sqrt(np.mean(np.diff(rr)**2))) if len(rr)>1 else np.nan,
                    "rr_interval_count": float(len(rr))})
    return out


def beat_feature_grid(X):
    # Mesmos intervalos [0, 1] usados por np.interp no notebook de feature engineering.
    xp, target = np.linspace(0, 1, X.shape[1]), np.linspace(0, 1, FS)
    left = np.clip(np.searchsorted(xp, target, side="right")-1, 0, len(xp)-2)
    weight = (target-xp[left])/(xp[left+1]-xp[left])
    a, b = X[:, left].astype(np.float64), X[:, left+1].astype(np.float64)
    return a + (b-a)*weight


def build_shard(job):
    dataset = job["dataset"]
    source = ICENTIA if dataset == "icentia11k" else PHYSIO
    output = OUTPUTS[dataset]
    target = output / job["shard_path"]
    marker = target.with_suffix(".json")
    signature = hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest()
    if target.is_file() and marker.is_file():
        saved = json.loads(marker.read_text())
        if saved.get("signature") == signature and saved.get("sha256") == sha256(target):
            return saved
    target.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(source / job["source_manifest"])
    names = job["feature_names"]
    if dataset == "icentia11k":
        assert frame["patient_id"].eq(job["patient_id"]).all()
        assert frame["split"].eq(job["split"]).all()
        X = np.empty((len(frame), 500), dtype=np.float32)
        labels = frame["label"].to_numpy(dtype=str)
        y = np.asarray([LABELS[dataset][value] for value in labels], dtype=np.int64)
        for index, row in enumerate(frame.itertuples(index=False)):
            with np.load(source / row.segment_path, allow_pickle=False) as data:
                beat = data["signal"]
                if beat.shape != (500,) or beat.dtype != np.float32 or str(data["label"]) != row.label or int(data["label_id"]) != int(y[index]):
                    raise ValueError(f"Sinal/label incompatível: {row.segment_path}")
                X[index] = beat
        values = batch_signal_features(beat_feature_grid(X))
        values.update({"source_peak_to_peak": frame["raw_peak_to_peak"].to_numpy(), "source_max_abs": frame["raw_max_abs"].to_numpy()})
        features = np.column_stack([values[name] for name in names]).astype(np.float32)
        sample_ids = frame["segment_path"].to_numpy(dtype=str)
        valid_lengths = np.full(len(X), 500, dtype=np.int32)
        extra = {"peak_index_200hz": frame["peak_index_200hz"].to_numpy(dtype=np.int64),
                 "annotation_index_200hz": frame["annotation_index_200hz"].to_numpy(dtype=np.int64),
                 "annotation_alignment_error_samples": frame["annotation_alignment_error_samples"].to_numpy(dtype=np.int64),
                 "quality": frame["quality_zhao_fuzzy"].to_numpy(dtype=str)}
    else:
        assert frame["client"].eq(job["client_id"]).all() and frame["split"].eq(job["split"]).all()
        with np.load(source / f"{job['client_id']}.npz", allow_pickle=False) as data:
            X = data["X"]
            labels, y, sample_ids = data["labels"], data["y"], data["record_ids"]
        assert np.array_equal(frame["record"].to_numpy(dtype=str), sample_ids)
        assert np.array_equal(frame["label"].to_numpy(dtype=str), labels)
        assert np.array_equal(frame["label_id"].to_numpy(dtype=np.int64), y)
        features = np.empty((len(X), len(names)), dtype=np.float32)
        valid_lengths = np.empty(len(X), dtype=np.int32)
        for index, record in enumerate(sample_ids):
            header = (ROOT / "datasets" / "training2017" / f"{record}.hea").read_text().splitlines()[0].split()
            valid_lengths[index] = min(X.shape[1], int(np.ceil(int(header[3])*FS/float(header[2]))))
            values = scalar_features(X[index, :valid_lengths[index]], include_rr=True)
            features[index] = [values[name] for name in names]
        extra = {}
    assert len(X) == job["n_examples"] and len(sample_ids) == len(set(sample_ids.tolist()))
    assert X.dtype == np.float32 and np.isfinite(X).all() and not np.isinf(features).any()
    assert features.shape == (len(X), len(names))
    feature_missing_mask = np.isnan(features)
    tmp = target.with_name(target.name + ".tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, X=X, features=features, feature_missing_mask=feature_missing_mask,
                            feature_names=np.asarray(names, dtype=str), y=y, labels=np.asarray(labels, dtype=str),
                            sample_ids=np.asarray(sample_ids, dtype=str), valid_lengths=valid_lengths,
                            dataset=np.asarray(dataset), split=np.asarray(job["split"]),
                            client_id=np.asarray(job["client_id"]), patient_id=np.asarray(job["patient_id"]), **extra)
    tmp.replace(target)
    # Validar o que ficou realmente guardado, incluindo os sinais e labels completos.
    with np.load(target, allow_pickle=False) as saved:
        assert np.array_equal(saved["X"], X) and np.array_equal(saved["y"], y)
        assert np.array_equal(saved["features"], features, equal_nan=True)
        assert np.array_equal(saved["sample_ids"], sample_ids)
        assert np.array_equal(saved["feature_missing_mask"], feature_missing_mask)
    result = {key: job[key] for key in ["dataset", "shard_path", "split", "client_id", "patient_id", "record_id", "n_examples"]}
    result.update({"signature": signature, "sha256": sha256(target), "signal_samples": X.shape[1],
                   "label_counts": {label: int(np.sum(labels == label)) for label in LABELS[dataset]},
                   "feature_missing_counts": {name: int(feature_missing_mask[:, i].sum()) for i, name in enumerate(names)}})
    atomic_json(marker, result)
    return result


def make_jobs(dataset, proposals):
    source = ICENTIA if dataset == "icentia11k" else PHYSIO
    config_name = "preparation_config.json" if dataset == "icentia11k" else "preprocessing_config.json"
    config = json.loads((source / config_name).read_text())
    common = {"dataset": dataset, "feature_names": proposals[dataset]["features"],
              "builder_sha256": sha256(__file__), "source_config_sha256": sha256(source / config_name),
              "versions": {name: version(name) for name in ["numpy", "scipy", "pandas"]}}
    jobs = []
    if dataset == "icentia11k":
        stats = pd.read_csv(source / "record_processing_stats.csv")
        assert not stats.duplicated(["patient_id", "record_id"]).any()
        for row in stats.itertuples(index=False):
            client = config["patient_to_client"][row.patient_id]
            assert row.patient_id in config["patient_ids"][row.split]
            manifest = Path("record_checkpoints") / row.split / row.patient_id / f"{row.record_id}.csv"
            jobs.append({**common, "source_manifest": str(manifest), "source_manifest_sha256": sha256(source / manifest),
                         "shard_path": str(Path(row.split) / client / f"{row.record_id}.npz"),
                         "split": row.split, "client_id": client, "patient_id": row.patient_id,
                         "record_id": row.record_id, "n_examples": int(row.accepted)})
    else:
        manifest = pd.read_csv(source / "manifest.csv")
        assert manifest.record.is_unique
        for client, frame in manifest.groupby("client", sort=True):
            split = config["client_roles"][client]
            assert frame["split"].eq(split).all()
            jobs.append({**common, "source_manifest": f"{client}.csv", "source_manifest_sha256": sha256(source / f"{client}.csv"),
                         "shard_path": str(Path(split) / client / f"{client}.npz"), "split": split,
                         "client_id": client, "patient_id": "", "record_id": "multiple_records",
                         "n_examples": len(frame)})
    return jobs, config


def finalize(dataset, results, proposals, source_config):
    output = OUTPUTS[dataset]
    rows = sorted(results, key=lambda r: (r["split"], r["client_id"], r["record_id"]))
    manifest = pd.DataFrame([{key: row[key] for key in ["shard_path", "split", "client_id", "patient_id", "record_id", "n_examples", "signal_samples", "sha256"]} for row in rows])
    assert not manifest.shard_path.duplicated().any()
    assert manifest.groupby("client_id")["split"].nunique().eq(1).all()
    expected = 4865610 if dataset == "icentia11k" else 8528
    assert int(manifest.n_examples.sum()) == expected
    label_rows = [{"split": row["split"], "client_id": row["client_id"], "patient_id": row["patient_id"], **row["label_counts"]} for row in rows]
    client_counts = pd.DataFrame(label_rows).groupby(["split", "client_id", "patient_id"], dropna=False)[list(LABELS[dataset])].sum()
    client_counts["Total"] = client_counts.sum(axis=1)
    if dataset == "icentia11k":
        assert manifest.groupby("client_id").patient_id.nunique().eq(1).all()
        assert manifest.groupby("patient_id").client_id.nunique().eq(1).all()
        expected_split_counts = {"train": 3904222, "dev": 470000, "test": 491388}
    else:
        expected_split_counts = {"train": 6398, "dev": 1065, "test": 1065}
    assert manifest.groupby("split").n_examples.sum().to_dict() == expected_split_counts
    manifest.to_csv(output / "manifest.csv", index=False)
    client_counts.to_csv(output / "client_class_counts.csv")
    for split in ["train", "dev", "test"]:
        manifest[manifest.split == split].to_csv(output / f"{split}_manifest.csv", index=False)
    missing = {name: sum(row["feature_missing_counts"][name] for row in rows) for name in proposals[dataset]["features"]}
    atomic_json(output / "dataset_config.json", {
        "schema_version": SCHEMA, "source_dataset": dataset,
        "folder_name_note": "Requested folder name says 2027; source is PhysioNet Challenge 2017" if dataset == "physionet2017" else None,
        "source_prepared_directory": str((ICENTIA if dataset == "icentia11k" else PHYSIO).relative_to(ROOT)),
        "feature_names": proposals[dataset]["features"], "label_to_id": LABELS[dataset],
        "sample_unit": "beat" if dataset == "icentia11k" else "record", "signal_dtype": "float32", "feature_dtype": "float32",
        "source_ecg_sampling_rate_hz": 200, "signal_vector_samples": 500 if dataset == "icentia11k" else 6000,
        "duration_seconds": 1 if dataset == "icentia11k" else 30,
        "representation_note": "500 interpolated representation points of a 1-s beat; feature grid reconstructed at 200 Hz" if dataset == "icentia11k" else "30-s vector with right zero-padding; features use valid_lengths only",
        "missing_features": "NaN with explicit feature_missing_mask; impute only using training data",
        "feature_standardization": "none; fit any imputer/scaler on train only",
        "selection": proposals[dataset], "source_preprocessing": source_config,
        "builder_sha256": sha256(__file__), "test_policy": "Fixed feature extraction only; no feature selection, training or evaluation on test",
        "npz_keys": ["X", "features", "feature_missing_mask", "feature_names", "y", "labels", "sample_ids", "valid_lengths", "dataset", "split", "client_id", "patient_id"],
    })
    report = {"status": "complete", "examples": expected, "shards": len(rows), "features_per_example": len(proposals[dataset]["features"]),
              "split_examples": expected_split_counts, "source_signals_and_labels_preserved": True,
              "every_shard_read_back_and_checked": True, "client_disjoint_splits_checked": True,
              "feature_missing_counts": missing, "test_metrics_computed": False}
    atomic_json(output / "validation_report.json", report)
    (output / "README.md").write_text(f"# {dataset}: sinais + features\n\nDataset materializado e autónomo: {expected:,} exemplos em {len(rows)} NPZ.\n\nCada linha de manifest.csv descreve um shard (ficheiro NPZ), não um exemplo. n_examples é o número de exemplos no shard. X[i], features[i], y[i], labels[i] e sample_ids[i] referem-se ao mesmo exemplo. feature_names define a ordem das colunas. Os splits e clientes originais são preservados.\n\nUsar featured_dataset.py na pasta superior para leitura por lotes. Features com NaN têm máscara explícita; ajustar imputação/normalização apenas em train. A seleção é experimental e os RR do PhysioNet continuam a exigir validação. Test recebeu apenas a extração fixa, sem treino nem métricas.\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dataset", choices=["all", "icentia11k", "physionet2017"], default="all")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers deve ser positivo")
    proposals = json.loads(PROPOSALS.read_text())
    datasets = list(OUTPUTS) if args.dataset == "all" else [args.dataset]
    all_jobs, configs = [], {}
    for dataset in datasets:
        jobs, configs[dataset] = make_jobs(dataset, proposals)
        OUTPUTS[dataset].mkdir(parents=True, exist_ok=True)
        atomic_json(OUTPUTS[dataset] / "validation_report.json", {"status": "building"})
        all_jobs.extend(jobs)
    completed = {dataset: [] for dataset in datasets}
    started = time.time()
    print(f"START jobs={len(all_jobs)} workers={args.workers}", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(build_shard, job): job for job in all_jobs}
        total = 0
        for index, future in enumerate(as_completed(futures), 1):
            result = future.result()
            completed[result["dataset"]].append(result)
            total += result["n_examples"]
            if index % 10 == 0 or index == len(all_jobs):
                print(f"PROGRESS {index}/{len(all_jobs)} examples={total} seconds={time.time()-started:.0f}", flush=True)
    for dataset in datasets:
        report = finalize(dataset, completed[dataset], proposals, configs[dataset])
        print("COMPLETE", dataset, json.dumps(report), flush=True)
    print(f"ALL_COMPLETE seconds={time.time()-started:.0f}", flush=True)


if __name__ == "__main__":
    main()
