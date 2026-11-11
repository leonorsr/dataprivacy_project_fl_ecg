"""Prepare LDP experiments from the preserved final-candidate NPZ files."""
import ast
import inspect
import json
from pathlib import Path

import numpy as np

from support.federated_global import shared
from support.federated_global.shared import ROOT, DOMAINS, dump, old_soft


FEATURE_NAMES = {
    "icentia11k": [
        "mean", "std", "peak_to_peak", "rms", "iqr",
        "skewness", "kurtosis", "mean_abs_difference",
        "max_abs_slope", "total_variation_per_second",
        "spectral_entropy", "dominant_frequency_hz",
        "power_fraction_0_5_5", "power_fraction_5_15",
        "power_fraction_15_40", "source_peak_to_peak",
        "source_max_abs",
    ],
    "physionet2017": [
        "mean", "std", "peak_to_peak", "rms", "iqr",
        "skewness", "kurtosis", "mean_abs_difference",
        "max_abs_slope", "total_variation_per_second",
        "spectral_entropy", "dominant_frequency_hz",
        "power_fraction_0_5_5", "power_fraction_5_15",
        "power_fraction_15_40", "rr_median_ms", "rr_iqr_ms",
        "rr_rmssd_ms", "rr_interval_count",
    ],
}


def public_anchor_generator():
    tree = ast.parse(inspect.getsource(old_soft.synthetic_queries))
    found = 0

    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Return)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "dict"
        ):
            node.value.keywords.append(
                ast.keyword(
                    arg="raw_features",
                    value=ast.Name(id="raw", ctx=ast.Load()),
                )
            )
            found += 1

    assert found == 1, "Não foi possível adaptar o gerador de âncoras."
    ast.fix_missing_locations(tree)

    namespace = dict(old_soft.__dict__)
    namespace["SEED"] = 2026
    exec(compile(tree, "<public synthetic features>", "exec"), namespace)
    return namespace["synthetic_queries"]


def prepare(base, calibration_anchors=128):
    base = Path(base)
    source = (
        ROOT
        / "SoftLabels tests/final candidates/artifacts/data/data"
    )
    metadata = (
        ROOT
        / "SoftLabels tests/final candidates/artifacts/data"
    )
    registry_path = metadata / "sample_registry.json"

    assert source.is_dir(), f"Diretório de NPZ não encontrado: {source}"
    assert registry_path.is_file(), "sample_registry.json não encontrado."

    registry = json.loads(registry_path.read_text())
    generator = public_anchor_generator()
    domains = {}
    public_metadata = {}
    counts = []

    for ds in DOMAINS:
        names = FEATURE_NAMES[ds]
        files = sorted((source / ds).glob("*.npz"))
        assert files, f"Nenhum NPZ encontrado para {ds}."

        clients = {}
        signal_length = None

        for path in files:
            cid = path.stem
            uid = f"{ds}:{cid}"
            assert uid in registry, f"Cliente ausente do registry: {uid}"

            with np.load(path, allow_pickle=False) as z:
                c = {key: z[key].copy() for key in z.files}

            required = {
                "X", "features", "y", "sample_ids", "valid_lengths",
                "record_group", "fit_indices", "selection_indices",
                "holdout_indices", "dev_indices",
            }
            assert required.issubset(c), f"Chaves ausentes em {path}"
            assert c["features"].shape[1] == len(names)
            assert c["X"].ndim == 2
            assert c["X"].shape[0] == len(c["y"]) == len(c["sample_ids"])
            assert c["features"].shape[0] == len(c["y"])
            assert np.isfinite(c["X"]).all()
            assert np.isfinite(c["features"]).all()

            if signal_length is None:
                signal_length = c["X"].shape[1]
            assert c["X"].shape[1] == signal_length

            original_y = c["y"].copy()
            assert not np.any(original_y == 3), (
                f"Classe 3 inesperada em {uid}; não é seguro binarizar."
            )
            c["original_y"] = original_y
            c["y"] = np.isin(original_y, [1, 2]).astype(np.int64)

            info = registry[uid]
            split = info["split"]
            sample_ids = c["sample_ids"].astype(str)

            if split == "train":
                fit = c["fit_indices"].astype(np.int64)
                selection = c["selection_indices"].astype(np.int64)
                hold = c["holdout_indices"].astype(np.int64)

                assert len(fit) and len(selection) and len(hold)
                assert len(c["dev_indices"]) == 0

                final_fit = np.concatenate([fit, selection])
                assert not (
                    set(sample_ids[final_fit])
                    & set(sample_ids[hold])
                )

                assert set(sample_ids[fit]) == set(info["fit"])
                assert set(sample_ids[selection]) == set(info["selection"])
                assert set(sample_ids[hold]) == set(info["holdout"])

            elif split == "dev":
                dev = c["dev_indices"].astype(np.int64)
                assert len(dev) == len(c["y"])
                assert not len(c["fit_indices"])
                assert not len(c["selection_indices"])
                assert set(sample_ids[dev]) == set(info["holdout"])

                fit = np.array([], dtype=np.int64)
                selection = np.array([], dtype=np.int64)
                hold = dev
                final_fit = np.array([], dtype=np.int64)
                c["holdout_indices"] = dev.copy()
            else:
                raise AssertionError(f"Split desconhecido em {uid}: {split}")

            c.update(
                dataset=ds,
                uid=uid,
                client_id=cid,
                split=split,
                K=2,
                fit_indices=final_fit,
                final_fit_indices=final_fit,
                selection_indices=selection,
                hold_indices=hold,
            )

            clients[cid] = c

            counts.append(dict(
                dataset=ds,
                client=uid,
                split=split,
                fit=len(fit),
                selection=len(selection),
                holdout=len(hold) if split == "train" else 0,
                dev=len(hold) if split == "dev" else 0,
            ))

        train_count = sum(c["split"] == "train" for c in clients.values())
        dev_count = sum(c["split"] == "dev" for c in clients.values())
        assert (train_count, dev_count) == (
            (16, 2) if ds == "icentia11k" else (6, 1)
        )

        config = dict(
            source_dataset=ds,
            signal_vector_samples=signal_length,
            feature_names=names,
        )

        projection = np.random.default_rng(2026).normal(
            0,
            1 / np.sqrt(32),
            (signal_length, 32),
        ).astype(np.float32)

        calibration = generator(
            config,
            dict(
                mean=np.zeros(len(names)),
                scale=np.ones(len(names)),
                projection=projection,
            ),
            n=calibration_anchors,
        )

        raw = calibration["raw_features"]
        mean = raw.mean(axis=0)
        scale = np.maximum(raw.std(axis=0), 1e-6)

        for c in clients.values():
            c["F"] = np.clip(
                (c["features"] - mean) / scale, -8, 8
            ).astype(np.float32)

            projected = np.einsum(
                "ij,jk->ik",
                c["X"].astype(np.float64),
                projection.astype(np.float64),
                optimize=False,
            )
            c["Z"] = np.column_stack([projected, c["F"]]).astype(
                np.float32
            )

            assert np.isfinite(c["F"]).all()
            assert np.isfinite(c["Z"]).all()

        anchor = generator(
            config,
            dict(mean=mean, scale=scale, projection=projection),
            n=32,
        )
        anchor.pop("raw_features")

        blocks = []
        positions = {}
        offset = 0

        for c in clients.values():
            ix = c["hold_indices"]
            positions[c["uid"]] = np.arange(offset, offset + len(ix))
            offset += len(ix)
            blocks.append((c, ix))

        query = {
            key: np.concatenate([c[key][ix] for c, ix in blocks])
            for key in [
                "X", "F", "Z", "valid_lengths", "sample_ids", "y"
            ]
        }
        query["positions"] = positions

        domains[ds] = dict(
            dataset=ds,
            clients=clients,
            config=config,
            projection=projection,
            query=query,
            anchor=anchor,
            task="binary",
            K=2,
        )

        fit_ids = {
            value
            for c in clients.values()
            for value in c["sample_ids"][c["fit_indices"]]
        }
        hold_ids = {
            value
            for c in clients.values()
            for value in c["sample_ids"][c["hold_indices"]]
        }
        anchor_ids = set(anchor["sample_ids"])

        assert not fit_ids & hold_ids
        assert not fit_ids & anchor_ids

        public_metadata[ds] = dict(
            feature_names=names,
            mean=mean.tolist(),
            scale=scale.tolist(),
            calibration=(
                f"{calibration_anchors} exemplos sintéticos públicos; "
                "sem estatísticas privadas dos clientes"
            ),
            source="geração sintética fixa com seed pública 2026",
            anchor_ids=anchor["sample_ids"].tolist(),
        )

    assert sum(c["split"] == "train" for d in domains.values()
               for c in d["clients"].values()) == 22
    assert sum(c["split"] == "dev" for d in domains.values()
               for c in d["clients"].values()) == 3

    dump(base / "public_preprocessing.json", public_metadata)
    dump(base / "sample_counts.json", counts)
    dump(
        base / "research_private" / "sample_registry.json",
        shared.sample_registry(domains),
    )

    print("Preparação LDP concluída.")
    print("Clientes: 22 treino + 3 dev.")
    print("Âncoras sintéticas: 32 por domínio.")
    print("Features: Icentia 17; PhysioNet 19.")

    return domains
