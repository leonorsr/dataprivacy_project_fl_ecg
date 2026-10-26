"""Leitura por lotes dos datasets com ECG + features, sem ajustar transformações.

Exemplo:
    ds = FeaturedDataset('.../icentia11k_prepared_featured', split='train', client_id='client_00')
    for batch in ds.iter_batches(batch_size=128):
        signals, features, labels = batch['X'], batch['features'], batch['y']
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd


class FeaturedDataset:
    def __init__(self, root, split=None, client_id=None):
        self.root = Path(root).resolve()
        self.config = json.loads((self.root / "dataset_config.json").read_text())
        report = json.loads((self.root / "validation_report.json").read_text())
        if report.get("status") != "complete":
            raise ValueError("A materialização do dataset ainda não está concluída.")
        if self.config["schema_version"] != 1:
            raise ValueError("Versão do formato não suportada.")
        if split not in (None, "train", "dev", "test"):
            raise ValueError("split deve ser train, dev, test ou None.")
        manifest = pd.read_csv(self.root / "manifest.csv", keep_default_na=False)
        if split is not None:
            manifest = manifest[manifest["split"] == split]
        if client_id is not None:
            manifest = manifest[manifest["client_id"] == client_id]
        self.manifest = manifest.reset_index(drop=True)
        if not len(manifest):
            raise ValueError("Não existem exemplos para o split/cliente pedido.")
        self.feature_names = tuple(self.config["feature_names"])
        self.label_to_id = dict(self.config["label_to_id"])
        self._ends = np.cumsum(self.manifest["n_examples"].to_numpy(dtype=np.int64))

    def __len__(self):
        return int(self._ends[-1])

    def _read_shard(self, index):
        row = self.manifest.iloc[int(index)]
        with np.load(self.root / row["shard_path"], allow_pickle=False) as data:
            if tuple(data["feature_names"].tolist()) != self.feature_names:
                raise ValueError("Ordem de features incompatível com a configuração.")
            arrays = {key: data[key] for key in ("X", "features", "feature_missing_mask", "y", "labels", "sample_ids", "valid_lengths")}
            meta = {key: str(data[key]) for key in ("dataset", "split", "client_id", "patient_id")}
        n = int(row["n_examples"])
        if any(len(value) != n for value in arrays.values()):
            raise ValueError("Número de exemplos incompatível com o manifesto.")
        if arrays["features"].shape != (n, len(self.feature_names)):
            raise ValueError("Forma da matriz de features incompatível.")
        if arrays["feature_missing_mask"].shape != arrays["features"].shape:
            raise ValueError("Máscara de features incompatível.")
        for key in ("split", "client_id", "patient_id"):
            if meta[key] != str(row[key]):
                raise ValueError(f"Metadados incompatíveis: {key}")
        return arrays, meta

    def get_sample(self, index):
        """Lê um exemplo; apenas o bloco correspondente é descomprimido."""
        if not isinstance(index, (int, np.integer)):
            raise TypeError("O índice deve ser inteiro.")
        index = int(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        shard = int(np.searchsorted(self._ends, index, side="right"))
        local = index - (int(self._ends[shard-1]) if shard else 0)
        arrays, meta = self._read_shard(shard)
        return {**{key: value[local] for key, value in arrays.items()}, **meta, "feature_names": self.feature_names}

    def iter_batches(self, batch_size=256, shuffle=False, seed=2024):
        """Itera por blocos: shuffle baralha a ordem dos blocos e as linhas de cada bloco.

        Um lote não mistura clientes. O último lote de cada bloco pode ser menor.
        Não são ajustados imputadores/scalers; features em falta mantêm-se NaN.
        """
        if not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
            raise ValueError("batch_size deve ser um inteiro positivo.")
        rng = np.random.default_rng(seed)
        shards = np.arange(len(self.manifest))
        if shuffle:
            rng.shuffle(shards)
        for shard in shards:
            arrays, meta = self._read_shard(shard)
            indices = np.arange(len(arrays["X"]))
            if shuffle:
                rng.shuffle(indices)
            for start in range(0, len(indices), batch_size):
                selection = indices[start:start+batch_size]
                yield {**{key: value[selection] for key, value in arrays.items()}, **meta, "feature_names": self.feature_names}
