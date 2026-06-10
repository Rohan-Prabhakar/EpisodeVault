from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

_SCHEMA = pa.schema([
    pa.field("model_version", pa.string()),
    pa.field("dataset_version", pa.string()),
    pa.field("framework", pa.string()),
    pa.field("logged_at", pa.float64()),
    pa.field("extras", pa.string()),
])


class LineageStore:
    def __init__(self, store_path: str | Path) -> None:
        self._root = Path(store_path)
        self._root.mkdir(parents=True, exist_ok=True)
        self._lineage_path = self._root / "lineage.parquet"

    def log_training_run(
        self,
        model_version: str,
        dataset_version: str,
        framework: str = "lerobot",
        **extras: Any,
    ) -> None:
        row = {
            "model_version": model_version,
            "dataset_version": dataset_version,
            "framework": framework,
            "logged_at": time.time(),
            "extras": json.dumps(extras),
        }
        table = pa.Table.from_pylist([row], schema=_SCHEMA)

        if self._lineage_path.exists():
            existing = pq.read_table(self._lineage_path)
            combined = pa.concat_tables([existing, table])
        else:
            combined = table

        tmp = self._lineage_path.with_suffix(".tmp")
        pq.write_table(combined, tmp, compression="snappy")
        os.replace(tmp, self._lineage_path)

    def dataset_version_for_model(self, model_version: str) -> str | None:
        if not self._lineage_path.exists():
            return None
        table = pq.read_table(
            self._lineage_path,
            filters=[("model_version", "=", model_version)],
        )
        if table.num_rows == 0:
            return None
        df = table.to_pandas().sort_values("logged_at", ascending=False)
        return str(df.iloc[0]["dataset_version"])

    def list_runs(self) -> list[dict[str, Any]]:
        if not self._lineage_path.exists():
            return []
        df = pq.read_table(self._lineage_path).to_pandas()
        df = df.sort_values("logged_at", ascending=False)
        records = df.to_dict(orient="records")
        for r in records:
            r["extras"] = json.loads(r["extras"]) if r["extras"] else {}
        return records
