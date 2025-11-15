import os
import random
from typing import List, Optional, Union, Callable, Dict, Any, Sequence

from nanorlhf.nanosets.io.ipc import read_table, write_table
from nanorlhf.nanosets.io.json_io import from_json, from_jsonl, to_json, to_jsonl
from nanorlhf.nanosets.table.record_batch import RecordBatch
from nanorlhf.nanosets.table.table import Table
from nanorlhf.nanosets.utils import DEFAULT_BATCH_SIZE


def _ext(path: str) -> str:
    base = os.path.basename(path)
    if "." not in base:
        return ""
    return base.rsplit(".", 1)[1].lower()


class Dataset:
    def __init__(self, table: Table):
        self._table = table

    @property
    def table(self) -> Table:
        return self._table

    def __len__(self) -> int:
        return self._table.length

    def __repr__(self) -> str:
        return f"Dataset(num_rows={len(self)}, schema={self._table.schema})"

    @classmethod
    def from_list(
        cls,
        rows: List[Optional[Dict[str, Any]]],
        *,
        strict_keys: bool = False,
        batch_size: Optional[int] = DEFAULT_BATCH_SIZE,
    ) -> "Dataset":
        table = Table.from_list(rows, strict_keys=strict_keys, batch_size=batch_size)
        return cls(table)

    def save_to_disk(self, path: str) -> None:
        with open(path, "wb") as fp:
            write_table(fp, self._table)

    def to_json(self, path: str, lines: bool = True) -> None:
        with open(path, "w", encoding="utf-8") as fp:
            if lines:
                to_jsonl(fp, self._table)
            else:
                to_json(fp, self._table)

    def to_dict(self) -> List[Optional[dict]]:
        return self._table.to_list()

    def select_columns(self, column_names: List[str]) -> "Dataset":
        return Dataset(self._table.select(column_names))

    def remove_columns(self, column_names: List[str]) -> "Dataset":
        all_names = self._table.column_names()
        drop_set = set(column_names)
        keep = [name for name in all_names if name not in drop_set]
        return Dataset(self._table.select(keep))

    def select(self, indices: Sequence[int]) -> "Dataset":
        return Dataset(self._table.take(indices))

    def shuffle(self, seed: Optional[int] = None) -> "Dataset":
        rng = random.Random(seed)
        idx = list(range(len(self)))
        rng.shuffle(idx)
        return self.select(idx)

    def map(
        self,
        function: Callable[
            [Union[Dict[str, Any], List[Dict[str, Any]]]],
            Union[Dict[str, Any], List[Dict[str, Any]]],
        ],
        batched: bool = False,
        batch_size: Optional[int] = DEFAULT_BATCH_SIZE,
    ) -> "Dataset":
        new_batches: List[RecordBatch] = []
        if not batched:
            for b in self._table.batches:
                rows = b.to_list()
                out_rows: List[Optional[Dict[str, Any]]] = []
                for r in rows:
                    out_rows.append(function(r))
                new_batches.append(RecordBatch.from_list(out_rows))
        else:
            actual_bs = batch_size if batch_size is not None and batch_size > 0 else None
            buffer: List[Dict[str, Any]] = []

            def flush(force: bool = False) -> None:
                nonlocal buffer
                if not buffer:
                    return
                if not force and actual_bs is not None and len(buffer) < actual_bs:
                    return
                mapped = function(buffer)
                if not isinstance(mapped, list):
                    raise TypeError("When batched=True, `function` must return a list of rows.")
                new_batches.append(RecordBatch.from_list(mapped))
                buffer = []

            for b in self._table.batches:
                rows = b.to_list()
                for r in rows:
                    buffer.append(r)
                    flush(False)
            flush(True)
        return Dataset(Table.from_batches(new_batches))

    def filter(
        self,
        predicate: Callable[[Dict[str, Any]], bool],
        batch_size: Optional[int] = DEFAULT_BATCH_SIZE,
    ) -> "Dataset":
        new_batches: List[RecordBatch] = []
        buffer: List[Optional[Dict[str, Any]]] = []
        bs = batch_size if batch_size is not None and batch_size > 0 else None

        for b in self._table.batches:
            rows = b.to_list()
            for r in rows:
                if r is None:
                    continue
                if predicate(r):
                    buffer.append(r)
                    if bs is not None and len(buffer) >= bs:
                        new_batches.append(RecordBatch.from_list(buffer))
                        buffer = []

        if buffer:
            new_batches.append(RecordBatch.from_list(buffer))

        if not new_batches:
            first_batch = self._table.batches[0]
            empty_cols = [col.take([]) for col in first_batch.columns]
            empty_batch = RecordBatch(self._table.schema, empty_cols)
            return Dataset(Table.from_batches([empty_batch]))

        return Dataset(Table.from_batches(new_batches))


def load_dataset(
    data_files: Union[str, List[str]],
    batch_size: Optional[int] = DEFAULT_BATCH_SIZE,
) -> Dataset:
    def _load_one(file: str) -> Table:
        e = _ext(file)
        if e == "json":
            return from_json(file, batch_size=batch_size)
        if e in ("jsonl", "ndjson"):
            return from_jsonl(file, batch_size=batch_size)
        if e == "nano":
            return read_table(file)
        raise ValueError(f"Unsupported extension for {file!r}. Expected .json, .jsonl/.ndjson, or .nano")

    def _load_many(files: Union[str, List[str]]) -> Dataset:
        flist = [files] if isinstance(files, str) else list(files)
        tables = [_load_one(f) for f in flist]
        table = Table.concat(tables) if len(tables) > 1 else tables[0]
        return Dataset(table)

    if isinstance(data_files, (str, list)):
        return _load_many(data_files)
    raise TypeError("data_files must be str or list[str].")


load_from_disk = load_dataset
