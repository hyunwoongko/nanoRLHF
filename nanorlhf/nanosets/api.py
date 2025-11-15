import os
import random
from typing import List, Optional, Union, Callable, Dict, Any, Sequence

from nanorlhf.nanosets.io.ipc import read_table, write_table
from nanorlhf.nanosets.io.json_io import from_json, from_jsonl, to_json, to_jsonl
from nanorlhf.nanosets.table.record_batch import RecordBatch
from nanorlhf.nanosets.table.table import Table
from nanorlhf.nanosets.utils import ext


class Dataset:
    def __init__(self, table: Table):
        self.table = table

    def __len__(self) -> int:
        return self.table.length

    def __repr__(self):
        return f"Dataset(num_rows={len(self)}, schema={self.table.schema})"

    def save_to_disk(self, path: str):
        with open(path, "wb") as fp:
            write_table(fp, self.table)

    def to_json(self, path: str, lines: bool = True):
        with open(path, "w", encoding="utf-8") as fp:
            if lines:
                to_jsonl(fp, self.table)
            else:
                to_json(fp, self.table)

    def to_dict(self) -> List[Optional[dict]]:
        return self.table.to_list()

    def select_columns(self, column_names: List[str]) -> "Dataset":
        return Dataset(self.table.select(column_names))

    def remove_columns(self, column_names: List[str]) -> "Dataset":
        all_names = self.table.column_names()
        drop_set = set(column_names)
        keep = [name for name in all_names if name not in drop_set]
        return Dataset(self.table.select(keep))

    def select(self, indices: Sequence[int]) -> "Dataset":
        return Dataset(self.table.take(indices))

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
        batch_size: int = 1000,
    ) -> "Dataset":
        new_batches: List[RecordBatch] = []
        if not batched:
            for b in self.table.batches:
                rows = b.to_list()
                out_rows: List[Optional[Dict[str, Any]]] = []
                for r in rows:
                    out_rows.append(function(r))
                new_batches.append(RecordBatch.from_list(out_rows))
        else:
            buffer: List[Dict[str, Any]] = []

            def flush_chunk(chunk: List[Dict[str, Any]]):
                mapped = function(chunk)
                if not isinstance(mapped, list):
                    raise TypeError("When batched=True, `function` must return a list of rows.")
                new_batches.append(RecordBatch.from_list(mapped))

            for b in self.table.batches:
                rows = b.to_list()
                for r in rows:
                    buffer.append(r)
                    if len(buffer) >= batch_size:
                        flush_chunk(buffer)
                        buffer = []

            if buffer:
                flush_chunk(buffer)

        return Dataset(Table.from_batches(new_batches))

    def filter(self, predicate: Callable[[Dict[str, Any]], bool]) -> "Dataset":
        new_batches: List[RecordBatch] = []
        for b in self.table.batches:
            rows = b.to_list()
            kept: List[Optional[Dict[str, Any]]] = []
            for r in rows:
                if r is None:
                    continue
                if predicate(r):
                    kept.append(r)
            if kept:
                new_batches.append(RecordBatch.from_list(kept))

        if not new_batches:
            first_batch = self.table.batches[0]
            empty_cols = [col.take([]) for col in first_batch.columns]
            empty_batch = RecordBatch(self.table.schema, empty_cols)
            return Dataset(Table.from_batches([empty_batch]))

        return Dataset(Table.from_batches(new_batches))


def load_dataset(data_files: Union[str, List[str]]) -> Dataset:
    def _load_one(file: str) -> Table:
        e = ext(file)
        if e == "json":
            return from_json(file)
        if e in ("jsonl", "ndjson"):
            return from_jsonl(file)
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
