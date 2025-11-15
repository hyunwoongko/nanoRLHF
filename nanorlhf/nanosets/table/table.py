from bisect import bisect_right
from typing import List, Optional, Dict, Any, Sequence, Iterable

from nanorlhf.nanosets.dtype.array import Array
from nanorlhf.nanosets.table.record_batch import RecordBatch
from nanorlhf.nanosets.table.schema import Schema
from nanorlhf.nanosets.utils import normalize_index


class Table:
    def __init__(self, batches: List[RecordBatch]):
        if not batches:
            raise ValueError("Table must have at least one RecordBatch")

        schema = batches[0].schema
        for b in batches:
            if b.schema != schema:
                raise ValueError("All RecordBatches must have the same schema")

        self.schema: Schema = schema
        self.batches: List[RecordBatch] = batches
        self.length: int = sum(b.length for b in batches)

    @classmethod
    def from_batches(cls, batches: List[RecordBatch]) -> "Table":
        return cls(batches)

    @classmethod
    def from_arrays(cls, schema: Schema, columns: List[Array]) -> "Table":
        batch = RecordBatch(schema, columns)
        return cls([batch])

    @classmethod
    def from_list(cls, rows: List[Optional[Dict[str, Any]]], *, strict_keys: bool = False) -> "Table":
        batch = RecordBatch.from_list(rows, strict_keys=strict_keys)
        return cls([batch])

    @classmethod
    def concat(cls, tables: List["Table"]) -> "Table":
        if not tables:
            raise ValueError("No tables to concatenate.")

        schema = tables[0].schema
        for t in tables:
            if t.schema != schema:
                raise ValueError("All tables must share the same schema to concatenate.")

        batches: List[RecordBatch] = []
        for t in tables:
            batches.extend(t.batches)

        return cls.from_batches(batches)

    def num_rows(self) -> int:
        return self.length

    def num_columns(self) -> int:
        return len(self.schema.fields)

    def column_names(self) -> List[str]:
        return self.schema.names()

    def iter_batches(self) -> Iterable[RecordBatch]:
        return iter(self.batches)

    def column(self, i_or_name) -> List[Array]:
        cols: List[Array] = []
        for b in self.batches:
            cols.append(b.column(i_or_name))
        return cols

    def slice(self, offset: int, length: int) -> "Table":
        if length < 0:
            raise ValueError("length must be non-negative")

        n = self.length
        if offset < 0:
            offset = n + offset
        if not (0 <= offset <= n):
            raise IndexError(f"offset {offset} out of range [0, {n}]")

        end = offset + length
        if end > n:
            raise IndexError(f"slice end {end} out of range [0, {n}]")

        if length == 0:
            first_batch = self.batches[0]
            empty_columns = [col.take([]) for col in first_batch.columns]
            empty_batch = RecordBatch(self.schema, empty_columns)
            return Table.from_batches([empty_batch])

        remaining = length
        batch_start_global = 0
        new_batches: List[RecordBatch] = []

        for b in self.batches:
            b_len = b.length
            batch_end_global = batch_start_global + b_len

            if batch_end_global <= offset:
                batch_start_global = batch_end_global
                continue

            local_start = max(0, offset - batch_start_global)
            local_available = b_len - local_start
            local_len = min(remaining, local_available)

            new_batches.append(b.slice(local_start, local_len))

            remaining -= local_len
            if remaining <= 0:
                break

            batch_start_global = batch_end_global

        return Table.from_batches(new_batches)

    def take(self, indices: Sequence[int]) -> "Table":
        if not indices:
            first_batch = self.batches[0]
            empty_columns = [col.take([]) for col in first_batch.columns]
            empty_batch = RecordBatch(self.schema, empty_columns)
            return Table.from_batches([empty_batch])

        n = self.length
        norm_indices = [normalize_index(idx, n) for idx in indices]

        batch_starts: List[int] = []
        current = 0
        for b in self.batches:
            batch_starts.append(current)
            current += b.length

        new_batches: List[RecordBatch] = []

        current_batch_idx: Optional[int] = None
        current_local_indices: List[int] = []
        prev_local: Optional[int] = None

        def flush():
            nonlocal current_batch_idx, current_local_indices, prev_local
            if current_batch_idx is None or not current_local_indices:
                return
            base_batch = self.batches[current_batch_idx]
            new_batches.append(base_batch.take(current_local_indices))
            current_batch_idx = None
            current_local_indices = []
            prev_local = None

        for gi in norm_indices:
            batch_idx = bisect_right(batch_starts, gi) - 1
            if batch_idx < 0 or batch_idx >= len(self.batches):
                raise IndexError(f"Global index {gi} not mapped to any batch.")
            local_idx = gi - batch_starts[batch_idx]

            if current_batch_idx is None:
                current_batch_idx = batch_idx
                current_local_indices = [local_idx]
                prev_local = local_idx
            else:
                if batch_idx == current_batch_idx and prev_local is not None and local_idx == prev_local + 1:
                    current_local_indices.append(local_idx)
                    prev_local = local_idx
                else:
                    flush()
                    current_batch_idx = batch_idx
                    current_local_indices = [local_idx]
                    prev_local = local_idx

        flush()

        if not new_batches:
            first_batch = self.batches[0]
            empty_columns = [col.take([]) for col in first_batch.columns]
            empty_batch = RecordBatch(self.schema, empty_columns)
            return Table.from_batches([empty_batch])

        return Table.from_batches(new_batches)

    def select(self, names: List[str]) -> "Table":
        new_batches = [b.select(names) for b in self.batches]
        return Table.from_batches(new_batches)

    def to_list(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []

        for b in self.batches:
            if b.num_columns() == 0:
                rows.extend({} for _ in range(b.length))
                continue

            cols = [c.to_list() for c in b.columns]
            for r in range(b.length):
                row: Dict[str, Any] = {}
                for f, col in zip(b.schema.fields, cols):
                    row[f.name] = col[r]
                rows.append(row)

        return rows
