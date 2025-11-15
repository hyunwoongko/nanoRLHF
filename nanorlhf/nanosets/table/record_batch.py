from typing import List, Union, Sequence, Optional, Dict, Any

from nanorlhf.nanosets.dtype.array import Array
from nanorlhf.nanosets.dtype.struct_array import StructArray
from nanorlhf.nanosets.table.schema import Schema
from nanorlhf.nanosets.table.field import Field


class RecordBatch:
    def __init__(self, schema: Schema, columns: List[Array]):
        if len(schema.fields) != len(columns):
            raise ValueError(
                f"Number of columns ({len(columns)}) must match schema fields ({len(schema.fields)})"
            )

        lengths = {len(c) for c in columns}
        if len(lengths) > 1:
            raise ValueError("All columns in a RecordBatch must have the same length.")

        self.schema: Schema = schema
        self.columns: List[Array] = columns
        self.length: int = next(iter(lengths)) if lengths else 0

    def num_rows(self) -> int:
        return self.length

    def num_columns(self) -> int:
        return len(self.columns)

    def column(self, i_or_name: Union[int, str]) -> Array:
        if isinstance(i_or_name, int):
            return self.columns[i_or_name]
        if isinstance(i_or_name, str):
            idx = self.schema.index(i_or_name)
            return self.columns[idx]
        raise TypeError("Argument must be an integer index or a string column name.")

    def slice(self, offset: int, length: int) -> "RecordBatch":
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
            new_cols = [col.take([]) for col in self.columns]
            return RecordBatch(self.schema, new_cols)

        row_range = range(offset, end)
        new_cols = [col.take(row_range) for col in self.columns]
        return RecordBatch(self.schema, new_cols)

    def take(self, indices: Sequence[int]) -> "RecordBatch":
        new_cols = [col.take(indices) for col in self.columns]
        return RecordBatch(self.schema, new_cols)

    def select(self, names: List[str]) -> "RecordBatch":
        field_indices = [self.schema.index(name) for name in names]
        new_fields = tuple(self.schema.fields[i] for i in field_indices)
        new_schema = Schema(new_fields)
        new_columns = [self.columns[i] for i in field_indices]
        return RecordBatch(new_schema, new_columns)

    def to_list(self) -> List[Dict[str, Any]]:
        if self.num_columns() == 0:
            return [{} for _ in range(self.length)]

        rows: List[Dict[str, Any]] = []
        per_column_lists = [col.to_list() for col in self.columns]

        for row_index in range(self.length):
            row: Dict[str, Any] = {}
            for field, column_values in zip(self.schema.fields, per_column_lists):
                row[field.name] = column_values[row_index]
            rows.append(row)

        return rows

    @classmethod
    def from_list(
        cls,
        rows: List[Optional[Dict[str, Any]]],
        *,
        strict_keys: bool = False,
    ) -> "RecordBatch":
        struct = StructArray.from_list(rows, strict_keys=strict_keys)
        field_names = struct.field_names

        fields = tuple(
            Field(
                name=name,
                dtype=child.dtype,
                nullable=(child.validity is not None),
            )
            for name, child in zip(field_names, struct.children)
        )
        schema = Schema(fields)
        return cls(schema, struct.children)
