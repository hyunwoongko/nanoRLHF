from typing import List, Union, Sequence, Optional, Dict, Any

from nanorlhf.nanosets.dtype.array import Array
from nanorlhf.nanosets.dtype.struct_array import StructArray
from nanorlhf.nanosets.table.schema import Schema
from nanorlhf.nanosets.table.field import Field


class RecordBatch:
    """
    A RecordBatch is a physical unit of a columnar table: a fixed set of columns that share a Schema
    and have the same number of rows.

    Discussion:
        Q. What does a RecordBatch represent in this system?
            It represents a chunk of a table where all columns are aligned by row index.
            Instead of storing rows as Python dicts, it stores column Arrays plus a Schema that describes them.

        Q. Why require all columns to have the same length?
            Row alignment is the invariant that makes column-wise operations (take/slice/select) safe:
            the same row index refers to the corresponding element across every column.

        Q. How should I think about "row operations" on a RecordBatch?
            Operations like slice/take feel row-oriented, but they are implemented by applying the same
            index selection to every column Array, producing a new RecordBatch that reuses buffers when possible.
    """

    def __init__(self, schema: Schema, columns: List[Array]):
        """
        Create a RecordBatch from a schema and a list of column Arrays.

        Args:
            schema (Schema): Column definitions (names/types/nullability) in order.
            columns (List[Array]): Column arrays in the same order as schema.fields.

        Discussion:
            Q. What invariants are enforced at construction time?
                - The number of columns must match the number of schema fields.
                - All columns must have exactly the same length.

            Q. Where is the row count stored?
                The row count is stored in self.length and is derived from the columns’ lengths.
        """
        if len(schema.fields) != len(columns):
            raise ValueError(f"Number of columns ({len(columns)}) must match schema fields ({len(schema.fields)})")

        lengths = {len(c) for c in columns}
        if len(lengths) > 1:
            raise ValueError("All columns in a RecordBatch must have the same length.")

        self.schema: Schema = schema
        self.columns: List[Array] = columns
        self.length: int = next(iter(lengths)) if lengths else 0

    def num_rows(self) -> int:
        """
        Return the number of rows in this batch.

        Returns:
            int: The batch row count.

        Discussion:
            Q. What does num_rows mean for a RecordBatch?
                It is the shared length of all columns (i.e., the number of row positions available).
        """
        return self.length

    def num_columns(self) -> int:
        """
        Return the number of columns in this batch.

        Returns:
            int: The number of column Arrays.

        Discussion:
            Q. Does this always equal the number of schema fields?
                Yes by construction: __init__ enforces len(columns) == len(schema.fields).
        """
        return len(self.columns)

    def column(self, i_or_name: Union[int, str]) -> Array:
        """
        Fetch a column by integer index or by field name.

        Args:
            i_or_name (Union[int, str]): Column index (int) or column name (str).

        Returns:
            Array: The requested column Array.

        Examples:
            >>> col0 = batch.column(0)
            >>> name_col = batch.column("name")

        Discussion:
            Q. How does name-based lookup work?
                It uses schema.index(name) to convert a field name into a column position.

            Q. Why return an Array instead of converting to Python objects?
                RecordBatch is a columnar container. The Array preserves the buffer-based representation
                and enables further zero-copy-ish operations at the column level.
        """
        if isinstance(i_or_name, int):
            return self.columns[i_or_name]
        if isinstance(i_or_name, str):
            idx = self.schema.index(i_or_name)
            return self.columns[idx]
        raise TypeError("Argument must be an integer index or a string column name.")

    def slice(self, offset: int, length: int) -> "RecordBatch":
        """
        Return a new RecordBatch containing a contiguous range of rows.

        Args:
            offset (int): Start row index (supports negative indexing).
            length (int): Number of rows to include (must be non-negative).

        Returns:
            RecordBatch: A new batch whose columns are the corresponding row slices.

        Examples:
            >>> head = batch.slice(0, 10)
            >>> tail3 = batch.slice(-3, 3)

        Discussion:
            Q. What is the core idea of slice in a columnar batch?
                It computes a row index range [offset, offset+length) and applies the same selection
                to every column via col.take(row_range).

            Q. Why is slice implemented using take(range(...)) on each column?
                It keeps one consistent selection primitive (take) for both contiguous and non-contiguous
                selections. Column Arrays can then decide how to implement take efficiently (e.g., buffer slicing).

            Q. What happens when length == 0?
                It returns an empty batch with the same schema by taking an empty selection on every column.
        """
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
        """
        Return a new RecordBatch containing rows selected by arbitrary indices.

        Args:
            indices (Sequence[int]): Row indices to take (the interpretation of negative indices is
                delegated to each column's take implementation).

        Returns:
            RecordBatch: A new batch whose columns contain the selected rows.

        Examples:
            >>> picked = batch.take([2, 0, 2])

        Discussion:
            Q. How is take implemented at the batch level?
                It applies the same indices to every column: new_cols = [col.take(indices) for col in self.columns].

            Q. Why is this the natural way to implement "row selection" in a columnar system?
                Because "a row" is not stored as one object. It is the aligned combination of per-column elements
                at the same logical row index, so selection must be applied consistently across columns.

            Q. Does RecordBatch.take attempt to optimize for contiguity?
                No. It delegates that to the column Arrays. Some Arrays may produce buffer slices for contiguous
                patterns and indices-based views for arbitrary patterns.
        """
        new_cols = [col.take(indices) for col in self.columns]
        return RecordBatch(self.schema, new_cols)

    def select(self, names: List[str]) -> "RecordBatch":
        """
        Return a new RecordBatch containing only a subset of columns (in the given order).

        Args:
            names (List[str]): Field names to keep.

        Returns:
            RecordBatch: A new batch with a narrowed schema and the corresponding columns.

        Examples:
            >>> small = batch.select(["id", "name"])

        Discussion:
            Q. What is the key correctness rule when selecting columns?
                You must filter both the schema and the columns using the same field indices,
                otherwise names/types and column data would no longer match.

            Q. How does it map names to columns?
                It uses schema.index(name) for each requested name, then slices schema.fields and columns
                by those indices to construct a new Schema and a new RecordBatch.
        """
        field_indices = [self.schema.index(name) for name in names]
        new_fields = tuple(self.schema.fields[i] for i in field_indices)
        new_schema = Schema(new_fields)
        new_columns = [self.columns[i] for i in field_indices]
        return RecordBatch(new_schema, new_columns)

    def to_list(self) -> List[Dict[str, Any]]:
        """
        Convert this batch into a row-oriented representation (list of dicts).

        Returns:
            List[Dict[str, Any]]: Rows as Python dicts mapping field name -> Python value.

        Examples:
            >>> rows = batch.to_list()
            [{'id': 1, 'name': 'a'}, {'id': 2, 'name': 'b'}, ...]

        Discussion:
            Q. Why does to_list exist in a columnar system?
                It is mainly for debugging, testing, and simple JSON-style interop.
                It materializes Python objects and therefore is not the preferred fast path.

            Q. How is it implemented efficiently at a high level?
                It first converts each column to a Python list once (per_column_lists),
                then assembles row dicts by indexing into those per-column lists.

            Q. What happens when there are zero columns?
                It returns a list of empty dicts with length equal to the number of rows,
                preserving the idea that the batch still has rows even if it has no fields.
        """
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
    def from_list(cls, rows: List[Optional[Dict[str, Any]]], strict_keys: bool = False) -> "RecordBatch":
        """
        Construct a RecordBatch from a row-oriented representation (list of dicts / None).

        Args:
            rows (List[Optional[Dict[str, Any]]]): Each element is a row dict or None (representing a null row),
                following StructArray.from_list semantics.
            strict_keys (bool): Passed through to StructArray.from_list.

        Returns:
            RecordBatch: A batch where columns are reconstructed as Arrays with an inferred schema.

        Examples:
            >>> batch = RecordBatch.from_list(
            ...     [{"id": 1, "name": "a"}, {"id": 2, "name": None}, None],
            ...     strict_keys=False,
            ... )

        Discussion:
            Q. What is the core strategy for turning rows into columns?
                It delegates to StructArray.from_list(rows), which builds a struct array whose children Arrays
                correspond to each field/column.

            Q. How is the Schema determined?
                It iterates over (field_name, child_array) pairs and constructs Field objects using:
                - dtype from child.dtype
                - nullable inferred from whether child.validity is present

            Q. Why infer nullable from child.validity is not None?
                In this representation, a validity bitmap is only needed when there exists at least one null.
                If validity is None, all entries are valid and the column can be treated as non-nullable.
        """
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
