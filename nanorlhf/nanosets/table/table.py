from bisect import bisect_right
from typing import List, Optional, Dict, Any, Sequence, Iterable

from nanorlhf.nanosets.dtype.array import Array
from nanorlhf.nanosets.table.record_batch import RecordBatch
from nanorlhf.nanosets.table.schema import Schema
from nanorlhf.nanosets.utils import normalize_index, DEFAULT_BATCH_SIZE


class Table:
    """
    A logical table composed of one or more RecordBatches that share the same Schema.

    Discussion:
        Q. What is the difference between Table and RecordBatch?
            RecordBatch is the physical unit: one chunk of aligned columns with a fixed row count.
            Table is the logical unit: it concatenates multiple RecordBatches and exposes a single global
            row index space (0..len(table)-1) across all batches.

        Q. Why keep multiple RecordBatches instead of merging everything into one big batch?
            Keeping batches makes it easier to:
            - build and process large datasets incrementally,
            - concatenate tables cheaply (by concatenating batch lists),
            - perform slice/take without re-materializing the entire table.

        Q. What is the key invariant of a Table?
            All batches must have exactly the same Schema (same fields in the same order).
    """

    def __init__(self, batches: List[RecordBatch]):
        """
        Create a Table from a list of RecordBatches.

        Args:
            batches (List[RecordBatch]): RecordBatches to concatenate logically.

        Discussion:
            Q. What does the constructor validate?
                It validates:
                - there is at least one batch,
                - all batches have the same schema as the first batch.

            Q. How is the table length determined?
                The total length is the sum of per-batch lengths:
                self.length = sum(b.length for b in batches)
        """
        if not batches:
            raise ValueError("Table must have at least one RecordBatch")
        schema = batches[0].schema
        for batch in batches:
            if batch.schema != schema:
                raise ValueError("All RecordBatches must have the same schema")
        self.schema: Schema = schema
        self.batches: List[RecordBatch] = batches
        self.length: int = sum(b.length for b in batches)

    def __getitem__(self, item):
        """
        Index or slice the Table using Python indexing semantics.

        Returns:
            - If item is int: returns a single row as a dict (materialized via to_list()).
            - If item is slice: returns a new Table (implemented via take on expanded indices).

        Examples:
            >>> row = table[0]
            >>> last = table[-1]
            >>> sub = table[10:20]
            >>> every_other = table[0:10:2]

        Discussion:
            Q. Why does integer indexing return a row dict instead of an Array or RecordBatch?
                This method is a convenience API for interactive use. It materializes a single row by
                delegating to slice(...).to_list()[0]. It is not intended as the fastest path.

            Q. How is slicing implemented?
                It expands the slice into an explicit list of indices and calls self.take(indices).
                This keeps selection logic centralized in take, at the cost of allocating an index list.

            Q. Are negative indices supported?
                Yes for integer indexing: negative indices are normalized into [0, n).
        """
        if isinstance(item, int):
            n = self.length
            if item < 0:
                item += n
            if not (0 <= item < n):
                raise IndexError(f"Index {item} out of range [0, {n})")
            return self.slice(item, 1).to_list()[0]
        elif isinstance(item, slice):
            indices = list(range(*item.indices(len(self))))
            return self.take(indices)
        else:
            raise TypeError("Invalid argument type.")

    def __len__(self):
        """
        Return the number of rows in the table.

        Returns:
            int: Total row count across all batches.

        Discussion:
            Q. What does len(table) measure?
                It measures the logical row count of the entire Table, not the number of batches.
        """
        return self.length

    @classmethod
    def from_batches(cls, batches: List[RecordBatch]) -> "Table":
        """
        Construct a Table from RecordBatches.

        Args:
            batches (List[RecordBatch]): RecordBatches to use.

        Returns:
            Table: A new Table over the given batches.

        Examples:
            >>> table = Table.from_batches([batch1, batch2])

        Discussion:
            Q. Why have from_batches if __init__ already accepts batches?
                It provides a more descriptive and discoverable constructor in user code, and matches other
                constructors like from_arrays / from_list.
        """
        return cls(batches)

    @classmethod
    def from_arrays(cls, schema: Schema, columns: List[Array]) -> "Table":
        """
        Construct a single-batch Table from a schema and column arrays.

        Args:
            schema (Schema): Schema describing the columns.
            columns (List[Array]): Column arrays aligned by row index.

        Returns:
            Table: A Table containing exactly one RecordBatch.

        Examples:
            >>> table = Table.from_arrays(schema, [col1, col2])

        Discussion:
            Q. What is the relationship between from_arrays and RecordBatch?
                It builds a RecordBatch(schema, columns) and wraps it into a Table with one batch.
        """
        batch = RecordBatch(schema, columns)
        return cls([batch])

    @classmethod
    def from_list(
        cls,
        rows: List[Optional[Dict[str, Any]]],
        strict_keys: bool = False,
        batch_size: Optional[int] = DEFAULT_BATCH_SIZE,
    ) -> "Table":
        """
        Construct a Table from row-oriented data (list of dicts / None), optionally chunked into batches.

        Args:
            rows (List[Optional[Dict[str, Any]]]): Row dicts (or None) to convert into a columnar table.
            strict_keys (bool): Passed through to RecordBatch.from_list / StructArray.from_list.
            batch_size (Optional[int]): If positive, splits rows into batches of at most batch_size.
                If None or <= 0, creates a single batch.

        Returns:
            Table: A Table built from the provided rows.

        Examples:
            >>> table = Table.from_list([{"id": 1}, {"id": 2}, None], batch_size=2)
            >>> len(table.batches)
            2

        Discussion:
            Q. Why allow batching when building from rows?
                Large row lists are expensive to convert in one shot. Splitting into batches makes construction
                incremental and often improves memory behavior.

            Q. What happens when batch_size is None or <= 0?
                All rows are converted into a single RecordBatch.

            Q. What happens when rows is empty?
                It still returns a valid Table with one empty RecordBatch so that schema/operations remain
                well-defined.
        """
        if batch_size is None or batch_size <= 0:
            batch = RecordBatch.from_list(rows, strict_keys=strict_keys)
            return cls([batch])

        n = len(rows)
        if n == 0:
            batch = RecordBatch.from_list([], strict_keys=strict_keys)
            return cls([batch])

        batches: List[RecordBatch] = []
        for start in range(0, n, batch_size):
            chunk = rows[start : start + batch_size]
            batch = RecordBatch.from_list(chunk, strict_keys=strict_keys)
            batches.append(batch)
        return cls(batches)

    @classmethod
    def concat(cls, tables: List["Table"]) -> "Table":
        """
        Concatenate multiple tables by concatenating their RecordBatch lists.

        Args:
            tables (List[Table]): Tables to concatenate.

        Returns:
            Table: A new Table with batches = sum(table.batches, ...).

        Examples:
            >>> out = Table.concat([t1, t2, t3])

        Discussion:
            Q. Why is this concat efficient in a batched columnar design?
                It does not rebuild rows or merge buffers. It simply concatenates batch references, which is
                typically O(number_of_batches).

            Q. What must be true for concat to be valid?
                All input tables must share the same schema (same fields in the same order).
        """
        if not tables:
            raise ValueError("No tables to concatenate.")
        schema = tables[0].schema
        for table in tables:
            if table.schema != schema:
                raise ValueError("All tables must share the same schema to concatenate.")
        batches: List[RecordBatch] = []
        for table in tables:
            batches.extend(table.batches)
        return cls.from_batches(batches)

    def num_rows(self) -> int:
        """
        Return the number of rows in the table.

        Returns:
            int: Total row count.

        Discussion:
            Q. Is this different from len(table)?
                No. It is a named convenience method equivalent to __len__.
        """
        return self.length

    def num_columns(self) -> int:
        """
        Return the number of columns in the table.

        Returns:
            int: Number of schema fields.

        Discussion:
            Q. Why is this derived from schema instead of batches?
                Schema defines the table’s column structure and is shared by all batches.
        """
        return len(self.schema.fields)

    def column_names(self) -> List[str]:
        """
        Return column names in schema order.

        Returns:
            List[str]: Field names.

        Discussion:
            Q. Where does the ordering come from?
                The ordering is defined by schema.fields, which is the canonical column order for the table.
        """
        return self.schema.names()

    def iter_batches(self) -> Iterable[RecordBatch]:
        """
        Iterate over underlying RecordBatches in order.

        Returns:
            Iterable[RecordBatch]: An iterator over batches.

        Examples:
            >>> for batch in table.iter_batches():
            ...     print(batch.num_rows())

        Discussion:
            Q. When would I use iter_batches?
                When you want to process the table in physical chunks, e.g., streaming computation,
                batch-wise export, or batch-wise conversion to other systems.
        """
        return iter(self.batches)

    def column(self, i_or_name) -> List[Array]:
        """
        Fetch a column across all batches.

        Args:
            i_or_name: Column index (int) or column name (str), forwarded to RecordBatch.column.

        Returns:
            List[Array]: One Array per batch, in batch order.

        Examples:
            >>> cols = table.column("id")
            >>> len(cols) == len(table.batches)
            True

        Discussion:
            Q. Why does this return List[Array] instead of a single Array?
                Because the table is physically split into multiple RecordBatches. Each batch stores its own
                column Array segment, and Table exposes them as a list of per-batch pieces.

            Q. If I need one contiguous column, what should I do?
                Typically you would keep it batched, or explicitly materialize/concatenate at the Array level
                depending on your system’s design goals.
        """
        cols: List[Array] = []
        for b in self.batches:
            cols.append(b.column(i_or_name))
        return cols

    def slice(self, offset: int, length: int) -> "Table":
        """
        Return a new Table containing a contiguous range of rows from the global row space.

        Args:
            offset (int): Global start row index (supports negative indexing).
            length (int): Number of rows to include (must be non-negative).

        Returns:
            Table: A new Table made of sliced RecordBatches.

        Examples:
            >>> head = table.slice(0, 10)
            >>> mid = table.slice(100, 50)
            >>> tail = table.slice(-20, 20)

        Discussion:
            Q. What is the main job of Table.slice compared to RecordBatch.slice?
                RecordBatch.slice works in a batch-local index space.
                Table.slice translates a global (table-wide) offset/length into per-batch local slices and
                stitches the resulting sliced batches back into a Table.

            Q. How does it decide which batches contribute to the result?
                It tracks each batch’s global range using (batch_start_global, batch_end_global).
                Batches that end before the requested offset are skipped.
                Once the offset falls inside (or before) a batch, it computes:
                - local_start: where to start inside the batch,
                - local_len: how many rows to take from this batch,
                and keeps consuming batches until the requested length is satisfied.

            Q. Why is there a special case for length == 0?
                It returns an empty Table that preserves schema by creating one empty RecordBatch with the
                same columns (via col.take([])).
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
            first_batch = self.batches[0]
            empty_columns = [col.take([]) for col in first_batch.columns]
            empty_batch = RecordBatch(self.schema, empty_columns)
            return Table.from_batches([empty_batch])

        batch_start_global = 0
        remaining = length
        new_batches: List[RecordBatch] = []
        for batch in self.batches:
            batch_length = batch.length
            batch_end_global = batch_start_global + batch_length
            if batch_end_global <= offset:
                batch_start_global = batch_end_global
                continue
            # Example:
            #   table.slice(10, 25): 10 ~ 35
            #   if batch_start_global is 7, this_batch[3:] is needed.
            #   so local_start is offset - batch_start_global
            local_start = max(0, offset - batch_start_global)
            local_available = batch_length - local_start
            local_len = min(remaining, local_available)
            new_batches.append(batch.slice(local_start, local_len))
            remaining -= local_len
            if remaining <= 0:
                break
            batch_start_global = batch_end_global
        return Table.from_batches(new_batches)

    def take(self, indices: Sequence[int]) -> "Table":
        """
        Select rows by arbitrary global indices and return a new Table.

        Args:
            indices (Sequence[int]): Global row indices in the table coordinate space.

        Returns:
            Table: A new Table composed of per-batch taken RecordBatches.

        Examples:
            >>> out = table.take([0, 10, 3, 3])
            >>> out2 = table.take([-1, -2])

        Discussion:
            Q. What is the main problem this method solves?
                The input indices are global (table-wide), but each RecordBatch only understands local indices.
                Table.take maps global indices to (batch_idx, local_idx) pairs, then calls RecordBatch.take on
                the appropriate batches and reassembles the result as a new Table.

            Q. How does it find which batch contains a global index quickly?
                It precomputes batch_starts = [global_start_of_batch0, global_start_of_batch1, ...].
                Then for each idx it uses bisect_right(batch_starts, idx) - 1 to find the batch whose start
                is the rightmost start <= idx.

            Q. Why normalize indices first?
                To support Python-style negative indexing consistently at the Table level:
                normalized_indices = [normalize_index(idx, n) for idx in indices].

            Q. Why does it group indices by batch (flush logic)?
                Calling RecordBatch.take repeatedly for the same batch would be inefficient.
                This implementation accumulates local indices for the current batch and flushes them as one
                RecordBatch.take call, producing one output batch per contiguous run (in the input order).

            Q. Does this preserve the order of the input indices?
                Yes. It processes indices in the given order and emits batches accordingly.
                If indices are unsorted, the output order follows that unsorted order; grouping may be less
                effective but the semantics remain correct.

            Q. What happens when indices is empty?
                It returns an empty table that preserves schema by creating one empty RecordBatch.
        """
        if not indices:
            first_batch = self.batches[0]
            empty_columns = [col.take([]) for col in first_batch.columns]
            empty_batch = RecordBatch(self.schema, empty_columns)
            return Table.from_batches([empty_batch])
        n = self.length
        normalized_indices = [normalize_index(idx, n) for idx in indices]
        batch_starts: List[int] = []
        current = 0
        for batch in self.batches:
            batch_starts.append(current)
            current += batch.length
        new_batches: List[RecordBatch] = []
        current_batch_idx: Optional[int] = None
        current_local_indices: List[int] = []
        prev_local: Optional[int] = None

        def flush() -> None:
            """Flush the accumulated local indices for the current batch into a new RecordBatch."""
            nonlocal current_batch_idx, current_local_indices, prev_local
            if current_batch_idx is None or not current_local_indices:
                return
            base_batch = self.batches[current_batch_idx]
            new_batches.append(base_batch.take(current_local_indices))
            current_batch_idx = None
            current_local_indices = []
            prev_local = None

        for idx in normalized_indices:
            batch_idx = bisect_right(batch_starts, idx) - 1
            if batch_idx < 0 or batch_idx >= len(self.batches):
                raise IndexError(f"Global index {idx} not mapped to any batch.")
            local_idx = idx - batch_starts[batch_idx]
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
        """
        Select a subset of columns by name and return a new Table.

        Args:
            names (List[str]): Column names to keep (in the desired order).

        Returns:
            Table: A new Table whose batches are batch.select(names).

        Examples:
            >>> small = table.select(["id", "name"])

        Discussion:
            Q. How is this implemented for a batched table?
                It applies RecordBatch.select(names) to every batch and wraps the resulting batches into
                a new Table. The operation is batch-local and does not require row materialization.

            Q. Does select change row count?
                No. It only changes which columns are present; row count is preserved.
        """
        new_batches = [b.select(names) for b in self.batches]
        return Table.from_batches(new_batches)

    def to_list(self) -> List[Dict[str, Any]]:
        """
        Convert the entire table into a row-oriented representation (list of dicts).

        Returns:
            List[Dict[str, Any]]: Rows as Python dicts mapping field name -> Python value.

        Examples:
            >>> rows = table.to_list()
            >>> rows[0]
            {'id': 1, 'name': 'a'}

        Discussion:
            Q. Why is to_list considered a materialization step?
                The table is stored as columnar Arrays. Converting to list[dict] creates Python objects for
                each cell and loses the compact buffer-based representation, which is slower and uses more memory.

            Q. How does it handle multiple batches?
                It iterates batches in order, converts each batch's columns to Python lists, then emits rows
                for that batch and extends the global output list.

            Q. What happens when a batch has zero columns?
                It emits empty dicts for each row in that batch, preserving row count even when schema has
                no fields.
        """
        rows: List[Dict[str, Any]] = []
        for batch in self.batches:
            if batch.num_columns() == 0:
                rows.extend({} for _ in range(batch.length))
                continue
            columns = [c.to_list() for c in batch.columns]
            for row in range(batch.length):
                row_dict: Dict[str, Any] = {}
                for field, column in zip(batch.schema.fields, columns):
                    row_dict[field.name] = column[row]
                rows.append(row_dict)
        return rows