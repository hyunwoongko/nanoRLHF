import os
import random
from typing import List, Optional, Union, Callable, Dict, Any, Sequence

from nanorlhf.nanosets.io.ipc import read_table, write_table
from nanorlhf.nanosets.io.json_io import from_json, from_jsonl, to_json, to_jsonl
from nanorlhf.nanosets.table.record_batch import RecordBatch
from nanorlhf.nanosets.table.table import Table
from nanorlhf.nanosets.utils import DEFAULT_BATCH_SIZE, ext


class Dataset:
    """
    A thin convenience wrapper around `Table`.

    A `Dataset` stores a single `Table` and provides user-facing helpers that are common in
    dataset workflows: indexing, selection, shuffling, mapping, filtering, and saving/loading.

    Discussion:
        Q. Why introduce Dataset when Table already exists?
            `Table` is the core columnar container and focuses on Arrow-like operations
            (batched structure, columnar arrays, slice/take/select at the table level).
            `Dataset` adds convenience APIs that are often expected by end users:
            - `dataset[i]` returning a row dict
            - `shuffle`, `map`, `filter`
            - simple serialization helpers

        Q. What is the primary design trade-off of Dataset?
            Some methods (`map`, `filter`, `__getitem__`, JSON IO) materialize row dicts
            via `to_list()` / `to_dict()`. This improves usability and flexibility but can
            increase memory usage compared to staying purely in columnar form.
    """

    def __init__(self, table: Table):
        """
        Create a Dataset from an existing Table.

        Args:
            table (Table): The underlying table.

        Discussion:
            Q. Does Dataset copy data from Table?
                No. `Dataset` only stores a reference to the provided `Table`.
                Any sharing/zero-copy behavior depends on how the `Table` was constructed.
        """
        self.table = table

    def __getitem__(self, item):
        """
        Index or slice the dataset using Python indexing semantics.

        Returns:
            - If item is int: returns a single row as a dict (materialized).
            - If item is slice: returns a list of row dicts (materialized).

        Examples:
            >>> ds = Dataset.from_list([{"x": 1}, {"x": 2}, {"x": 3}])
            >>> ds[0]
            {'x': 1}
            >>> ds[-1]
            {'x': 3}
            >>> ds[1:3]
            [{'x': 2}, {'x': 3}]
            >>> ds[0:3:2]
            [{'x': 1}, {'x': 3}]

        Discussion:
            Q. Why does integer indexing return a row dict instead of a Table/Dataset?
                This method is a convenience API. It selects one row and materializes it
                into a Python dict for interactive use.

            Q. How is slicing implemented?
                The slice is expanded into an explicit index list using `range(*slice.indices(...))`,
                then delegated to `select(indices)` and materialized via `to_dict()`.

            Q. Are negative indices supported?
                Yes for integer indexing. Negative indices are normalized into [0, n).
        """
        if isinstance(item, int):
            n = len(self)
            if item < 0:
                item += n
            if not (0 <= item < n):
                raise IndexError(f"Index {item} out of range [0, {n})")
            return self.select([item]).to_dict()[0]
        elif isinstance(item, slice):
            indices = list(range(*item.indices(len(self))))
            return self.select(indices).to_dict()
        else:
            raise TypeError("Invalid argument type.")

    def __len__(self) -> int:
        """
        Return the number of rows in the dataset.

        Returns:
            int: Total number of rows.

        Discussion:
            Q. What does len(dataset) measure?
                It measures the logical row count of the underlying `Table`.
        """
        return self.table.length

    def __repr__(self) -> str:
        """
        Return a concise string representation.

        Returns:
            str: Representation including num_rows and schema.

        Discussion:
            Q. What is shown here and why?
                The row count and schema are typically the two most useful quick facts
                when inspecting a dataset interactively.
        """
        return f"Dataset(num_rows={len(self)}, schema={self.table.schema})"

    @classmethod
    def from_list(
        cls,
        rows: List[Optional[Dict[str, Any]]],
        strict_keys: bool = False,
        batch_size: Optional[int] = DEFAULT_BATCH_SIZE,
    ) -> "Dataset":
        """
        Construct a Dataset from row-oriented data (list of dicts / None).

        Args:
            rows (List[Optional[Dict[str, Any]]]): Row dicts (or None) to convert into a columnar dataset.
            strict_keys (bool): Passed through to `Table.from_list` (and transitively to `StructArray.from_list`).
            batch_size (Optional[int]): If positive, splits rows into batches of at most batch_size.
                If None or <= 0, creates a single batch.

        Returns:
            Dataset: A Dataset backed by a Table built from the provided rows.

        Examples:
            >>> ds = Dataset.from_list([{"id": 1}, {"id": 2}, None], batch_size=2)
            >>> len(ds)
            3

        Discussion:
            Q. Why accept row-oriented input in a columnar system?
                Row dicts are the most common interchange format at Python boundaries.
                `from_list` provides an ergonomic entry point and converts to columnar form internally.

            Q. Why allow batching here?
                Converting a very large row list at once can be expensive.
                Splitting into batches makes construction incremental and often improves memory behavior.
        """
        return cls(Table.from_list(rows, strict_keys=strict_keys, batch_size=batch_size))

    def save_to_disk(self, path: str) -> None:
        """
        Save the dataset to disk in nanosets IPC format (`.nano`).

        Args:
            path (str): Output path.

        Examples:
            >>> ds = Dataset.from_list([{"x": 1}, {"x": 2}])
            >>> ds.save_to_disk("data/out.nano")

        Discussion:
            Q. What is stored on disk?
                The underlying `Table` is serialized using `write_table`, which writes a compact
                binary format with a JSON header and raw buffer blobs.

            Q. Why is this format useful?
                The corresponding `read_table` path can use `mmap` and buffer slicing to
                reconstruct arrays with minimal copying.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "wb") as fp:
            write_table(fp, self.table)

    def to_json(self, path: str, lines: bool = True) -> None:
        """
        Export the dataset to JSON or JSONL.

        Args:
            path (str): Output path.
            lines (bool): If True, writes JSONL (one row per line). If False, writes a JSON array.

        Examples:
            >>> ds = Dataset.from_list([{"x": 1}, {"x": 2}])
            >>> ds.to_json("data/out.jsonl", lines=True)
            >>> ds.to_json("data/out.json", lines=False)

        Discussion:
            Q. Why provide both JSON and JSONL?
                JSON is a single document (often convenient for small datasets),
                while JSONL is streaming-friendly and common for larger datasets.

            Q. Does this require materialization?
                The JSON writers operate on row-oriented data, so exporting typically involves
                converting columnar buffers into Python values.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "w", encoding="utf-8") as fp:
            if lines:
                to_jsonl(fp, self.table)
            else:
                to_json(fp, self.table)

    def to_dict(self) -> List[Optional[dict]]:
        """
        Convert the dataset to a row-oriented representation.

        Returns:
            List[Optional[dict]]: Rows as Python dicts (and possibly None rows depending on source data).

        Examples:
            >>> ds = Dataset.from_list([{"x": 1}, None, {"x": 3}])
            >>> ds.to_dict()
            [{'x': 1}, None, {'x': 3}]

        Discussion:
            Q. Why is this considered materialization?
                The dataset is stored as columnar arrays, but this returns Python objects per cell.
                It is convenient for Python interop and debugging, but it may be slower and use more memory.
        """
        return self.table.to_list()

    def select_columns(self, column_names: List[str]) -> "Dataset":
        """
        Select a subset of columns by name.

        Args:
            column_names (List[str]): Column names to keep (in the desired order).

        Returns:
            Dataset: A new Dataset with only the selected columns.

        Examples:
            >>> ds = Dataset.from_list([{"a": 1, "b": 2}, {"a": 3, "b": 4}])
            >>> ds2 = ds.select_columns(["b"])
            >>> ds2.to_dict()
            [{'b': 2}, {'b': 4}]

        Discussion:
            Q. Does this operation change row count?
                No. It only changes which columns are present.

            Q. Does this materialize rows?
                No. It delegates to `Table.select`, which operates on schema/column references.
        """
        return Dataset(self.table.select(column_names))

    def remove_columns(self, column_names: List[str]) -> "Dataset":
        """
        Remove a subset of columns by name.

        Args:
            column_names (List[str]): Column names to drop.

        Returns:
            Dataset: A new Dataset without the dropped columns.

        Examples:
            >>> ds = Dataset.from_list([{"a": 1, "b": 2}, {"a": 3, "b": 4}])
            >>> ds2 = ds.remove_columns(["a"])
            >>> ds2.to_dict()
            [{'b': 2}, {'b': 4}]

        Discussion:
            Q. How is this implemented?
                It computes the complement set of columns to keep and calls `Table.select(keep)`.

            Q. Does this materialize rows?
                No. This is a schema/column selection operation.
        """
        all_names = self.table.column_names()
        drop_set = set(column_names)
        keep = [name for name in all_names if name not in drop_set]
        return Dataset(self.table.select(keep))

    def select(self, indices: Sequence[int]) -> "Dataset":
        """
        Select rows by global indices.

        Args:
            indices (Sequence[int]): Global row indices in dataset coordinate space.

        Returns:
            Dataset: A new Dataset containing the selected rows.

        Examples:
            >>> ds = Dataset.from_list([{"x": 1}, {"x": 2}, {"x": 3}])
            >>> ds.select([2, 0]).to_dict()
            [{'x': 3}, {'x': 1}]

        Discussion:
            Q. What does this delegate to?
                It calls `Table.take(indices)` and wraps the resulting Table.

            Q. Does it preserve the order of indices?
                Yes. `Table.take` processes indices in input order and produces rows accordingly.
        """
        return Dataset(self.table.take(indices))

    def shuffle(self, seed: Optional[int] = None) -> "Dataset":
        """
        Return a shuffled copy of the dataset.

        Args:
            seed (Optional[int]): Random seed for reproducible shuffling.

        Returns:
            Dataset: A new Dataset with rows permuted.

        Examples:
            >>> ds = Dataset.from_list([{"x": 1}, {"x": 2}, {"x": 3}])
            >>> ds2 = ds.shuffle(seed=0)
            >>> len(ds2) == len(ds)
            True

        Discussion:
            Q. How is shuffling implemented?
                It creates an index list [0..n-1], shuffles it using `random.Random(seed)`,
                then selects rows via `select(shuffled_indices)`.

            Q. Is this a stable/streaming shuffle?
                No. It builds an index list in memory, so it is a simple full permutation shuffle.
        """
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
        """
        Apply a function to rows and return a new dataset.

        Args:
            function (Callable): Mapping function.
                - If batched=False: called as `function(row_dict)` and must return a row dict.
                - If batched=True: called as `function(list_of_row_dicts)` and must return a list of row dicts.
            batched (bool): Whether to call the function on batches (lists of rows) instead of single rows.
            batch_size (Optional[int]): Buffer size used when batched=True. If None or <= 0, the buffer
                grows to include all rows before flushing.

        Returns:
            Dataset: A new Dataset built from mapped rows.

        Examples:
            >>> ds = Dataset.from_list([{"x": 1}, {"x": 2}])
            >>> ds2 = ds.map(lambda r: {"y": r["x"] + 10})
            >>> ds2.to_dict()
            [{'y': 11}, {'y': 12}]

            >>> ds3 = ds.map(lambda rows: [{"y": r["x"] * 2} for r in rows], batched=True, batch_size=2)
            >>> ds3.to_dict()
            [{'y': 2}, {'y': 4}]

        Discussion:
            Q. Why does map materialize rows?
                The mapping function can be arbitrary Python code, so the implementation converts each
                RecordBatch to `list[dict]` and calls the function in Python space.

            Q. What happens to schema/types?
                The output dataset is rebuilt via `RecordBatch.from_list` on the mapped rows, so schema
                is inferred from the resulting row dicts.

            Q. What does batched=True change?
                It buffers rows across batches and calls `function(buffer)` when the buffer reaches
                `batch_size` (or at the end), which can reduce Python call overhead.
        """
        new_batches: List[RecordBatch] = []
        if not batched:
            for batch in self.table.batches:
                rows = batch.to_list()
                out_rows: List[Optional[Dict[str, Any]]] = []
                for row in rows:
                    out_rows.append(function(row))
                new_batches.append(RecordBatch.from_list(out_rows))
        else:
            actual_bs = batch_size if batch_size is not None and batch_size > 0 else None
            buffer: List[Dict[str, Any]] = []

            def flush(force: bool = False) -> None:
                """
                Flush buffered rows by applying `function` and emitting a new RecordBatch.

                Discussion:
                    Q. Why flush conditionally?
                        When batched=True and batch_size is set, we want to call the function only when
                        enough rows have accumulated, to amortize Python overhead.

                    Q. What does force=True do?
                        It flushes any remaining rows at the end, even if the buffer is smaller than batch_size.
                """
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

            for batch in self.table.batches:
                rows = batch.to_list()
                for row in rows:
                    buffer.append(row)
                    flush(False)
            flush(True)
        return Dataset(Table.from_batches(new_batches))

    def filter(
        self,
        predicate: Callable[[Dict[str, Any]], bool],
        batch_size: Optional[int] = DEFAULT_BATCH_SIZE,
    ) -> "Dataset":
        """
        Filter rows by a predicate and return a new dataset.

        Args:
            predicate (Callable[[Dict[str, Any]], bool]): Function that returns True to keep a row.
            batch_size (Optional[int]): Number of kept rows per output RecordBatch. If None or <= 0,
                all kept rows are emitted as one batch (subject to input size).

        Returns:
            Dataset: A new Dataset containing only rows that pass the predicate.

        Examples:
            >>> ds = Dataset.from_list([{"x": 1}, {"x": 2}, {"x": 3}])
            >>> ds2 = ds.filter(lambda r: r["x"] % 2 == 1)
            >>> ds2.to_dict()
            [{'x': 1}, {'x': 3}]

        Discussion:
            Q. Why does filter materialize rows?
                The predicate is arbitrary Python code, so rows are converted to dicts to call it.

            Q. What happens to None rows?
                Rows that are None are skipped and never passed to the predicate.

            Q. Why rebuild batches with RecordBatch.from_list?
                The filtered rows are collected and then converted back into columnar form.
                `batch_size` controls the granularity of the output batches.

            Q. What if no rows pass the predicate?
                It returns an empty dataset that preserves schema by constructing an empty RecordBatch
                using `col.take([])` for each column of the first batch.
        """
        new_batches: List[RecordBatch] = []
        buffer: List[Optional[Dict[str, Any]]] = []
        batch_size = batch_size if batch_size is not None and batch_size > 0 else None

        for batch in self.table.batches:
            rows = batch.to_list()
            for row in rows:
                if row is None:
                    continue
                if predicate(row):
                    buffer.append(row)
                    if batch_size is not None and len(buffer) >= batch_size:
                        new_batches.append(RecordBatch.from_list(buffer))
                        buffer = []

        if buffer:
            new_batches.append(RecordBatch.from_list(buffer))

        if not new_batches:
            first_batch = self.table.batches[0]
            empty_cols = [col.take([]) for col in first_batch.columns]
            empty_batch = RecordBatch(self.table.schema, empty_cols)
            return Dataset(Table.from_batches([empty_batch]))

        return Dataset(Table.from_batches(new_batches))


def load_dataset(
    data_files: Union[str, List[str]],
    batch_size: Optional[int] = DEFAULT_BATCH_SIZE,
) -> Dataset:
    """
    Load one or more data files into a `Dataset`.

    Supported formats are selected by file extension:
    - `.json` (row-oriented JSON array)
    - `.jsonl` / `.ndjson` (row-oriented JSON Lines)
    - `.nano` (nanosets IPC format)

    Args:
        data_files (Union[str, List[str]]): One file path or a list of file paths.
        batch_size (Optional[int]): Batch size used by JSON/JSONL loaders. If None or <= 0, loaders
            may produce a single batch depending on their implementation.

    Returns:
        Dataset: Loaded dataset. If multiple files are provided, they are concatenated.

    Examples:
        >>> ds = load_dataset("data/train.jsonl", batch_size=1024)
        >>> ds2 = load_dataset(["a.nano", "b.nano"])
        >>> len(ds2) >= 0
        True

    Discussion:
        Q. Why use extension-based dispatch?
            It keeps the user API simple: the caller provides file paths and the loader picks
            the correct parser/reader.

        Q. How are multiple files handled?
            Each file is loaded into a `Table`. If there are multiple tables, they are combined
            via `Table.concat`, which concatenates RecordBatch lists without rebuilding row data.

        Q. Why does batch_size apply only to JSON paths?
            `.nano` is already stored as a batched columnar table; JSON inputs must be parsed and
            converted into columnar form, where batching is a construction choice.
    """

    def _load_one(file: str) -> Table:
        """
        Load a single file into a `Table` based on extension.

        Args:
            file (str): Path to one data file.

        Returns:
            Table: Loaded table.

        Discussion:
            Q. Why return Table here instead of Dataset?
                `_load_many` is responsible for concatenation across multiple inputs, so `_load_one`
                returns the lower-level `Table` to make that composition simple.
        """
        e = ext(file)
        if e == "json":
            return from_json(file, batch_size=batch_size)
        if e in ("jsonl", "ndjson"):
            return from_jsonl(file, batch_size=batch_size)
        if e == "nano":
            return read_table(file)
        raise ValueError(f"Unsupported extension for {file!r}. Expected .json, .jsonl/.ndjson, or .nano")

    def _load_many(files: Union[str, List[str]]) -> Dataset:
        """
        Load one or many files and return a Dataset.

        Args:
            files (Union[str, List[str]]): File path(s).

        Returns:
            Dataset: Dataset over the loaded (and possibly concatenated) table.

        Discussion:
            Q. Why normalize files into a list?
                It allows a single code path to handle both a single string path and a list of paths.

            Q. When is Table.concat used?
                When more than one file is provided, the loaded tables are concatenated to form
                a single logical dataset.
        """
        flist = [files] if isinstance(files, str) else list(files)
        tables = [_load_one(f) for f in flist]
        table = Table.concat(tables) if len(tables) > 1 else tables[0]
        return Dataset(table)

    if isinstance(data_files, (str, list)):
        return _load_many(data_files)
    raise TypeError("data_files must be str or list[str].")


load_from_disk = load_dataset
