import json
from typing import Any, Dict, Iterable, List, Optional, TextIO, Union

from nanorlhf.nanosets.table.record_batch import RecordBatch
from nanorlhf.nanosets.table.table import Table
from nanorlhf.nanosets.utils import DEFAULT_BATCH_SIZE


Row = Optional[Dict[str, Any]]
TableLike = Union[Table, RecordBatch]


def iter_rows(obj: TableLike) -> Iterable[Row]:
    """
    Iterate over rows of a `Table` or `RecordBatch` as Python objects.

    Args:
        obj (TableLike): Input object. Must be a `Table` or `RecordBatch`.

    Returns:
        Iterable[Row]: An iterator that yields each row as either a `dict` (row data) or `None` (null row).

    Examples:
        >>> for row in iter_rows(table):
        ...     print(row)
        >>> for row in iter_rows(record_batch):
        ...     print(row)

    Discussion:
        Q. Why return an iterator instead of a list?
            Iteration avoids materializing all rows at once, which can reduce peak memory usage when
            streaming rows into another sink (e.g., JSONL writing).

        Q. What is a "row" here?
            A row is represented as a Python `dict` mapping column names to values, or `None` if the
            row is null at the struct level (depending on the underlying representation).

        Q. Why handle `RecordBatch` and `Table` separately?
            A `RecordBatch` directly contains rows for a single batch, while a `Table` is a collection
            of batches. Iteration over a `Table` flattens across all batches in order.
    """
    if isinstance(obj, RecordBatch):
        for row in obj.to_list():
            yield row
        return
    if isinstance(obj, Table):
        for batch in obj.batches:
            for row in batch.to_list():
                yield row
        return
    raise TypeError(f"Unsupported object: {type(obj).__name__}")


def materialize(obj: TableLike) -> List[Row]:
    """
    Materialize all rows from a `Table` or `RecordBatch` into a Python list.

    Args:
        obj (TableLike): Input object. Must be a `Table` or `RecordBatch`.

    Returns:
        List[Row]: A list of rows, where each row is a `dict` or `None`.

    Examples:
        >>> rows = materialize(table)
        >>> len(rows)
        >>> rows[0]

    Discussion:
        Q. When should I use this instead of `iter_rows`?
            Use this when you need random access, repeated passes, or to pass rows into APIs that
            require an in-memory list.

        Q. What is the cost of materialization?
            Materialization allocates Python objects for every row, which can increase memory usage
            compared to iterating and processing rows one by one.
    """
    return list(iter_rows(obj))


def to_json(
    fp: TextIO,
    obj: TableLike,
    indent: Optional[int] = 2,
) -> None:
    """
    Write a `Table` or `RecordBatch` to JSON as a rows-only array.

    Args:
        fp (TextIO): A writable text file-like object.
        obj (TableLike): Input object to serialize (`Table` or `RecordBatch`).
        indent (Optional[int]): JSON indentation level. Use `None` for compact output.

    Returns:
        None

    Examples:
        >>> with open("data.json", "w", encoding="utf-8") as f:
        ...     to_json(f, table, indent=2)

    Discussion:
        Q. What does "rows-only" JSON mean?
            The JSON root is a list where each element is one row represented as a `dict` (or `null`).
            Schema and columnar buffers are not preserved; this is a human-friendly interchange format.

        Q. Why use `ensure_ascii=False`?
            This keeps non-ASCII characters (e.g., Korean) readable in the output instead of escaping them.

        Q. Why does this materialize rows first?
            JSON array output generally requires a complete list structure at serialization time.
            This implementation builds the list via `materialize(obj)` before calling `json.dump`.
    """
    rows = materialize(obj)
    json.dump(rows, fp, ensure_ascii=False, indent=indent)


def to_jsonl(fp: TextIO, obj: TableLike) -> None:
    """
    Write a `Table` or `RecordBatch` to JSON Lines (one JSON object per line).

    Args:
        fp (TextIO): A writable text file-like object.
        obj (TableLike): Input object to serialize (`Table` or `RecordBatch`).

    Returns:
        None

    Examples:
        >>> with open("data.jsonl", "w", encoding="utf-8") as f:
        ...     to_jsonl(f, table)

    Discussion:
        Q. Why use JSONL instead of JSON?
            JSONL supports streaming: each row is written independently, making it more suitable for
            large datasets and incremental processing.

        Q. Does this avoid materialization?
            Yes. It iterates rows via `iter_rows(obj)` and writes each row immediately, which keeps
            memory usage lower than building a full list first.

        Q. How are null rows represented?
            A null row is serialized as the JSON literal `null` on its own line.
    """
    for row in iter_rows(obj):
        fp.write(json.dumps(row, ensure_ascii=False))
        fp.write("\n")


def from_json(path: str, batch_size: Optional[int] = DEFAULT_BATCH_SIZE) -> Table:
    """
    Load a rows-only JSON file into a `Table`.

    Args:
        path (str): Path to the JSON file. The JSON root must be a list of rows.
        batch_size (Optional[int]): Batch size used by `Table.from_list`. If `None`, implementation-defined
            default behavior is used.

    Returns:
        Table: A `Table` constructed from the loaded rows.

    Examples:
        >>> table = from_json("data.json")
        >>> table = from_json("data.json", batch_size=1024)

    Discussion:
        Q. What JSON shape does this function expect?
            It expects a JSON array (list) at the root, where each element is a row `dict` or `null`.

        Q. Why is `batch_size` needed?
            `Table` is internally organized into `RecordBatch`es. `batch_size` controls how many rows
            are grouped per batch when reconstructing the table from row materialization.

        Q. What is the tradeoff of this format?
            Rows-only JSON is simple and portable, but it does not preserve Arrow-style buffers,
            dtypes, or zero-copy storage. It is intended for interchange and readability.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError("JSON root must be a list of rows (rows-only).")
    return Table.from_list(data, batch_size=batch_size)


def from_jsonl(path: str, batch_size: Optional[int] = DEFAULT_BATCH_SIZE) -> Table:
    """
    Load a JSON Lines file into a `Table`.

    Args:
        path (str): Path to the JSONL file. Each non-empty line must be a JSON value representing a row.
        batch_size (Optional[int]): Batch size used by `Table.from_list`. If `None`, implementation-defined
            default behavior is used.

    Returns:
        Table: A `Table` constructed from the loaded rows.

    Examples:
        >>> table = from_jsonl("data.jsonl")
        >>> table = from_jsonl("data.jsonl", batch_size=2048)

    Discussion:
        Q. Why strip and skip empty lines?
            JSONL files in practice may contain trailing newlines or blank lines; skipping them makes the reader
            more tolerant and convenient.

        Q. Does this implementation stream into the table builder?
            No. It collects rows into a Python list first, then calls `Table.from_list`. This keeps the code simple,
            but it means peak memory usage grows with the number of rows.

        Q. What is the expected row representation?
            Each line should be a JSON object (row dict) or the literal `null` for a null row.
    """
    rows: List[Row] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return Table.from_list(rows, batch_size=batch_size)