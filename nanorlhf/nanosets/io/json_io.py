import json
from typing import Any, Dict, Iterable, List, Optional, TextIO, Union

from nanorlhf.nanosets.table.record_batch import RecordBatch
from nanorlhf.nanosets.table.table import Table
from nanorlhf.nanosets.utils import DEFAULT_BATCH_SIZE


Row = Optional[Dict[str, Any]]
TableLike = Union[Table, RecordBatch]


def iter_rows(obj: TableLike) -> Iterable[Row]:
    if isinstance(obj, RecordBatch):
        for row in obj.to_pylist():
            yield row
        return
    if isinstance(obj, Table):
        for batch in obj.batches:
            for row in batch.to_pylist():
                yield row
        return
    raise TypeError(f"Unsupported object: {type(obj).__name__}")


def materialize(obj: TableLike) -> List[Row]:
    return list(iter_rows(obj))


def to_json(
    fp: TextIO,
    obj: TableLike,
    indent: Optional[int] = 2,
) -> None:
    rows = materialize(obj)
    json.dump(rows, fp, ensure_ascii=False, indent=indent)


def to_jsonl(fp: TextIO, obj: TableLike) -> None:
    for row in iter_rows(obj):
        fp.write(json.dumps(row, ensure_ascii=False))
        fp.write("\n")


def from_json(path: str, batch_size: Optional[int] = DEFAULT_BATCH_SIZE) -> Table:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError("JSON root must be a list of rows (rows-only).")
    return Table.from_list(data, batch_size=batch_size)


def from_jsonl(path: str, batch_size: Optional[int] = DEFAULT_BATCH_SIZE) -> Table:
    rows: List[Row] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return Table.from_list(rows, batch_size=batch_size)
