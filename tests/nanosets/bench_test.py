import argparse
import json
import os
import random
import sys
import time
from contextlib import contextmanager
from typing import Dict

from tabulate import tabulate

from nanorlhf.nanosets.dtype.list_array import ListArray
from nanorlhf.nanosets.dtype.primitive_array import PrimitiveArray
from nanorlhf.nanosets.dtype.string_array import StringArray
from nanorlhf.nanosets.dtype.struct_array import StructArray
from nanorlhf.nanosets.io.ipc import write_table, read_table
from nanorlhf.nanosets.table.record_batch import RecordBatch
from nanorlhf.nanosets.table.table import Table


def gen_rows(n: int = 1000, seed: int = 42):
    """Generate synthetic nested rows for benchmarking."""
    random.seed(seed)
    rows = []
    for i in range(n):
        rows.append({
            "id": i,
            "name": f"user_{i}",
            "score": None if i % 7 == 0 else i / 10,
            "tags": None if i % 10 == 0 else [f"t{i % 3}", f"t{(i + 1) % 3}"],
            "metrics": None if i % 13 == 0 else {"clicks": i * 2, "rate": (i % 5) / 5.0},
            "nested": None if i % 9 == 0 else [[i, i + 1], []],
        })
    return rows


@contextmanager
def timer():
    """Context manager returning a lambda for elapsed seconds."""
    t0 = time.perf_counter()
    try:
        yield lambda: time.perf_counter() - t0
    finally:
        pass


def human_bytes(n: int) -> str:
    """Human-readable byte size."""
    size = float(n)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024.0:
            return f"{size:,.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} PB"


def size_of(path: str) -> str:
    """Return human-readable file size if exists, otherwise '0 B'."""
    return human_bytes(os.path.getsize(path)) if os.path.exists(path) else "0 B"


def deep_size(obj, seen=None) -> int:
    """Rough recursive size of Python objects to compare materialized lists."""
    if seen is None:
        seen = set()
    oid = id(obj)
    if oid in seen:
        return 0
    seen.add(oid)
    size = sys.getsizeof(obj)
    if isinstance(obj, dict):
        for k, v in obj.items():
            size += deep_size(k, seen)
            size += deep_size(v, seen)
    elif isinstance(obj, (list, tuple, set, frozenset)):
        for x in obj:
            size += deep_size(x, seen)
    # strings/bytes/bytearray/ints/floats/bools are accounted by sys.getsizeof
    return size


def ensure_json(path_json: str, rows):
    """Write JSON once if file does not exist."""
    if not os.path.exists(path_json):
        with open(path_json, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, separators=(",", ":"))


def ensure_jsonl(path_jsonl: str, rows):
    """Write JSONL once if file does not exist."""
    if not os.path.exists(path_jsonl):
        with open(path_jsonl, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")


def read_json(path: str):
    """Load JSON file into Python list."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: str):
    """Load JSONL file fully into Python list."""
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                out.append(json.loads(s))
    return out


def partial_from_rows(rows, field: str):
    """Extract a single field from fully loaded Python objects."""
    return [r.get(field) for r in rows]


def partial_from_jsonl_stream(path: str, field: str):
    """Stream JSONL and extract only the requested field into a list."""
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            obj = json.loads(s)
            out.append(obj.get(field))
    return out


def ensure_nano(path_nano: str, rows):
    """Write NANO-IPC file once if file does not exist."""
    if not os.path.exists(path_nano):
        batch = RecordBatch.from_list(rows)
        table = Table([batch])
        with open(path_nano, "wb") as fp:
            write_table(fp, table)


def read_nano_table(path: str) -> Table:
    """Read NANO-IPC table with header parse + zero-copy mmap."""
    return read_table(path)


def partial_from_nano_column(table: Table, column_name: str):
    """Extract one column across all batches without materializing the whole table."""
    idx = table.schema.index(column_name)
    out = []
    for b in table.batches:
        col = b.columns[idx]
        out.extend(col.to_list())
    return out


def array_physical_bytes(arr) -> int:
    """Sum physical bytes of buffers that back an Array (recursive)."""
    total = 0
    validity = getattr(arr, "validity", None)
    if validity is not None:
        total += len(validity.buf.data)
    if isinstance(arr, PrimitiveArray):
        total += len(arr.values.data)
    elif isinstance(arr, StringArray):
        total += len(arr.offsets.data) + len(arr.values.data)
    elif isinstance(arr, ListArray):
        total += len(arr.offsets.data)
        total += array_physical_bytes(arr.values)
    elif isinstance(arr, StructArray):
        for ch in arr.children:
            total += array_physical_bytes(ch)
    return total


def nano_column_physical_bytes(table: Table, column_name: str) -> int:
    """Sum backing buffer bytes for a named column across batches."""
    idx = table.schema.index(column_name)
    total = 0
    for b in table.batches:
        total += array_physical_bytes(b.columns[idx])
    return total


def print_benchmark_tables(
    filesizes: Dict[str, str],
    times: Dict[str, float],
    sizes: Dict[str, str],
):
    """Render a single summary table using tabulate (no file list section)."""
    headers = ["Format", "File Size", "Read (s)", "Partial (s)", "Materialized Size", "Column Buffers"]
    rows = [
        ["JSON", filesizes["json"], f"{times['read_json']:.3f}", f"{times['partial_json']:.3f}", sizes["part_json"],
         "-"],
        ["JSONL", filesizes["jsonl"], f"{times['read_jsonl']:.3f}", f"{times['partial_jsonl']:.3f}",
         sizes["part_jsonl"], "-"],
        ["JSONL (stream field)", filesizes["jsonl"], "—", f"{times['partial_jsonl_stream']:.3f}",
         sizes["part_jsonl_stream"], f"≈ file size {filesizes['jsonl']}"],
        ["NANO-IPC", filesizes["nano"], f"{times['read_nano']:.3f}", f"{times['partial_nano']:.3f}", sizes["part_nano"],
         sizes["nano_col_bytes"]],
    ]

    print("=== Read + Partial Access Benchmark ===\n")
    print(tabulate(rows, headers=headers, tablefmt="github"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=100_000, help="rows to generate if files are absent")
    ap.add_argument("--out", type=str, default="data/bench_test", help="output directory (data storage)")
    ap.add_argument("--field", type=str, default="name", help="field/column to partially read")
    ap.add_argument("--verify", action="store_true", help="verify equality across formats for the requested field")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    p_json = os.path.join(args.out, "data.json")
    p_jsonl = os.path.join(args.out, "data.jsonl")
    p_nano = os.path.join(args.out, "table.nano")

    # Ensure files exist (generate once)
    rows = gen_rows(args.rows)
    ensure_json(p_json, rows)
    ensure_jsonl(p_jsonl, rows)
    ensure_nano(p_nano, rows)

    # JSON: full read, then partial extraction
    with timer() as t:
        rows_json = read_json(p_json)
    t_read_json = t()
    with timer() as t:
        part_json = partial_from_rows(rows_json, args.field)
    t_partial_json = t()
    size_part_json = deep_size(part_json)

    # JSONL: full read, then partial extraction (and streaming variant)
    with timer() as t:
        rows_jsonl = read_jsonl(p_jsonl)
    t_read_jsonl = t()
    with timer() as t:
        part_jsonl = partial_from_rows(rows_jsonl, args.field)
    t_partial_jsonl = t()
    size_part_jsonl = deep_size(part_jsonl)

    with timer() as t:
        part_jsonl_stream = partial_from_jsonl_stream(p_jsonl, args.field)
    t_partial_jsonl_stream = t()
    size_part_jsonl_stream = deep_size(part_jsonl_stream)

    # NANO-IPC: table read (zero-copy), then single-column materialization
    with timer() as t:
        table_nano = read_nano_table(p_nano)
    t_read_nano = t()
    with timer() as t:
        part_nano = partial_from_nano_column(table_nano, args.field)
    t_partial_nano = t()
    size_part_nano = deep_size(part_nano)
    bytes_nano_field = nano_column_physical_bytes(table_nano, args.field)

    # Optional verification
    if args.verify:
        assert part_json == part_jsonl == part_jsonl_stream == part_nano, "Partial results mismatch across formats"

    # Compose rows for tabulate
    filesizes = {
        "json": size_of(p_json),
        "jsonl": size_of(p_jsonl),
        "nano": size_of(p_nano),
    }
    times = {
        "read_json": t_read_json,
        "read_jsonl": t_read_jsonl,
        "read_nano": t_read_nano,
        "partial_json": t_partial_json,
        "partial_jsonl": t_partial_jsonl,
        "partial_jsonl_stream": t_partial_jsonl_stream,
        "partial_nano": t_partial_nano,
    }
    sizes = {
        "part_json": human_bytes(size_part_json),
        "part_jsonl": human_bytes(size_part_jsonl),
        "part_jsonl_stream": human_bytes(size_part_jsonl_stream),
        "part_nano": human_bytes(size_part_nano),
        "nano_col_bytes": human_bytes(bytes_nano_field),
    }

    # Pretty tables
    print_benchmark_tables(
        filesizes=filesizes,
        times=times,
        sizes=sizes,
    )


if __name__ == "__main__":
    main()

# === Read + Partial Access Benchmark ===
#
# | Format               | File Size   | Read (s)   |   Partial (s) | Materialized Size   | Column Buffers      |
# |----------------------|-------------|------------|---------------|---------------------|---------------------|
# | JSON                 | 12.3 MB     | 0.421      |         0.011 | 6.4 MB              | -                   |
# | JSONL                | 12.3 MB     | 0.360      |         0.017 | 6.4 MB              | -                   |
# | JSONL (stream field) | 12.3 MB     | —          |         0.178 | 6.4 MB              | ≈ file size 12.3 MB |
# | NANO-IPC             | 8.3 MB      | 0.005      |         0.059 | 6.4 MB              | 1.3 MB              |
#
# Why is NANO-IPC faster on initial read but slower for partial materialization?
#
# 1) Initial read
#    - NANO-IPC: Parses a tiny JSON header and memory-maps the raw column buffers.
#      No per-value parsing or Python object allocation happens up front.
#      With demand paging, only the touched pages are brought into memory.
#    - JSON/JSONL: The whole file is parsed immediately into Python dict/list/str/number
#      objects. Tokenization, UTF-8 decoding, and many allocations all happen at once.
#
# 2) Partial access (extracting one field)
#    - JSON/JSONL: Data is already fully materialized as Python objects in memory,
#      so selecting a single field is basically an O(n) list comprehension with
#      near-zero extra parsing/allocations.
#    - NANO-IPC: Columns are still byte buffers until you ask for them.
#      Materializing a column means converting bytes into Python values on demand
#      (e.g., for strings: use offsets → slice bytes → UTF-8 decode → create str;
#      for numbers: struct.unpack repeatedly → build a Python list).
#      That deferred conversion cost makes partial materialization look slower.
#
# 3) Takeaways
#    - NANO-IPC follows “defer parsing, pay as you go.” It shines when you scan a few
#      columns repeatedly, do streaming/analytics, or want zero-copy interop.
#    - JSON/JSONL follows “parse up front.” After paying the upfront cost, per-field
#      access is very cheap.
#    - The gap depends on column types. Numeric columns tend to be cheaper to
#      materialize; strings and nested lists/structs cost more due to decoding and assembly.
#      (Selected field in this run: 'name')
