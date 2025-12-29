# JSON / JSONL IO (Rows-only)
This document explains utility functions that export `Table`/`RecordBatch` to a row-based JSON format (rows-only) and read them back. nanosets’ internal representation is an Arrow-like columnar structure (buffer-based such as `values`/`offsets`/`validity`/`indices`), but JSON/JSONL is intended to convert that into a human-readable row (dict) representation rather than preserving the buffer layout.

## 1. What is rows-only JSON?
Here, rows-only means that the JSON root is a list of rows.

- Each row is a `dict` (e.g., `{"name": "Kevin", "age": 30}`), or
- If it is a null row, it is `None` (in JSON, `null`).

Columnar schema/buffer information is not stored together in JSON. Instead, it focuses on “materializing rows into Python objects and dumping them to JSON.”
In other words, when exporting to JSON, you give up most of the columnar benefits (contiguous buffers, type stability, zero-copy) and gain readability and general portability.

## 2. Type definitions: `Row`, `TableLike`
This module handles two input types.

- `TableLike = Union[Table, RecordBatch]`
- `Row = Optional[Dict[str, Any]]`

That is, the functions are unified around reading/writing at the row level, taking either the whole `Table` or a single `RecordBatch`.

## 3. `iter_rows(obj)`: row streaming interface
`iter_rows(obj)` `yield`s rows one by one from a `Table`/`RecordBatch`.

- If `obj` is a `RecordBatch`, it iterates rows from `obj.to_list()` as-is.
- If `obj` is a `Table`, it iterates `obj.batches` in order and flattens rows by chaining each `batch.to_list()`.

## 4. `materialize(obj)`: build a list of all rows
`materialize(obj)` calls `list(iter_rows(obj))` internally, bringing all rows into memory at once.

### Why is this function needed?
Some output formats/libraries require an “entire list” shape. A representative case is `to_json`.

- Since `to_json` produces a JSON root that is a “list,” the simplest implementation is to first build the rows list and then call `json.dump(rows, ...)`.

In contrast, `to_jsonl` can write line by line, so it does not need to materialize everything.

## 5. `to_json(fp, obj, indent=2)`: write as a JSON array
`to_json` writes rows-only JSON.

- First it builds a row list with `rows = materialize(obj)`, and
- Then it calls `json.dump(rows, fp, ensure_ascii=False, indent=indent)`.

## 6. `to_jsonl(fp, obj)`: write as JSONL
`to_jsonl` writes one row per line.

- `for row in iter_rows(obj):`
  - `fp.write(json.dumps(row, ensure_ascii=False))`
  - `fp.write("\n")`

## 7. `from_json(path, batch_size=DEFAULT_BATCH_SIZE)`: read a JSON array
`from_json` opens the file, reads everything via `json.load(f)`, then calls `Table.from_list(data, batch_size=batch_size)`.

Key points:

- Since the input JSON is already a list of rows, reconstructing the `Table` also uses a row-based constructor (`from_list`).
- Because a `Table` is internally composed of `RecordBatch`es, `batch_size` determines “how many rows to group per batch.”

### Why does `batch_size` exist?
If you build one very large chunk, it can be heavy in terms of memory/cache, while splitting too small can add overhead. So an option for “how many rows per chunk” appears.

## 8. `from_jsonl(path, batch_size=DEFAULT_BATCH_SIZE)`: read JSONL
`from_jsonl` parses each line with `json.loads(line)`, collects them into `rows: List[Row]`, then calls `Table.from_list(rows, batch_size=batch_size)`.

## 9. When should you use this IO, and when should you use IPC?
### Good cases for this JSON/JSONL IO
- Debugging: when you want to inspect `Table` contents directly as a human
- General interchange: when you want to exchange data with other languages/environments as “row dicts” first
- Simple storage: when convenience matters more than performance/size

### Good cases for IPC
- When you want to preserve the columnar buffer layout
- When you want `mmap`-based reading to be close to zero-copy
- When performance/memory efficiency matters for large datasets

## 10. JSON / JSONL IO: 3-line summary
- This module converts `Table`/`RecordBatch` into row(dict)/null form and saves/loads it as JSON or JSONL.
- `to_json` builds a rows list first via `materialize` to produce a JSON array, while `to_jsonl` streams line by line via `iter_rows`.
- If you want to preserve Arrow-like columnar buffers, IPC is the right choice; JSON/JSONL is useful for debugging/interchange.