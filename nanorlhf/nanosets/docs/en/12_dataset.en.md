# Dataset

In this document, we explain what `Dataset` is and what functionality it provides on top of `Table`. `Dataset` holds a single `Table` internally, and layers common operations (selection, shuffling, transformation, filtering, saving/loading) on top of it.

## 1. Structure and role

`Dataset` is structured as a wrapper around `Table`, like this.

```python
class Dataset:
    def __init__(self, table: Table):
        self.table = table
```

So it is helpful to understand `Dataset` behavior as belonging to two broad categories.

### 1) A category that directly leverages Table operations
It operates while keeping the batch/columnar structure, without separately constructing rows.

- `__len__`
- `select`
- `shuffle`
- `select_columns`
- `remove_columns`
- the saving part of `save_to_disk` itself (buffer-based serialization)

### 2) A category that converts to row(dict) and then rebuilds
For reasons like applying Python functions or producing JSON output, it converts to a row-based representation in the middle.

- `__getitem__`
- `to_dict`
- `to_json`
- `map`
- `filter`
- JSON-family loading (`from_json/from_jsonl`) is row-based at the input, so it ends up having a similar flow

## 2. Length and representation

### Length
The length of a Dataset is the total number of rows in its Table.

```python
def __len__(self) -> int:
    return self.table.length
```

### Representation
For debugging, it shows the number of rows and the schema in a simple form.

```python
def __repr__(self) -> str:
    return f"Dataset(num_rows={len(self)}, schema={self.table.schema})"
```

## 3. Indexing and slicing

### Integer indexing: dataset[i]
Integer indexing selects one row and returns it as a row dict.

```python
return self.select([item]).to_dict()[0]
```

- Negative indices are adjusted to follow Python rules
- One row is selected via `select([i])`
- `to_dict()` converts to a row-based list
- It returns the first row

### Slicing: dataset[a:b:c]
A slice expands into an index list, selects those rows, and returns a list of row dicts.

```python
indices = list(range(*item.indices(len(self))))
return self.select(indices).to_dict()
```

## 4. Selection operations: select and shuffle

### select
`Dataset.select(indices)` calls `Table.take(indices)` internally.

```python
def select(self, indices: Sequence[int]) -> "Dataset":
    return Dataset(self.table.take(indices))
```

`Table.take` partitions global indices by batch, calls `RecordBatch.take` for each, and then stitches the resulting batches back into a `Table`. `Dataset` wraps and returns that result.

### shuffle
`shuffle` builds indices 0..n-1, shuffles them randomly, and reorders via `select`.

```python
idx = list(range(len(self)))
rng.shuffle(idx)
return self.select(idx)
```

## 5. Column selection and removal

### select_columns
It creates a Dataset that keeps only the specified columns.

```python
def select_columns(self, column_names: List[str]) -> "Dataset":
    return Dataset(self.table.select(column_names))
```

### remove_columns
It selects all columns except those to remove.

```python
all_names = self.table.column_names()
keep = [name for name in all_names if name not in drop_set]
return Dataset(self.table.select(keep))
```

These two operations proceed without converting to rows, by changing the schema and the column references.

## 6. to_dict and to_json

### to_dict
It converts the entire dataset into a list of row dicts.

```python
def to_dict(self) -> List[Optional[dict]]:
    return self.table.to_list()
```

### to_json
If `lines=True`, it saves as JSONL (one row per line). If `lines=False`, it saves as JSON (one large list).

```python
if lines:
    to_jsonl(fp, self.table)
else:
    to_json(fp, self.table)
```

## 7. Saving and loading

### save_to_disk: nano(IPC) saving
It saves the Table in IPC format.

```python
with open(path, "wb") as fp:
    write_table(fp, self.table)
```

This path produces a `.nano` file so that `read_table` can load it with an `mmap`-based path.

### load_dataset: load depending on extension
It selects a loader based on the file extension.

```python
if e == "json": return from_json(...)
if e in ("jsonl", "ndjson"): return from_jsonl(...)
if e == "nano": return read_table(...)
```

If multiple files are provided, it reads each into a Table and concatenates them with `Table.concat` if needed.

Also, `load_from_disk = load_dataset` is an alias.

## 8. map

`map` supports two modes.

### batched=False
It converts each batch into a row list, applies the function to each row, and rebuilds the results with `RecordBatch.from_list`.

```python
rows = batch.to_list()
out_rows = [function(row) for row in rows]
new_batches.append(RecordBatch.from_list(out_rows))
```

### batched=True
It accumulates rows into a buffer, and when the buffer is large enough, calls the function on the list. In this mode, the function must take a list and return a list.

```python
mapped = function(buffer)
if not isinstance(mapped, list):
    raise TypeError(...)
new_batches.append(RecordBatch.from_list(mapped))
```

Finally, it wraps the new batches with `Table.from_batches` and returns a Dataset.

## 9. filter

`filter` checks each row dict with a predicate and keeps only the rows that pass. When the buffer reaches a certain size (`batch_size`), it creates a batch via `RecordBatch.from_list`, and after processing everything it also turns any remaining buffer into a batch.

If the result is empty (everything is filtered out), it constructs empty columns to return a "0-row table" while preserving the schema.

## 10. Commonalities between map and filter
- They convert to an intermediate row(dict) representation.
- During this process, data is materialized and uses additional memory.
- Given that they accept an arbitrary user function, this is unavoidable.
- To avoid this, you would need to use lower-level operations like Arrow’s `compute`, but that tends to reduce usability significantly.
- In practice, Hugging Face `datasets` also prioritizes usability, so `map` and `filter` go through row-based transformations internally.

## 11. Summary

- `Dataset` is a wrapper around `Table`, and operations like selection/shuffling/column selection mainly use `Table.take/select`.
- Saving and loading `.nano` uses the IPC format to enable an `mmap`-based loading path.
- Functions like `map/filter/to_dict/to_json/__getitem__` involve row(dict)-based transformations, and then rebuild results back into `RecordBatch/Table`.