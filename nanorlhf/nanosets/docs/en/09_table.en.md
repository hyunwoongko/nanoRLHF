# Table / RecordBatch / Schema / Field
This document explains what roles nanosets `Table`/`RecordBatch`/`Schema`/`Field` play, and why this structure is natural in an Arrow-like (columnar, buffer-based) system.

## 1. Big picture: why split `Table` into multiple pieces instead of one big block?
nanosets’ data model is roughly composed like this.

- `Field`: a "definition" of a single column (name/type/nullable)
- `Schema`: a "table structure" made by collecting multiple `Field`s
- `RecordBatch`: a "bundle of columns that share the same schema" + "row count (length)" (holds actual data)
- `Table`: a "logical table" made by concatenating multiple `RecordBatch`es

The core of this structure is separating data (column buffers) from metadata (schema), and managing a large table by splitting it into batch units.

### Why do we need batches (`RecordBatch`)?
When dealing with large-scale data, it’s hard to build and process "the entire table" in one shot. Having a batch unit makes the following easier.

- partial processing: slice/take/select only a subrange
- concat: merge multiple tables by just concatenating their batches
- memory/cache friendliness: avoid an overly large single block and manage data in appropriately sized chunks

In the end, you can think of `Table` as the "logical whole" and `RecordBatch` as the "physical unit".

## 2. `Schema` and `Field`: defining the data
### What is `Field`?
`Field` is the definition of a single column.

- `name`: column name
- `dtype`: column type (`DataType`)
- `nullable`: whether null is allowed

Important point: `Field` does not hold actual data. It is purely metadata saying "this column follows these name/type/nullable rules".  
Also, because it is `@dataclass(frozen=True)`, once created it cannot be changed.

### What is `Schema`?
`Schema` is a tuple collecting `Field`s. In other words, it defines the structure: "the table consists of these columns".

`Schema`’s responsibilities are simple but very important.

- guarantees column order: the order of `fields` is the column order
- name → index mapping: `schema.index("name")`
- the basis for consistency checks across tables/batches

## 3. `RecordBatch`: a bundle of "same-length columns"
`RecordBatch` holds a `schema` and `columns` (a list of Arrays), and those columns together form a batch with the same length.

### Methods of `RecordBatch`

The operations below don’t "manipulate row dicts directly". Instead, they apply the same operation to each column `Array` to build a new `RecordBatch`. The key is that `RecordBatch` is "a bundle of columns + same length".

#### 1) `column(i_or_name)`
```python
def column(self, i_or_name: Union[int, str]) -> Array:
    if isinstance(i_or_name, int):
        return self.columns[i_or_name]
    if isinstance(i_or_name, str):
        idx = self.schema.index(i_or_name)
        return self.columns[idx]
    raise TypeError("Argument must be an integer index or a string column name.")
```

Fetch a specific column `Array` by index (`int`) or by name (`str`). If a name is given, it finds the index via `Schema.index(name)` and returns that column.

#### 2) `slice(offset, length)`

```python
def slice(self, offset: int, length: int) -> "RecordBatch":
    if length == 0:
        new_cols = [col.take([]) for col in self.columns]
        return RecordBatch(self.schema, new_cols)

    row_range = range(offset, offset + length)
    new_cols = [col.take(row_range) for col in self.columns]
    return RecordBatch(self.schema, new_cols)
```

Creates a new `RecordBatch` by taking a contiguous range of rows. From an implementation perspective, the key is:

- build a "row index range" as `range(offset, offset + length)`
- call `col.take(row_range)` for every column to extract the same row range

If the length is 0, it applies `take([])` to every column to create an empty result.

#### 3) `take(indices)`

```python
def take(self, indices: Sequence[int]) -> "RecordBatch":
    new_cols = [col.take(indices) for col in self.columns]
    return RecordBatch(self.schema, new_cols)
```

Given an arbitrary list/sequence of row indices, it selects only those rows and returns a new `RecordBatch`.  
The implementation follows the same pattern as `slice`: apply `col.take(indices)` to each column.

#### 4) `select(names)`
```python
def select(self, names: List[str]) -> "RecordBatch":
    field_indices = [self.schema.index(name) for name in names]

    new_fields = tuple(self.schema.fields[i] for i in field_indices)
    new_schema = Schema(new_fields)

    new_columns = [self.columns[i] for i in field_indices]
    return RecordBatch(new_schema, new_columns)
```

Selects only some columns and returns a new `RecordBatch`. The key is "filter schema and columns in the same way".

- compute field indices from the selected names
- rebuild `Schema(fields)` using those indices
- pick the corresponding `columns` with the same indices to build the new batch

#### 5) `to_list()`

```python
def to_list(self) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    per_column_lists = [col.to_list() for col in self.columns]

    for row_index in range(self.length):
        row: Dict[str, Any] = {}
        for field, column_values in zip(self.schema.fields, per_column_lists):
            row[field.name] = column_values[row_index]
        rows.append(row)

    return rows
```

Converts a `RecordBatch` into a "rows-only" form: `List[Dict[str, Any]]`. The main flow is:

1) convert each column with `col.to_list()` to prepare per-column Python lists  
2) iterate `row_index` and build a row dict by filling each field name with the corresponding column value

#### 6) `from_list(rows)`

```python
@classmethod
def from_list(cls, rows: List[Optional[Dict[str, Any]]], strict_keys: bool = False) -> "RecordBatch":
    struct = StructArray.from_list(rows, strict_keys=strict_keys)

    fields = tuple(
        Field(
            name=name,
            dtype=child.dtype,
            nullable=(child.validity is not None),
        )
        for name, child in zip(struct.field_names, struct.children)
    )
    schema = Schema(fields)
    return cls(schema, struct.children)
```

Takes row-based input (`List[Optional[Dict[str, Any]]]`) and reconstructs a columnar `RecordBatch`. The key is: "first build a `StructArray` from rows, and its children become the columns".

- convert row dicts into a struct column with `StructArray.from_list(rows, ...)`
- build `Field`s by reading each child column’s `dtype` and `validity`
- create `RecordBatch(schema, columns)` using `Schema(fields)` + `struct.children`

In summary, most `RecordBatch` operations look like row operations, but in reality they apply the same indexing operation to column `Array`s to produce a new batch.

## 4. `Table`: a logical table formed by concatenating `RecordBatch`es
`Table` forms a logical table by collecting `batches: List[RecordBatch]` that all share the same `schema`.

### Perspective provided by `Table`
- logical length: `self.length = sum(b.length for b in batches)`
- batch iteration: `iter_batches()`
- column access: `column(i_or_name)` returns a "per-batch list of columns"
  - meaning a Table column is not a single Array, but a list of Arrays (one per batch)

### Simple methods

`Table` operations basically work "over multiple `RecordBatch`es". If `RecordBatch` is the physical unit, `Table` is a wrapper that provides global indexing (based on the whole table).

#### 1) `__getitem__`
```python
def __getitem__(self, item):
    if isinstance(item, int):
        return self.slice(item, 1).to_list()[0]
    elif isinstance(item, slice):
        indices = list(range(*item.indices(len(self))))
        return self.take(indices)
    else:
        raise TypeError("Invalid argument type.")
```

- `table[i]` returns a row dict by creating a 1-row slice at global index `i` and calling `to_list()[0]`.
- `table[a:b:c]` expands the Python slice into an index list and internally calls `take(indices)`.

#### 2) `column(i_or_name)`

```python
def column(self, i_or_name) -> List[Array]:
    cols: List[Array] = []
    for b in self.batches:
        cols.append(b.column(i_or_name))
    return cols
```

A Table column is not a single `Array`, but one per batch. Therefore `column(...)` returns `List[Array]`.

#### 3) `select(names)`
`select` applies `RecordBatch.select(names)` to every batch and then wraps the results back into a `Table`.

```python
def select(self, names: List[str]) -> "Table":
    new_batches = [b.select(names) for b in self.batches]
    return Table.from_batches(new_batches)
```

#### 4) `concat(tables)`
`concat` does not "rebuild rows or rearrange data"; it creates a new `Table` by only concatenating batch lists. (All schemas must match.)

```python
@classmethod
def concat(cls, tables: List["Table"]) -> "Table":
    batches: List[RecordBatch] = []
    for table in tables:
        batches.extend(table.batches)
    return cls.from_batches(batches)
```

## 5. `table.slice(offset, length)`

The goal of `Table.slice` is to make the whole table look like "one contiguous row array" while still preserving the internal structure split across multiple `RecordBatch`es, and to create a new `Table` by cutting only the necessary batch segments.

The core idea is simple.

- the user’s `offset`, `length` are global (whole table) coordinates
- each `RecordBatch` understands only local indices inside itself
- so we compute a global → local index conversion, slice only what we need from each batch, and assemble a new table


```python
remaining = length
batch_start_global = 0
new_batches: List[RecordBatch] = []

for batch in self.batches:
    batch_length = batch.length
    batch_end_global = batch_start_global + batch_length

    if batch_end_global <= offset:
        batch_start_global = batch_end_global
        continue

    local_start = max(0, offset - batch_start_global)
    local_available = batch_length - local_start
    local_len = min(remaining, local_available)

    new_batches.append(batch.slice(local_start, local_len))
    remaining -= local_len

    if remaining <= 0:
        break

    batch_start_global = batch_end_global

return Table.from_batches(new_batches)
```

### 1) `batch_start_global` and `batch_end_global`: global indices
When there are multiple batches, each batch occupies a global range like this.

- batch 0: `[0, len0)`
- batch 1: `[len0, len0+len1)`
- batch 2: `[len0+len1, len0+len1+len2)`
- ...

The code tracks this with `batch_start_global` (start) and `batch_end_global` (end).

- `batch_start_global`: the global row index where the current batch starts in the table
- `batch_end_global = batch_start_global + batch_length`: the global end of the current batch (exclusive)

### 2) `remaining`: how many rows we still need to take

For example, with `slice(offset=5, length=10)` we must take 10 rows total.
- if we took 3 rows from batch0, `remaining` becomes 7
- if we took 5 rows from batch1, `remaining` becomes 2
- initially, nothing is taken yet, so `remaining=10=length`
- whenever we take some rows from a batch, we reduce `remaining` accordingly

### 3) `if batch_end_global <= offset`: this batch is entirely before the slice start
This condition checks: "is the current batch completely before the slice start (offset)?"

- current batch global range is `[batch_start_global, batch_end_global)`
- the slice starts at `offset`
- if `batch_end_global <= offset`, the batch ends at or before offset, so it contributes nothing to the slice

So the code skips this batch and moves to the next one.

```python
if batch_end_global <= offset:
    batch_start_global = batch_end_global
    continue
```

The reason we update `batch_start_global` is to compute the next batch’s global range correctly.

### 4) `local_start`: where to start slicing inside this batch
Now we’re in a situation where "the slice begins to include this batch".

- slice global start is `offset`
- batch global start is `batch_start_global`

The batch-local start is:
- `offset - batch_start_global` (moving the global offset into the batch’s coordinate system)
- for example if `offset=10`, `batch_start_global=7`,
  - the slice includes `current_batch[3:]`
  - so `local_start = 10 - 7 = 3`
  - meaning the slice starts at row 3 inside this batch
- but offset could be before the batch start (e.g., the slice started in a previous batch and continues here).
  then the local start cannot be negative, so we use `max(0, ...)`.

```python
local_start = max(0, offset - batch_start_global)
```

This is the key line that "projects" a global offset into batch-local indices.

### 5) `local_len`: how many rows to take from this batch
The maximum we can take from this batch is:

- how many rows remain from `local_start` to the end of the batch
- i.e., `local_available = batch_length - local_start`

But the total slice still needs `remaining` rows.
So we take the smaller one.

```python
local_available = batch_length - local_start
local_len = min(remaining, local_available)
```

This guarantees we don’t cross batch boundaries and we take exactly as much as needed.

### 6) slice the batch and update remaining
```python
new_batches.append(batch.slice(local_start, local_len))
remaining -= local_len
```

Important point: `Table.slice` does not touch columns directly. It calls `RecordBatch.slice` on each batch to create new batches, then wraps them into a new `Table`.

So `Table` is responsible for "cutting and assembling at the batch level", and `RecordBatch.slice` performs the actual "cutting at the column level".

### 7) stop when done
```python
if remaining <= 0:
    break
```

Once we’ve taken the required number of rows, there’s no need to inspect further batches.

### 8) example

Suppose batch lengths are `[3, 5, 2]`.

- batch0 global `[0,3)`
- batch1 global `[3,8)`
- batch2 global `[8,10)`

For `slice(offset=2, length=6)`, the global range is `[2,8)`.

- batch0: overlap is `[2,3)` → local start 2, length 1
- batch1: overlap is `[3,8)` → local start 0, length 5
- batch2: starts at 8 which equals the slice end, so it contributes nothing

So the result is:
- 1 row from batch0
- 5 rows from batch1
assembled into a new Table.

### 9) advantages of this approach
- for users: can call `slice(offset, length)` as if `Table` is one big array
- implementation/performance: can preserve internal batch structure and only create/assemble the needed batch pieces
- Arrow-like columnar view: instead of "copying row by row", it naturally composes "batch/column operations"

## 6. `table.take(indices)`

The purpose of this function is to `take` from the whole `Table` under "global indices (0..len(table)-1)". But since `Table` is internally split into multiple `RecordBatch`es, it converts global indices into batch-local indices, applies `RecordBatch.take(local_indices)` per batch, and then wraps those result batches back into a new `Table`.

### 1) empty input handling
If `indices` is empty, the result should be a "table with 0 rows". The implementation applies `col.take([])` to the first batch’s columns so the schema is preserved while the columns become empty, then builds a `RecordBatch` and returns it.

```python
if not indices:
    first_batch = self.batches[0]
    empty_columns = [col.take([]) for col in first_batch.columns]
    empty_batch = RecordBatch(self.schema, empty_columns)
    return Table.from_batches([empty_batch])
```

The key point is "preserve the schema". Even with 0 rows, keeping the column definitions helps maintain consistency for later ops/IO.

### 2) normalize global indices: allow negative indices
To allow Python-style indices like `-1`, we normalize all indices into the `[0, n)` range first.

```python
n = self.length
normalized_indices = [normalize_index(idx, n) for idx in indices]
```

Here `normalize_index` usually means:
- if negative, `idx += n`
- if out of range, raise an error

### 3) `batch_starts`: a basis for mapping global indices to batches
Collect the global starting position of each batch.

For example if batch lengths are `[3, 5, 2]`:
- `batch_starts = [0, 3, 8]`

```python
batch_starts: List[int] = []
current = 0
for batch in self.batches:
    batch_starts.append(current)
    current += batch.length
```

With this, when a global index `idx` is given, we can quickly find "which batch it belongs to".

### 4) `bisect_right`: find "the batch containing idx"

```python
batch_idx = bisect_right(batch_starts, idx) - 1
```

`bisect_right(batch_starts, idx)` returns the insertion position that keeps ordering if we inserted `idx`. Subtracting 1 gives the "rightmost start <= idx", i.e., the batch containing idx.

Example: `batch_starts = [0, 3, 8]`
- `idx = 0`  -> `bisect_right(...)=1` -> `batch_idx=0`
- `idx = 2`  -> `bisect_right(...)=1` -> `batch_idx=0`
- `idx = 3`  -> `bisect_right(...)=2` -> `batch_idx=1`
- `idx = 7`  -> `bisect_right(...)=2` -> `batch_idx=1`
- `idx = 8`  -> `bisect_right(...)=3` -> `batch_idx=2`

Then convert to local index:

```python
local_idx = idx - batch_starts[batch_idx]
```

### 5) why "grouping"? (`current_batch_idx`, `flush`)
As we process indices one by one, we collect those belonging to the same batch and call `RecordBatch.take()` once per group, rather than many times.

That reduces repeated calls to `take()` on the same batch.

State variables:

- `current_batch_idx`: which batch we are currently collecting for
- `current_local_indices`: local indices to take from that batch
- `prev_local`: previous local index (for checking consecutiveness)

And `flush()` means "materialize the collected group by calling batch.take and append to new_batches".

```python
def flush() -> None:
    nonlocal current_batch_idx, current_local_indices, prev_local
    if current_batch_idx is None or not current_local_indices:
        return
    base_batch = self.batches[current_batch_idx]
    new_batches.append(base_batch.take(current_local_indices))
    current_batch_idx = None
    current_local_indices = []
    prev_local = None
```

### 6) consecutive-index optimization: same batch + (prev_local + 1) keeps accumulating
Key branch in the loop:

```python
if batch_idx == current_batch_idx and prev_local is not None and local_idx == prev_local + 1:
    current_local_indices.append(local_idx)
    prev_local = local_idx
else:
    flush()
    current_batch_idx = batch_idx
    current_local_indices = [local_idx]
    prev_local = local_idx
```

This keeps a group when indices are consecutive within the same batch. If consecutiveness breaks (different batch, or gap within the same batch), it flushes the previous group and starts a new one.

Note: this implementation follows the input indices order.
- if indices are sorted, grouping works well
- if indices are scrambled, flush happens more often and grouping mirrors that order (still correct semantically)

### 7) final flush after the loop
Because flush happens only when starting a new group inside the loop, we need one more flush at the end to include the last group.

```python
flush()
```

### 8) if result is empty, return an empty table again
As a defensive fallback, if `new_batches` is empty, it returns an empty table in the same "preserve schema" manner.

```python
if not new_batches:
    first_batch = self.batches[0]
    empty_columns = [col.take([]) for col in first_batch.columns]
    empty_batch = RecordBatch(self.schema, empty_columns)
    return Table.from_batches([empty_batch])
return Table.from_batches(new_batches)
```

### 9) summary
- convert global indices `indices` into batch-local indices
- group by batch (especially consecutive indices) and call `RecordBatch.take()` per group
- assemble the resulting `RecordBatch`es into a new `Table`

## 7. Table / RecordBatch / Schema / Field: 3-line summary
- `Field`/`Schema` define columns (name/type/nullable), and are the basis for ensuring consistency across batches/tables.
- `RecordBatch` is the unit that bundles same-length columns (Arrays) with a schema, and is the fundamental unit for slice/take/select.
- `Table` is a logical table formed by concatenating `RecordBatch`es, and this batch-based structure naturally fits large-scale processing and IPC (mmap, Zero-copy oriented) systems.