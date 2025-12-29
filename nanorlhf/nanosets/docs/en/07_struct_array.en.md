# StructArray

This document explains what `StructArray` is, why it is needed, and what design intentions its internal implementation has. `StructArray` is an Arrow-style struct array type that does not store each row as a Python `dict` object as-is, but instead stores **field-wise child Arrays in a columnar form**, and manages row-level missingness (null) separately via a **Validity Bitmap**.

## 1. Why do we use a columnar (column-oriented) layout?

The key to understanding `StructArray` is that the data is stored not in a **row-oriented** manner but in a **columnar (column-oriented)** manner.

### Row-oriented vs Columnar

- **Row-oriented**
  - One row (e.g., a dict) is stored as a single bundled object.
  - Example: `[{name: Kevin, age: 30}, {name: Ada, age: 31}, ...]`
  - It is natural for reading and writing by row, but memory access can be scattered when repeatedly processing only a specific field.

- **Columnar**
  - Values are collected per field and stored by “column”.
  - Example: `name = [Kevin, Ada, ...]`, `age = [30, 31, ...]`
  - Values of the same field are laid out contiguously, which is advantageous for per-field operations.

`StructArray` uses the columnar approach. That is, instead of storing the row dict itself, it stores **per-field columns** in `children`.

### Advantages of a columnar layout (= children layout)

- **Good for field-level processing**
  - For example, if you filter/compute statistics/transform only `age`, you can access only the `age` column sequentially.

- **Cache-friendly**
  - Values of the same field are placed contiguously, which can increase CPU cache hit rates.

- **Easier type-specific optimization**
  - Like `name` as `StringArray`, `age` as `PrimitiveArray`, `tags` as `ListArray`, and `meta` as a nested `StructArray`,
    each field can be stored in a representation that matches its type.

- **Consistent application of operations like selection (`take`)**
  - Selecting rows is effectively applying the same index selection to each column,
    and applying the same `take` to `children` preserves row alignment.

- **Good for sharing across languages/runtimes**
  - A “field-wise arrays + metadata” shape fits well with Arrow-family formats,
    making it easier to share/exchange the same structure across different systems.

## 2. What is StructArray?

`StructArray` represents a struct column with the following components.

- `field_names: List[str]`  
  A list of field names that the struct has. For example, fields like `name` and `age` can appear here.

- `children: List[Array]`  
  A list of child arrays corresponding to each field. `children[i]` stores the column data for the `field_names[i]` field.  
  In other words, instead of storing a dict per row, it stores each field’s column as a separate `Array`.

- `validity: Optional[Bitmap]`  
  Stores row-level missingness information at the bit level. 1 means valid, 0 means null. If there are no nulls, it can be `None`.

In summary, `StructArray` does not store rows as dicts, but stores them decomposed into per-field columns (child arrays).

### Example of the internal structure of StructArray

For example, suppose we have the following rows.

- `rows = [{name: Kevin, age: 30}, None, {name: Ada, age: None}]`

Then `StructArray` is represented as follows.

- `field_names = [name, age]`
- `children` are per-field columns.
  - `children[name]` is `[Kevin, None, Ada]` (e.g., `StringArray`)
  - `children[age]` is `[30, None, None]` (e.g., `PrimitiveArray`)
- `validity` is `[1, 0, 1]` at the row level.

| row i | row value | validity[i] | child[name][i] | child[age][i] |
|---:|---|---:|---|---|
| 0 | `{name: Kevin, age: 30}` | 1 | `Kevin` | `30` |
| 1 | `None` | 0 | `None` | `None` |
| 2 | `{name: Ada, age: None}` | 1 | `Ada` | `None` |

The important point here is that if a row is null, the entire struct is null, so `StructArray.__getitem__` returns `None`. At the same time, in the builder implementation, when a row is null, `None` is appended to every child as well, so the length consistency of the child arrays is maintained.

## 3. Length rules and children consistency

The length of `StructArray` is defined by the length of its children.

- If children are not empty, then `length = len(children[0])`.
- And all children must have the same length.

In other words, if even one `children[i]` has a different length, it is an error. This is a strong invariant to prevent row alignment from breaking.

If children are empty, then `length = 0`.

## 4. validity rules

If `validity` exists, it must satisfy `len(validity) == length`.

- If `validity[i] == 0`, then row i is null for the entire struct.
- If `validity[i] == 1`, then row i is valid, and `__getitem__` constructs a dict by combining values from children.

If `validity` is `None`, it is interpreted as all rows being valid.

## 5. Field name lookup: name → index mapping

`StructArray` stores the following mapping internally to find field names quickly.

- `name_to_index: Dict[str, int] = {name: i for i, name in enumerate(field_names)}`

Through this, `field_index(name)` returns the child index corresponding to the name.

Also, `check_field_index` is a utility that checks the valid range of a field index.

## 6. `__getitem__`: How do we read a row?

`StructArray.__getitem__` supports int indexing and slice indexing.

### Integer indexing

1) If null, return `None`  
2) Normalize the index via `normalize_index(key, self.length)`  
3) Iterate over `field_names` and `children` and collect `child[normalized_idx]` into a dict  
4) Return the dict

That is, it does not store the row ahead of time; it reads values from children and constructs the row dict on demand.

```python
if isinstance(key, int):
    if self.is_null(key):
        return None

    normalized_idx = normalize_index(key, self.length)

    row: Dict[str, Any] = {}
    for name, child in zip(self.field_names, self.children):
        row[name] = child[normalized_idx]
    return row
```

### Slice indexing

`array[start:stop:step]` is delegated to `take(range(...))`.

```python
if isinstance(key, slice):
    start, stop, step = key.indices(self.length)
    return self.take(range(start, stop, step))
```

## 7. `take(indices)`: Detailed behavior of selection

`StructArray.take` takes a sequence of logical indices and returns the selected result as a `StructArray`. The key goal is to apply the same selection to children, preserving row alignment while producing a new struct.

### When the input is empty

If there are no elements to select:

- Call `child.take([])` for each child to make empty children.
- Set `new_validity = None`.
- Return `StructArray(field_names, new_children, new_validity)`.

```python
if num_items == 0:
    new_children = [child.take([]) for child in self.children]
    new_validity = None
    return StructArray(self.field_names, new_children, new_validity)
```

### Contiguity (contiguous slice) check

After creating `normalized`, check whether it forms a contiguous range.

```python
normalized = [normalize_index(i, self.length) for i in indices]
is_contiguous_slice = all(
    normalized[k] + 1 == normalized[k + 1] for k in range(num_items - 1)
)
```

### validity construction logic

- If `self.validity is None`, then `new_validity = None`.
- If validity exists and the selection is contiguous, slice quickly via `validity.slice(start, num_items)`.
- If validity exists and the selection is non-contiguous, call `is_null` for each selected index to build bits, then construct via `Bitmap.from_list(bits)`.

```python
if self.validity is None:
    new_validity = None
else:
    if is_contiguous_slice:
        start = normalized[0]
        new_validity = self.validity.slice(start, num_items)
    else:
        bits: List[int] = []
        for src_i in normalized:
            bits.append(0 if self.is_null(src_i) else 1)
        new_validity = Bitmap.from_list(bits)
```

### children construction logic

Children always apply the same index selection via `child.take(normalized)`.

```python
new_children = [child.take(normalized) for child in self.children]
return StructArray(self.field_names, new_children, new_validity)
```

This approach naturally guarantees the core struct invariant: all children have the same length.

## 8. `to_list`: Convert to a Python list

`to_list` iterates over all rows:

- If null, append `None`.
- Otherwise, read each field value from children, construct a dict, and append it.

That is, it is an explicit expansion of the row construction logic in `__getitem__`.

## 9. `from_list`: Create StructArray from Python rows

`StructArray.from_list(rows, strict_keys=False)` creates a struct using a builder.

- Create a builder via `get_struct_array_builder_from_rows(rows)`.
- Accumulate each row via `builder.append(row)`.
- Return the final array via `builder.finish()`.

`strict_keys` controls whether to raise an error when an unexpected key appears in a row dict.

## 10. `StructArrayBuilder`: Role of the builder

`StructArrayBuilder` accumulates the following state to build a struct.

- `field_names`: list of fields
- `child_builders`: per-field builders
- `strict_keys`: whether to allow unexpected keys
- `validity`: row-level 0/1
- `length`: number of rows

### `append(row)`: Row accumulation rules

- If row is `None`
  - `validity += [0]`
  - Append `None` to every child builder
  - `length += 1`

- If row is a dict
  - If strict_keys is enabled, validate unexpected keys
  - `validity += [1]`
  - For each field, read `value = row.get(name, None)` and append to the corresponding child builder
  - `length += 1`

Because of this design, even when a struct row is null, the children’s length consistency is preserved.

### `finish()`: Create the final StructArray

- `children = [b.finish() for b in child_builders]`
- `validity_bitmap = Bitmap.from_list(validity)` (but if length is 0, then `None`)
- Return `StructArray(field_names, children, validity_bitmap)`

## 11. `get_struct_array_builder_from_rows`: How do we infer the struct schema?

`get_struct_array_builder_from_rows(rows)` does the following based on the rows.

1) Scan all row dicts and collect encountered keys into `inner_names`.  
   The collection order follows the first-seen order while iterating through rows.

2) Create `inner_columns: Dict[str, List[Optional[Any]]]` and build column data per field.  
   If a row does not have a key, fill it with `None`.

3) For each field column, call `inference_builder_for_column(values)` to create a child builder.

4) Return `StructArrayBuilder(inner_names, inner_child_builders, strict_keys=False)`.

In other words, the struct builds its field schema from the set of keys in rows, and infers each field type at the column level.

## 12. `inference_builder_for_column`: How do we decide the field type?

`inference_builder_for_column(values)` decides a builder by looking at the column values list for a single field.

- Find one sample value (the first non-null).
- If the sample is `None`, return `StringArrayBuilder()`.  
  That is, an all-null field is handled as a string builder by default.

- If the sample is a dict
  - Validate that all values are dict or None
  - Create a nested struct builder via `get_struct_array_builder_from_rows(values)`

- If the sample is a list/tuple
  - Validate that all values are list/tuple or None
  - Build the list’s child builder via `infer_child_builder(values)`
  - Return `ListArrayBuilder(child_builder)`

- If the sample is a str
  - Validate that all values are str or None
  - Return `StringArrayBuilder()`

- If the sample is a primitive (bool/int/float)
  - Validate that all values are primitive or None
  - Determine dtype via `infer_primitive_dtype(values)` and return `PrimitiveArrayBuilder(dtype)`

- If the sample is a tensor
  - Validate that all values are tensor or None
  - Return `TensorArrayBuilder()`

- Otherwise, raise an error because it is unsupported.

This logic strongly enforces the assumption that each struct field must be stored as a single array representation (field-level type consistency).

## 13. StructArray: 3-line summary

- `StructArray` stores `field_names` and per-field `children` in a columnar layout, and separates row nulls via `Bitmap`.
- `__getitem__` constructs a dict for row i by collecting each child’s i-th value, and returns `None` if the row is null.
- `take` applies the same selection to children, and constructs validity by slicing for contiguous selections or building a new bitmap for non-contiguous selections.