# ListArray

This document explains what `ListArray` is, why it is needed, and what design intent its internal implementation carries. `ListArray` is an Arrow-style list array type that does not hold lists as nested Python object lists; instead, it stores **child(Array)** as a single contiguous array, uses an **offsets(int32) buffer** to express list boundaries, and manages missing values (null) separately using a **Validity Bitmap**.

## 1. What is ListArray?

`ListArray` represents a list column with the following components.

- `child: Array`  
  Stores all elements (the elements inside the lists) in a single contiguous 1D array. For example, for `[[1,2], [], [3]]`, the child becomes something like `[1,2,3]` by concatenation.

- `offsets: Buffer`  
  An int32 offsets array indicating where each list starts and ends within `child`. Its length is always `base_length + 1`, and the `i`-th list occupies the range `child[offsets[i] : offsets[i+1]]`.

- `validity: Optional[Bitmap]`  
  Stores missingness information at the bit level. 1 means valid, 0 means null. If there are no nulls at all, it can be `None`.

- `indices: Optional[Buffer]`  
  As with `PrimitiveArray`/`StringArray`, this is an int32 index mapping buffer that enables arbitrary selection or view semantics without copying offsets/child. If indices exist, the array operates as a non-contiguous view, and logical indices are mapped to actual list positions (base index).

In short, `ListArray` is an Arrow-style structure that gathers all list elements into child and stores list boundaries via offsets.

### Example of ListArray internal structure

For example, suppose the following data exists.

- `data = [[1, 2, 3], [], [4, 5], None, [6]]`

Then `ListArray` is represented as follows.

- `child` is an array formed by concatenating the elements of non-null lists in order.
- `offsets` expresses the start/end (child index range) of each list.
- `validity` expresses whether each row is null.

| i (list idx) | value | validity[i] | start = offsets[i] | end = offsets[i+1] | child[start:end] |
|---:|---|---:|---:|---:|---|
| 0 | `[1, 2, 3]` | 1 | 0 | 3 | `[1, 2, 3]` |
| 1 | `[]` | 1 | 3 | 3 | `[]` |
| 2 | `[4, 5]` | 1 | 3 | 5 | `[4, 5]` |
| 3 | `None` | 0 | 5 | 5 | `[]` (does not increase elements) |
| 4 | `[6]` | 1 | 5 | 6 | `[6]` |

```text
offsets = [0, 3, 3, 5, 5, 6]
validity = [1, 1, 1, 0, 1]
child = [1, 2, 3, 4, 5, 6]
```
Above, i=3 is null, so `validity[3]=0`, and offsets keep the previous value (5). In other words, `[offsets[3], offsets[4])` is an empty range, but since validity marks it as null, `__getitem__` returns None.

## 2. Why is the offsets + child structure needed?

If you store nested lists directly in Python, the outer list holds pointers to inner list objects, and the inner lists also hold pointers to element objects. In contrast, Arrow-style storage works like this.

- `child` is a contiguous storage created by concatenating all elements
- `offsets[i]`, `offsets[i+1]` specify the start/end of the i-th list

The advantages of this structure are as follows.

- Elements inside lists live in contiguous storage (child), making memory access more predictable.
- List boundaries are clearly defined by offsets, making cross-language/platform sharing easier.
- It becomes easier to implement operations like slice/take without copying child/offsets (or with minimal copying).

## 3. Offsets rules and base_length

Since `offsets` is an int32 buffer, its byte length must be a multiple of 4.

```python
if len(offsets) % 4 != 0:
    raise ValueError("offsets buffer size must be a multiple of 4 (int32)")
```

`base_length` is the number of offsets entries minus 1.

```python
base_length = len(offsets) // 4 - 1
```

Because offsets must contain at least 1 entry, it is an error if `base_length < 0`.

```python
if base_length < 0:
    raise ValueError("offsets buffer must contain at least one entry")
```

Here, `base_length` is the actual number of lists that offsets can represent (= the length of the base array).

Also, `ListArray` validates that the last offsets value does not exceed the child length.

```python
total_elems = unpack_int32(offsets, base_length)
if total_elems > len(child):
    raise ValueError(f"offsets refer to {total_elems} child elements, but child length is {len(child)}")
```
That is, offsets must form valid list boundaries within the range of child.

## 4. Logical length and the meaning of indices

`ListArray` is contiguous when there is no `indices`, and it is a non-contiguous view when `indices` exist.

### In the contiguous case

- `logical_length = length`
- And this `length` must be equal to `base_length`.

```python
if indices is None:
    logical_length = length
    if logical_length != base_length:
        raise ValueError(f"length mismatch: base_length={base_length}, length argument={length}")
```
That is, in a contiguous array, the number of lists represented by offsets must match the array length exactly.

### In the non-contiguous view case

If indices exist, the logical length is determined by the number of indices entries.

```python
if len(indices) % 4 != 0:
    raise ValueError("indices buffer size must be a multiple of 4 (int32)")
logical_length = len(indices) // 4
```
At this time, offsets and child share the base array’s storage, and indices map logical indices to base indices.

## 5. Index management: normalized index and base index

`ListArray` uses the same concepts as `PrimitiveArray`/`StringArray`.

### normalized index

Since Python allows negative indexing, the implementation first normalizes indices into the range `[0, length)` internally.

- `-1` is converted to the last element.
- It uses `normalize_index(i, self.length)`.

### base index

The base index is the value that maps a logical index to the position where the actual list exists (in the base_length coordinate system).

- For a contiguous array, base index equals the normalized index.
- For a non-contiguous view, base index is read from `indices[normalized]`.

Using this base index, list boundaries are found from offsets.

## 6. `__getitem__`: How are lists read?

Integer indexing follows the process below.

1) If null, return None  
2) Compute the base index  
3) Extract start/end from offsets  
4) Take the [start:end) range from child  
5) Convert sub_array to a Python list and return it

### null handling and base index range check

```python
if self.is_null(key):
    return None

idx = self.base_index(key)
if not (0 <= idx < self.base_length):
    raise IndexError(f"base index {idx} out of range [0, {self.base_length})")
```

### Computing the child range via offsets

```python
start = unpack_int32(self.offsets, idx)
end = unpack_int32(self.offsets, idx + 1)
```

If offsets are invalid (out of child bounds, or end < start), it raises an error.

```python
if start < 0 or end < start or end > len(self.child):
    raise ValueError(f"Invalid child range: start={start}, end={end}, child_length={len(self.child)}")
```

### Empty list handling

If the list length is 0, it immediately returns `[]`.

```python
if start == end:
    return []
```

### Slicing child and converting to a list

```python
sub_array = self.child.take(range(start, end))
return sub_array.to_list()
```

### Slice indexing

`array[start:stop:step]` is delegated to `take(range(...))`.

```python
start, stop, step = key.indices(self.length)
return self.take(range(start, stop, step))
```

## 7. Why there is no `__setitem__`: aiming for immutability

`ListArray` also has no `__setitem__`. Arrow-style arrays are typically treated as immutable, and this implementation aims in the same direction.

If you aim for immutability, you get the following benefits.

- Safe sharing: even if multiple views share the same offsets/child/validity buffers, the risk of in-place modifications breaking the meaning of other views is reduced.
- Simplified zero-copy optimization: when expressing operations like `take` or slicing as views, safety and design become simpler.
- Favorable for parallel processing: when multiple operations are only reading concurrently, it reduces data contention.

Therefore, changes are encouraged not via in-place mutation but by creating new arrays through builders or creating new views via `take`.

## 8. take(indices): Detailed behavior of the selection operation

`take` takes a sequence of logical indices and returns the selection result as a `ListArray`. The key goal is to represent it as a view whenever possible, without copying child/offsets.

### When the input is empty

If there are no elements at all, it constructs an empty array with empty `offsets([0])`.

```python
empty_offsets = pack_int32([0])
return ListArray(
    offsets=empty_offsets,
    length=0,
    child=self.child,
    validity=None,
    indices=None,
)
```
In this case, child continues to reference the existing child.

### Normalizing input indices and checking contiguity

```python
normalized = [normalize_index(i, self.length) for i in indices]
is_contiguous_slice = all(normalized[k] + 1 == normalized[k + 1] for k in range(num_items - 1))
```
If contiguous, it attempts the contiguous-slice optimization.

## 9. Contiguous selection in take: contiguous base vs non-contiguous base

### Contiguous selection from a contiguous base

If the base is contiguous and you select a contiguous range, the range is also contiguous in the base_length coordinate system, so the child range can also be captured as a single contiguous range. In this case, `ListArray` does the following.

- It creates `new_child` by slicing the child range (via child.take).
- Offsets must be rebased to a local (0-based) coordinate system, so it creates new offsets.
- It slices validity to match the selected logical range.

#### 1) Determine the base range

```python
base_start = start
base_end = start + length
```

#### 2) Compute the child range and create new_child

```python
child_start = unpack_int32(self.offsets, base_start)
child_end = unpack_int32(self.offsets, base_end)
new_child = self.child.take(range(child_start, child_end))
```

#### 3) Create new_offsets: rebase to local coordinates (starting from 0)

Since the existing offsets are absolute offsets relative to the base child, it recalculates them by subtracting `child_start` to make the start of new_child align with 0.

```python
local_offsets: List[int] = []
for i in range(base_start, base_end + 1):
    off = unpack_int32(self.offsets, i)
    local_offsets.append(off - child_start)

new_offsets = pack_int32(local_offsets)
```

#### 4) Slice validity

```python
new_validity = self.validity.slice(start, length) if self.validity else None
```

#### 5) Return the result

```python
return ListArray(
    offsets=new_offsets,
    child=new_child,
    length=length,
    validity=new_validity,
    indices=None,
)
```
In this case, the result becomes a contiguous `ListArray` again.

### Contiguous selection from a non-contiguous base

If the base is already an indices-based view, even if the selection is logically contiguous, the actual base indices may not be contiguous. In this case, it does not rebuild offsets/child; it only slices indices to create a smaller view.

```python
index_offset = start * 4
index_length = length * 4

sub_indices = self.indices.slice(index_offset, index_length)
new_validity = self.validity.slice(start, length) if self.validity else None
return ListArray(
    offsets=self.offsets,
    length=length,
    child=self.child,
    validity=new_validity,
    indices=sub_indices,
)
```
In this case, the result is a non-contiguous view sharing offsets/child.

## 10. Non-contiguous selection in take: building a view with new indices

If the selection is not contiguous, it is difficult to express it as a slice of offsets/child. In that case, it constructs a view by creating a new indices buffer.

- If the source is contiguous, base_indices are just normalized.
- If the source is non-contiguous, it maps normalized again through indices to obtain base indices.

```python
base_indices = normalized if self.is_contiguous() else [unpack_int32(self.indices, i) for i in normalized]
new_indices = pack_int32(base_indices)
return ListArray(
    offsets=self.offsets,
    length=len(base_indices),
    child=self.child,
    validity=self.validity,
    indices=new_indices,
)
```
In this branch, validity is also shared as-is. That is, the selection result shares offsets/child/validity and only creates a new indices buffer to form the view.

## 11. `to_list`: Converting to a Python list

`to_list` iterates over logical indices; if null it appends None, otherwise it constructs the list via `self[i]` and appends it.

```python
outputs = []
for i in range(self.length):
    if self.is_null(i):
        outputs.append(None)
    else:
        outputs.append(self[i])
return outputs
```

## 12. `from_list` and `ListArrayBuilder`

`from_list` constructs offsets/child/validity at once via a builder.

```python
child_builder = infer_child_builder(data)
builder = ListArrayBuilder(child_builder)
for row in data:
    builder.append(row)

return builder.finish()
```
The key point here is that it infers the child dtype/builder from the input data.

### Internal state of `ListArrayBuilder`

- `child_builder: ArrayBuilder` is the builder that accumulates child elements.
- `offsets: List[int]` always starts with 0.
- `validity: List[int]` accumulates 0/1.
- `length: int` is the number of appended rows.

```python
self.child_builder = child_builder
self.offsets: List[int] = [0]
self.validity: List[int] = []
self.length: int = 0
```

### `append`: Handling list/None

- If None: validity=0, offsets keep the previous value (no new elements), length += 1
- If list (Iterable): validity=1, append each elem into child_builder, reflect cumulative child element count into offsets, length += 1

Also, strings/bytes are iterable but treating them as list elements is usually unintended, so they are explicitly disallowed.

```python
if value is None:
    self.validity.append(0)
    self.offsets.append(self.offsets[-1])
    self.length += 1
    return self

if isinstance(value, (str, bytes, bytearray)) or not hasattr(value, "__iter__"):
    raise TypeError(
        f"ListArrayBuilder.append expects an iterable (non-string) or None, got {type(value).__name__}"
    )

self.validity.append(1)
start_count = self.offsets[-1]
count = 0
for elem in value:
    self.child_builder.append(elem)
    count += 1

self.offsets.append(start_count + count)
self.length += 1
return self
```

### finish: Building buffers, child array, bitmap, and then creating ListArray

finish validates the validity length and offsets length.

```python
num_items = self.length
if len(self.validity) != num_items:
    raise ValueError(f"validity length {len(self.validity)} does not match number of items {num_items}")
if len(self.offsets) != num_items + 1:
    raise ValueError(
        f"offsets length must be num_items + 1, got offsets={len(self.offsets)}, num_items={num_items}"
    )
```

Then it packs offsets into an int32 buffer, creates the child array via child_builder.finish(), and creates the validity bitmap.

```python
offsets_buffer = pack_int32(self.offsets)
child_array = self.child_builder.finish()
validity_bitmap = Bitmap.from_list(self.validity)
```

Finally, it returns a contiguous ListArray.

```python
return ListArray(
    offsets=offsets_buffer,
    length=num_items,
    child=child_array,
    validity=validity_bitmap,
    indices=None,
)
```

## 13. infer_child_builder: How is child_builder inferred?

`ListArray.from_list` uses `infer_child_builder(rows)` to automatically choose a child builder suitable for the element types in the input data. This function has the following goals.

- Find a representative sample element from the input rows and decide the type.
- If list elements are nested lists, recursively infer the inner element type to construct a **nested ListArrayBuilder**.
- Support multiple types such as dict, str, primitive, tensor, and explicitly error out if types are mixed.

Below is the actual code.

```python
def infer_child_builder(rows: List[Optional[Iterable[Any]]]) -> ArrayBuilder:
    from nanorlhf.nanosets.dtype.primitive_array import PrimitiveArrayBuilder
    from nanorlhf.nanosets.dtype.string_array import StringArrayBuilder
    from nanorlhf.nanosets.dtype.list_array import ListArrayBuilder
    from nanorlhf.nanosets.dtype.tensor_array import TensorArrayBuilder

    sample: Any = None
    for row in rows:
        if row is None:
            continue
        for element in row:
            if element is not None:
                sample = element
                break
        if sample is not None:
            break

    if sample is None:
        raise ValueError("Cannot infer element type: all rows are None or empty.")

    if isinstance(sample, (list, tuple)):
        inner_rows: List[Optional[Iterable[Any]]] = []
        for row in rows:
            if row is None:
                continue
            for sub in row:
                if sub is None:
                    inner_rows.append(None)
                elif isinstance(sub, (list, tuple)):
                    inner_rows.append(sub)
                else:
                    raise TypeError(f"Expected nested list elements, found {type(sub).__name__}")
        inner_child_builder = infer_child_builder(inner_rows)
        return ListArrayBuilder(inner_child_builder)

    if isinstance(sample, dict):
        dict_elements: List[Optional[Dict[str, Any]]] = []
        for row in rows:
            if row is None:
                continue
            for element in row:
                if element is None:
                    dict_elements.append(None)
                elif isinstance(element, dict):
                    dict_elements.append(element)
                else:
                    raise TypeError(f"Mixed element types: expected dict, got {type(element).__name__}")

        return get_struct_array_builder_from_rows(dict_elements)

    if isinstance(sample, str):
        for row in rows:
            if row is None:
                continue
            for element in row:
                if element is None:
                    continue
                if not isinstance(element, str):
                    raise TypeError(f"Mixed element types: expected str, got {type(element).__name__}")
        return StringArrayBuilder()

    if isinstance(sample, (bool, int, float)):
        prims: List[Optional[PrimitiveType]] = []
        for row in rows:
            if row is None:
                continue
            for element in row:
                if element is None:
                    prims.append(None)
                    continue
                if isinstance(element, (bool, int, float)):
                    prims.append(element)
                else:
                    raise TypeError(f"Mixed element types: expected primitive, got {type(element).__name__}")

        data_type = infer_primitive_dtype(prims)
        return PrimitiveArrayBuilder(data_type)

    if torch.is_tensor(sample):
        for row in rows:
            if row is None:
                continue
            for element in row:
                if element is None:
                    continue
                if not torch.is_tensor(element):
                    raise TypeError(f"Mixed element types: expected tensor-like, got {type(element).__name__}")
        return TensorArrayBuilder()

    raise TypeError(f"Unsupported element type for list: {type(sample).__name__}")
```

### Summary of infer_child_builder behavior

- Sample selection step
  - It scans rows from the beginning and selects the first element that is not None within a row that is not None as `sample`.
  - If everything is None or empty, it cannot infer the type, so it raises an error.

- Branch handling (notably in the following order)
  - If `sample` is `(list, tuple)`, it treats it as a nested list, constructs `inner_rows` from the inner lists, recursively calls itself to create an inner child builder, and then returns `ListArrayBuilder(inner_child_builder)`.
  - If `sample` is `dict`, it collects dicts and builds a struct builder via `get_struct_array_builder_from_rows(...)`.
  - If `sample` is `str`, it validates that all elements are `str` or `None`, then returns `StringArrayBuilder()`.
  - If `sample` is `(bool, int, float)`, it validates that all elements are primitive or `None`, chooses dtype via `infer_primitive_dtype(...)`, then returns `PrimitiveArrayBuilder(dtype)`.
  - If `sample` is a tensor, it validates that all elements are tensor or `None`, then returns `TensorArrayBuilder()`.
  - Otherwise, it does not support the type and raises an error.

### Why does it strictly forbid “mixed types”?

Since `ListArray` must internally represent child as a single `Array`, the child dtype (or structure) must be determined consistently. For example, if one row has strings and another row has numbers, it becomes ambiguous what dtype child should have, and subsequent operations also become unstable. Therefore, `infer_child_builder` decides a type based on a sample and then scans all rows, immediately raising an exception if the type rules are violated.

## 14. ListArray: 3-line summary

- `ListArray` stores all list elements contiguously in a child array, expresses list boundaries via `offsets(int32)`, and separates nulls using a `Bitmap`.
- `__getitem__` computes [start, end) from offsets using the base index, then takes that range from child and returns it as a Python list.
- `from_list` infers a child builder via `infer_child_builder`, and `ListArrayBuilder` accumulates offsets/validity and then creates the final `ListArray` in finish.