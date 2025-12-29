# StringArray

This document explains what `StringArray` is, why it is needed, and what design intentions its internal implementation has. `StringArray` is an Arrow-style string array type that does not hold strings as a Python object list, but instead stores them in a **contiguous byte buffer (Buffer)** and an **offsets (int32) buffer**, while managing missing values (null) separately with a **Validity Bitmap**.

## 1. What is StringArray?

`StringArray` represents a string column with the following components.

- `values: Buffer`  
  It stores the UTF-8 encoded bytes of all strings by concatenating them into a single contiguous byte buffer.

- `offsets: Buffer`  
  It is an int32 offsets array that indicates where each string starts and ends within `values`. Its length is always `physical_length + 1`, and the i-th string occupies the range `[offsets[i], offsets[i+1])`.

- `validity: Optional[Bitmap]`  
  It stores missingness information at the bit level. 1 means valid and 0 means null. If there are no nulls at all, it can be left as `None`.

- `indices: Optional[Buffer]`  
  As with `PrimitiveArray`, this is an int32 index-mapping buffer for creating arbitrary selections or view semantics without copying values/offsets. If this indices buffer exists, the array behaves as a non-contiguous view, and logical indices are mapped to actual string positions (physical indices).

In short, `StringArray` is an Arrow-style structure that gathers strings into one large UTF-8 byte buffer and stores each string boundary in offsets.

### Example of `StringArray` internal structure

| i (string idx) | string         | UTF-8 bytes       | byte_len | start = offsets[i] | end = offsets[i+1] | values[start:end] (hex)               |
|---------------:|----------------|-------------------|---------:|-------------------:|-------------------:|---------------------------------------|
|              0 | `Hello.`       | `b"Hello."`       |        6 |                  0 |                  6 | `48 65 6C 6C 6F 2E`                   |
|              1 | `I am Kevin.`  | `b"I am Kevin."`  |       10 |                  6 |                 16 | `49 20 61 6D 20 4B 65 76 69 6E 2E`    |
|              2 | `How are you?` | `b"How are you?"` |       12 |                 16 |                 28 | `48 6F 77 20 61 72 65 20 79 6F 75 3F` |

```python
offsets = [0, 6, 16, 28]
>>> values (utf-8) = b"Hello.I am Kevin.How are you?"
>>> values (hex)   = 48 65 6C 6C 6F 2E 49 20 61 6D 20 4B 65 76 69 6E 2E 48 6F 77 20 61 72 65 20 79 6F 75 3F
```

## 2. Why do we need the offsets + values structure?

If you store strings in a Python list, each string exists as an individual Python object, and the list stores pointers to those objects. In contrast, Arrow style stores them like this.

- `values` is a contiguous buffer formed by concatenating all strings as bytes
- `offsets[i]`, `offsets[i+1]` specify the start/end of the i-th string

This structure has the following advantages.

- String data is stored in a contiguous buffer, making it cache-friendly.
- String boundaries are clearly defined by offsets, making cross-language/platform sharing easier.
- Operations like slice/take are easier to implement without copying values (or with minimal copying).

## 3. Offsets rules and physical_length

Since `offsets` is an int32 buffer, its byte length must be a multiple of 4.

```python
if len(offsets) % 4 != 0:
    raise ValueError("offsets buffer size must be a multiple of 4 (int32)")
```

`physical_length` is the number of offsets entries minus 1.

```python
physical_length = len(offsets) // 4 - 1
```

Since offsets must contain at least one entry, it is an error if `physical_length < 0`.

```python
if physical_length < 0:
    raise ValueError("offsets buffer must contain at least one entry")
```

Here, `physical_length` is the actual number of strings representable by offsets (= the length of the base array).

## 4. Logical length and the meaning of indices

`StringArray` is a contiguous array when `indices` is absent, and a non-contiguous view when `indices` is present.

### In the contiguous case

- `logical_length = length`
- and this `length` must equal `physical_length`.

```python
if indices is None:
    logical_length = length
    if logical_length != physical_length:
        raise ValueError(f"length mismatch: base_length={physical_length}, length argument={length}")
```

In other words, for a contiguous array, the number of strings represented by offsets must match the array length exactly.

### In the non-contiguous view case

If indices exists, the logical length is determined by the number of indices entries.

```python
if len(indices) % 4 != 0:
    raise ValueError("indices buffer size must be a multiple of 4 (int32)")
logical_length = len(indices) // 4
```

In this case, offsets and values are shared from the base array as-is, and indices maps logical indices to physical indices.

## 5. Index management: normalized index and base index

`StringArray` uses the same concepts as `PrimitiveArray`.

### normalized index

Python allows negative indexing, so internally we first normalize indices into the range `[0, length)`.

- `-1` is converted to the last element
- we use `normalize_index(i, self.length)`

### base index

A base index is the value obtained by mapping a logical index to the position (physical index) where the actual string exists.

- For a contiguous array, the base index is the same as the normalized index.
- For a non-contiguous view, the base index is read from `indices[normalized]`.

Using this base index, we find string boundaries from offsets.

## 6. `__getitem__`: How do we read a string?

Integer indexing follows the steps below.

1) If null, return None  
2) Compute base index  
3) Extract start/end from offsets  
4) Slice the range [start:end) from values  
5) Decode UTF-8 and return the string

### Null handling and base index range check

```python
if self.is_null(key):
    return None

index = self.base_index(key)
if not (0 <= index < self.physical_length):
    raise IndexError(f"base index {index} out of range [0, {self.physical_length})")
```

### Compute string boundaries via offsets

```python
start = unpack_int32(self.offsets, index)
end = unpack_int32(self.offsets, index + 1)
```

If offsets are wrong and exceed the values range, or if there is an abnormal state such as end < start, it is an error.

```python
if start < 0 or end < start or end > len(self.values):
    raise ValueError(
        f"Invalid string slice range: start={start}, end={end}, values_size={len(self.values)}"
    )
```

### Empty string handling

If the string length is 0, it returns `""` immediately.

```python
length = end - start
if length == 0:
    return ""
```

### Slice values and decode UTF-8

```python
sub_buffer = self.values.slice(start, length)
return bytes(sub_buffer.data).decode("utf-8")
```

### Slice indexing

`array[start:stop:step]` is delegated to `take(range(...))`.

```python
start, stop, step = key.indices(self.length)
return self.take(range(start, stop, step))
```

## 7. Why `__setitem__` is absent: immutability

`StringArray` also does not have `__setitem__`. Arrow-style arrays are typically treated as immutable, and this implementation aims in the same direction.

Immutability has the following benefits.

- Safe sharing: even if multiple views share the same `values/offsets/validity` buffers, the chance that an in-place modification breaks the meaning of other views is reduced.
- Simplified zero-copy optimization: safety and design become simpler when representing operations like `take` or slicing as views.
- Favorable for parallel processing: it reduces data races when multiple operations read the same data concurrently.

Therefore, changes are guided not by in-place mutation but by building new arrays via a builder or producing new views via `take`.

## 8. take(indices): Detailed behavior of selection

`take` takes a sequence of logical indices and returns the selection result as a `StringArray`. The key goal is to represent the result as a view (view) without copying `values` as much as possible.

### When the input is empty

If there are no elements, it constructs an empty array using empty `offsets([0])` and empty values.

```python
empty_offsets = pack_int32([0])
empty_values = Buffer.from_bytearray(bytearray())
return StringArray(empty_offsets, 0, empty_values, validity=None, indices=None)
```

### Normalize input indices and check contiguity

```python
normalized = [normalize_index(i, self.length) for i in indices]
is_contiguous_slice = all(normalized[k] + 1 == normalized[k + 1] for k in range(num_items - 1))
```

If contiguous, it attempts the contiguous-slice optimization.

## 9. Contiguous selection in take: contiguous base vs non-contiguous base

### Contiguous selection from a contiguous base

If the base is contiguous and you select a contiguous range, the strings occupy a contiguous byte range in values, so you can slice values as a byte slice. However, a key difference from `PrimitiveArray` is that `StringArray` has offsets, so offsets must also be rebuilt.

#### 1) Decide the base range

```python
base_start = start
base_end = start + length
```

#### 2) Compute the byte range to slice from values

```python
byte_start = unpack_int32(self.offsets, base_start)
byte_end = unpack_int32(self.offsets, base_end)
byte_length = byte_end - byte_start
sub_values = self.values.slice(byte_start, byte_length)
# Slicing as values[byte_start:byte_start + byte_length]; values is memoryview-based, so no actual copy occurs
```

#### 3) Build sub_offsets: rebase to local coordinates (starting at 0)

Since existing offsets are absolute offsets relative to the base values, we recompute them by subtracting `byte_start` so that the start of sub_values becomes 0.

```python
local_offsets: List[int] = []
for i in range(base_start, base_end + 1):
    offset = unpack_int32(self.offsets, i)
    local_offsets.append(offset - byte_start)

sub_offsets = pack_int32(local_offsets)
```

This ensures offsets for sub_values always form a valid layout starting from 0.

#### 4) Slice validity

```python
sub_validity = self.validity.slice(start, length) if self.validity else None
```

#### 5) Return the result

```python
return StringArray(
    offsets=sub_offsets,
    length=length,
    values=sub_values,
    validity=sub_validity,
    indices=None,
)
```

In this case, the result becomes a contiguous `StringArray` again.

### Contiguous selection from a non-contiguous base

If the base is already an indices-based view, even a logically contiguous range might not be contiguous in physical string positions. In this case, it does not rebuild offsets/values; instead, it slices only indices to create a smaller view.

```python
index_offset = start * 4
index_length = length * 4
sub_indices = self.indices.slice(index_offset, index_length)
sub_validity = self.validity.slice(start, length) if self.validity else None
return StringArray(
    offsets=self.offsets,
    length=length,
    values=self.values,
    validity=sub_validity,
    indices=sub_indices,
)
```

In this case, the result is a non-contiguous view sharing offsets/values.

## 10. Non-contiguous selection in take: build a view with new indices

If the selection is not contiguous, it is hard to represent values/offsets as a slice. In that case, it creates a new indices buffer and represents the result as a view.

- If the base is contiguous, base_indices is just normalized itself.
- If the base is non-contiguous, it maps normalized through indices once more to get physical indices.

```python
base_indices = normalized if self.is_contiguous() else [unpack_int32(self.indices, i) for i in normalized]
new_indices = pack_int32(base_indices)
return StringArray(
    offsets=self.offsets,
    length=len(base_indices),
    values=self.values,
    validity=self.validity,
    indices=new_indices,
)
```

In this branch, validity is also shared as-is. That is, the selection result shares values/offsets/validity and builds the view by allocating only indices.

## 11. `to_list`: Convert to a Python list

`to_list` iterates over logical indices and returns None for nulls; otherwise it decodes the string via `self[i]`.

```python
output = []
for i in range(self.length):
    if self.is_null(i):
        output.append(None)
    else:
        output.append(self[i])
return output
```

## 12. `from_list` and `StringArrayBuilder`

`from_list` constructs offsets/values/validity at once via a builder.

```python
builder = StringArrayBuilder()
for value in data:
    builder.append(value)
return builder.finish()
```

### Internal state of `StringArrayBuilder`

- `offsets: List[int]` always starts with 0.
- It concatenates UTF-8 encoded strings into `data_bytes: bytearray`.
- It accumulates 0/1 into `validity: List[int]`.

```python
self.offsets: List[int] = [0]
self.data_bytes = bytearray()
self.validity: List[int] = []
```

### `append`: handling string/None

- If None: validity=0, offsets keeps the previous value (adds a 0-length string)
- If a string: encode as UTF-8, extend data_bytes, validity=1, append the cumulative length to offsets

```python
if value is None:
    self.validity.append(0)
    self.offsets.append(self.offsets[-1])
else:
    encoded = value.encode("utf-8")
    self.data_bytes.extend(encoded)
    self.validity.append(1)
    self.offsets.append(len(self.data_bytes))
```

### finish: build buffers and bitmap, then create StringArray

In finish, it verifies that offsets length is always `num_items + 1`.

```python
num_items = len(self.validity)
if len(self.offsets) != num_items + 1:
    raise ValueError(
        f"offsets length must be num_items + 1, got offsets={len(self.offsets)}, num_items={num_items}"
    )
```

Then it packs offsets into an int32 buffer, builds the values buffer, and creates the validity bitmap.

```python
offsets_buffer = pack_int32(self.offsets)
values_buffer = Buffer.from_bytearray(self.data_bytes)
validity_bitmap = Bitmap.from_list(self.validity)
```

Finally, it returns a contiguous StringArray.

```python
return StringArray(
    offsets=offsets_buffer,
    length=num_items,
    values=values_buffer,
    validity=validity_bitmap,
    indices=None,
)
```

## 13. StringArray: 3-line summary

- `StringArray` stores strings contiguously in a UTF-8 `values` buffer, uses `offsets(int32)` to represent each string boundary, and separates nulls via a `Bitmap`.
- `__getitem__` uses the base index to get [start, end) from offsets, slices values, and decodes UTF-8 to return the string.
- `take` byte-slices values and rebuilds offsets in local coordinates for contiguous selections, and builds a view with new indices for non-contiguous selections without copying values/offsets.