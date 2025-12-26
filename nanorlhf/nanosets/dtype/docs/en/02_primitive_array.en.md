# PrimitiveArray

This document explains what `PrimitiveArray` is, why it is needed, and what design intentions its internal implementation has. `PrimitiveArray` is an array type that stores primitive types such as `int32`, `int64`, `float32`, `float64`, `bool` in a contiguous binary buffer (Buffer), and manages missing values (null) separately using a Validity Bitmap.

## 1. What is PrimitiveArray?

`PrimitiveArray` represents data using the following three components.

- `values: Buffer`  
  This is contiguous memory (a byte sequence) that contains the actual values. For example, for int32, 4 bytes per element are stored contiguously.

- `validity: Optional[Bitmap]`  
  This stores missing-value information in units of bits. 1 means valid and 0 means null. If all values are valid, it can be set to `None` to save additional memory.

- `indices: Optional[Buffer]`  
  If you have been following this series for a while, you may remember that we defined the absolute position index in the validity bitmap. This indices array is similar in that it is a buffer that enables the array to act like a view by not reading the values buffer contiguously from position 0, but instead referencing only the desired parts. If this indices buffer exists, this array behaves as a non-contiguous view.

In summary, `PrimitiveArray` is an Arrow-style array structure that stores values, missingness, and (if needed) reference indices separately.

## 2. Why use struct?

`PrimitiveArray` needs to store Python values (integers/floats/booleans) into a byte buffer and read them back. Here, `struct` specifies how many bytes this type uses and what rule to use to interpret the bytes, performing pack/unpack.

- `struct.pack_into(fmt, buffer, offset, value)`  
  Writes a value into a buffer at a specific position in binary form.

- `struct.unpack_from(fmt, buffer, offset)`  
  Reads a value from a buffer at a specific position in binary form.

Here, the format and element size are managed per dtype in the form `FMT[dtype] = (fmt, item_size)`. For example, int32 is generally 4 bytes and float64 is 8 bytes.

## 3. Index management

The key point of `PrimitiveArray` is that logically it looks like an array of length `length`, but internally it may be a contiguous array (no indices), or a non-contiguous view with indices. If you organize the index-related concepts first, it becomes much easier to understand methods like `__getitem__` and `take`.

### normalized index

Python allows negative indices like `arr[-1]`. Therefore, the internal implementation first normalizes indices to always be within the range `[0, length)`. Here we call this a normalized index.

For example, if the length is 5, the following holds.

- `-1 -> 4`
- `-2 -> 3`

In the code, this is done by `normalize_index(idx, self.length)`.

### base index

The base index is an index that indicates which element in the actual values buffer a logical index (idx from the user's perspective) refers to.

- If the array is contiguous, the logical index and actual position are the same, so `base_index(idx) = normalized_idx`.
- If the array is a non-contiguous view, indices maps the logical index to the actual position, so `base_index(idx) = indices[normalized_idx]`.

The implementation has the following form.

```python
def base_index(self, idx: int) -> int:
    normalized_idx = normalize_index(idx, self.length)
    if self.is_contiguous():
        return normalized_idx
    return unpack_int32(self.indices, normalized_idx)
```

With this single concept, the same rule can be applied to all subsequent reads/writes.

## 4. The length of PrimitiveArray (length)

The key point of `PrimitiveArray.__init__` is how to represent the logical length (logical length).

- If there is no `indices`, this array is an array that has never been sliced, and the logical length becomes `length`.
- If there is an `indices`, the logical length becomes `len(indices) // 4`. Since indices is an int32 array, one entry is 4 bytes, so the number of elements corresponds to the byte length divided by 4.

That is, if indices exists, this array becomes a view that holds the values as-is and reads only the positions pointed to by indices, and the logical length is determined by indices.

## 5. `__getitem__`: How do we read values?

`__getitem__` handles integer indices and slices.

```python
def __getitem__(self, key: Union[int, slice]):
    if isinstance(key, int):
        if self.is_null(key):
            return None
        offset = self.base_index(key) * self.item_size
        return struct.unpack_from(self.fmt, self.values.data, offset)[0]

    if isinstance(key, slice):
        start, stop, step = key.indices(self.length)
        return self.take(range(start, stop, step))
```

The behavior of integer indexing follows this order.

1) Check whether the position is null.
2) If null, return `None`.
3) If not null, compute the actual values position (element-wise) using `base_index(key)`.
4) Compute the byte offset as `offset = base_index(key) * item_size`.
5) Interpret bytes into a value using `struct.unpack_from`.

For example, if `item_size = 4` (int32) and `base_index(key) = 10`, then `offset = 40`, meaning you should read 4 bytes starting from the 40th byte in the bytes object.

A slice (`array[start:stop:step]`) is delegated internally to `take(range(...))`. (The `take` method will be described below.)

## 6. Why there is no `__setitem__`: aiming for immutability

`PrimitiveArray` does not have `__setitem__`. This is because this implementation aims to treat arrays as immutable by following the design philosophy of Apache Arrow.

If you aim for an immutable structure, you get the following advantages.

- Safe sharing: even if multiple array views share the same `values`/`validity` buffers, the meaning of other views will not break because someone changed the value.
- zero-copy optimization: the strategy of expressing operations like `slice` or `take` as a zero-copy view becomes much simpler and safer.
- Favorable for parallel processing: even if multiple threads/operations read the same data simultaneously, data races are reduced.

Therefore, rather than modifying values in place, `PrimitiveArray` recommends creating a brand-new array or creating a zero-copy view through methods like `from_list`, `builder.finish`, `take`, `slice`.

## 7. `take(indices)`: The concrete behavior of selection operations

`take` returns the result of selecting elements by specific indices as a new `PrimitiveArray`. The important point here is that it tries to express the result as a view while not copying the values buffer as much as possible (zero-copy).

### Input: a list or sequence of logical indices

The input `indices` to `take` are logical indices from the user's perspective. That is, regardless of whether it is contiguous or non-contiguous, you can think of the input as coming in as logical indices first.

### Normalize input indices

First, normalize all indices to be within `[0, length)`.

```python
normalized = [normalize_index(i, self.length) for i in indices]
```

From this point on, you can proceed with the same logic without worrying about negative indices.

### Determine whether it is a contiguous slice

Check whether the normalized indices increase consecutively.

```python
is_contiguous_slice = all(normalized[k] + 1 == normalized[k + 1] for k in range(num_items - 1))
```

The contiguity here is contiguity in terms of logical indices. For example, `[3, 4, 5, 6]` is contiguous, while `[3, 5, 8]` is not.

### When it is a contiguous slice

If contiguous, it can be summarized as `start = normalized[0]`, `length = num_items`. The key branch is now whether the original is contiguous.

#### If the original is contiguous: slice values/validity directly

In a contiguous original, the logical index is the position in values, so values is also a truly contiguous region. Therefore, you can slice values at the byte level and slice validity over the same range to produce the result.

```python
byte_offset = start * self.item_size
byte_length = length * self.item_size
sub_values = self.values.slice(byte_offset, byte_length)
# Slice the values buffer as values[byte_offset:byte_offset + byte_length].
sub_validity = self.validity.slice(start, length) if self.validity else None
# Since values and validity manage data via memoryview internally, no data copy occurs.
return PrimitiveArray(self.dtype, length, sub_values, sub_validity)
```

In this case, the result becomes a contiguous array that does not need indices.

#### If the original is non-contiguous: slice only indices to make a smaller view

In a non-contiguous original, even if the region is logically contiguous, there is no guarantee that it is contiguous in values. This is because values is a shared original buffer, and the actual position is determined by indices.

Therefore, in this case, you do not slice values, but slice only the corresponding range in the indices buffer to make a result view. Since indices consists of int32 entries, one entry is 4 bytes.

```python
index_offset = start * 4
index_length = length * 4
sub_indices = self.indices.slice(index_offset, index_length)
return PrimitiveArray(self.dtype, length, self.values, self.validity, sub_indices)
```

In this case, the result is still a non-contiguous view, and it shares values/validity with the original.

### When it is not a contiguous slice

If indices jump, have duplicates, or are in an arbitrary order, you cannot express values as a simple slice. In this case, express the result as a new indices mapping.

The important point here is that the result indices must store not logical indices, but base indices, i.e., the actual positions in values.

- If the original is contiguous, the actual position is the same as the logical index, so the base index is `normalized` itself.
- If the original is non-contiguous, `normalized` is the original logical index, so you need to map it once more through indices to obtain the actual position.

```python
base_indices = normalized if self.is_contiguous() else [unpack_int32(self.indices, i) for i in normalized]
```

Now `base_indices` is the list of actual element positions to read from the values buffer. Pack this into an int32 buffer to create new indices, and return the result while sharing values/validity.

```python
new_indices = pack_int32(base_indices)
return PrimitiveArray(self.dtype, len(base_indices), self.values, self.validity, new_indices)
```

The properties of this design are as follows.

- The values buffer is never newly copied.
- Even if you apply take multiple times, the result can accumulate as an indices-based view.

## 8. `to_list`: Convert to a Python list

`to_list` iterates through all elements and appends `None` if null, otherwise reads the value with `struct.unpack_from` and returns Python values.

```python
output = []
for i in range(self.length):
    if self.is_null(i):
        output.append(None)
    else:
        offset = self.base_index(i) * self.item_size
        value = struct.unpack_from(self.fmt, self.values.data, offset)[0]
        output.append(value)
return output
```

This function is convenient for exporting outward, but since it creates Python objects via `struct.unpack_from`, it is costly for large-scale data.

## 9. `from_list`: Create from a Python list

`from_list` infers the dtype of the input list (`infer_primitive_dtype`) or uses a dtype specified by the user. Then, using `PrimitiveArrayBuilder` described below, it does the following.

- Separates and accumulates the `values` list and the `validity` list.
- Finally writes values into a byte buffer with `struct.pack_into`.
- Converts validity with `Bitmap.from_list` (it may be `None` if all are valid).

The reasons for strictly handling input value types per dtype are as follows.

- If a float is provided to an integer dtype, unintended loss/conversion may occur, so it is treated as an error.
- bool can be represented as an integer internally, so it is allowed in some cases, but for BOOL dtype only bool is allowed.
- int32 range checks prevent overflow.

## 10. `infer_primitive_dtype`: dtype inference

```python
def infer_primitive_dtype(values: List[Optional[PrimitiveType]]) -> DataType:
    saw_float = False
    saw_int = False
    saw_bool = False

    for value in values:
        if value is None:
            continue
        if isinstance(value, bool):
            saw_bool = True
            continue
        if isinstance(value, float):
            saw_float = True
        elif isinstance(value, int):
            saw_int = True
        else:
            raise ValueError(f"Unsupported primitive type: {type(value).__name__}")

    if saw_float:
        return FLOAT64
    if saw_int:
        return INT64
    if saw_bool:
        return BOOL

    raise ValueError("Cannot infer primitive dtype from empty or unsupported values")
```

The `infer_primitive_dtype` function inspects the types of input values and infers an appropriate `DataType`.
Real Arrow would infer more complexly, but here we decide the dtype with a simple priority of float > int > bool.
If [1, 3.14, True] is given, this list is inferred as float64.

## 11. `PrimitiveArrayBuilder`: Building the buffer with `pack_into`

The builder does not stack Python values directly into bytes. Instead, it first collects the `values` list and `validity` list, and then builds the buffer at once at the end.

The key steps are as follows.

1) Pre-allocate `raw_buffer = bytearray(num_items * item_size)`.
2) While increasing `offset` by `item_size`, write values using `struct.pack_into`.
3) (If needed) create a validity bitmap. `Bitmap.from_list` returns None as-is if `self.validity` is None.

```python
raw_buffer = bytearray(num_items * self.item_size)

offset = 0
for value in self.values:
    struct.pack_into(self.fmt, raw_buffer, offset, value)
    offset += self.item_size

buffer = Buffer.from_bytearray(raw_buffer)
validity = Bitmap.from_list(self.validity)
return PrimitiveArray(self.dtype, num_items, buffer, validity, indices=None)
```

If validity exists, even for null values a placeholder value is needed in the values buffer.
So for BOOL it inserts `False`, and for other numeric types it inserts `0`, and the actual nullness is determined by validity.

## 12. PrimitiveArray: 3-line summary

- `PrimitiveArray` stores primitive-type values in a contiguous binary `Buffer`, and separates nulls into a `Bitmap`.
- `base_index` converts a logical index into the actual position in values: unchanged if contiguous, mapped through indices if non-contiguous.
- Following Arrow’s philosophy, it aims for immutability and does not provide `__setitem__`, and `take` represents contiguous selections as slices and arbitrary selections as new indices, constructing views without copying values.