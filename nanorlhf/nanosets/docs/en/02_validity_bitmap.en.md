# Validity Bitmap
This document explains Validity Bitmap.

## 1. What is Validity Bitmap?
Validity Bitmap is a method for efficiently managing missing values (null values).
Real-world datasets can contain many missing values (None, NaN, etc.).
If we store such missing values together with the data, different Python data types get mixed.

Consider the following data.

```python
data = [10, None, 30, 40]
```

Python internally stores these as object pointers (PyObject*),
which makes it likely that they will be scattered across non-contiguous memory regions.

| Index | Value | Actual Storage Type            |
|-------|-------|--------------------------------|
| 0     | 10    | PyObject* (points to int)      |
| 1     | None  | PyObject* (points to NoneType) |
| 2     | 30    | PyObject* (points to int)      |
| 3     | 40    | PyObject* (points to int)      |

This fragmented memory layout causes several issues.

- SIMD optimization becomes difficult.
- Cache efficiency decreases.
- PyObject overhead is incurred.

To address this, we can consider storing missing values separately.

```python
values = [10, 0, 30, 40]
validity = [1, 0, 1, 1]  # 1 indicates a valid value, 0 indicates a missing value.
```

By separating values from validity, numeric data can be stored in a compact, contiguous memory block (int32, float64, etc.).
We can store this in byte-level (`bytearray`), ensuring that the data is stored in a physically contiguous space.
So we can solve the three issues mentioned above.

```python
values = bytearray([10, 0, 30, 40])  # int with contiguous memory layout
validity = bytearray([0b00001011])  # Bitmap to store missing-value information
```

| Index | Value | Actual Storage Type               |
|-------|-------|-----------------------------------|
| 0     | 10    | int (stored in contiguous memory) |
| 1     | 0     | int (stored in contiguous memory) |
| 2     | 30    | int (stored in contiguous memory) |
| 3     | 40    | int (stored in contiguous memory) |

## 2. Why is a contiguous memory layout important?

### SIMD optimization
SIMD is short for Single Instruction, Multiple Data, a CPU feature that allows the same operation to be performed on multiple data elements at the same time using a single instruction.
  
For example, instead of performing four additions separately:

```python
[1+1, 2+2, 3+3, 4+4]  # process one by one
```

With SIMD, the CPU can process them at once via vectorized instructions:

```python
[1, 2, 3, 4] + [1, 2, 3, 4]  # computed together internally
```

However, SIMD works effectively only when data is stored contiguously in memory.
If data is scattered like Python objects, it becomes hard for the CPU to efficiently load or process many values at once.
For this reason, libraries like Arrow require a contiguous memory layout, enabling them to maximize the vectorization performance of CPU hardware.

### Cache efficiency
Modern CPUs are much faster than main memory, so the time spent loading data from memory can become the bottleneck.
To mitigate this, CPUs use a small, fast memory called cache, storing recently accessed data there so it can be accessed quickly when needed again.

![](https://github.com/hyunwoongko/nanoRLHF/blob/main/nanorlhf/nanosets/docs/assets/cache.png?raw=true)

When loading data from memory, the CPU brings multiple contiguous bytes into the cache at once.
If the data is stored contiguously in memory, like NumPy or Arrow arrays, the CPU can load multiple consecutive elements into the cache with a single load.
As a result, when processing sequentially, nearby data is already in the cache and can be accessed quickly.

In contrast, if data is scattered across memory like a list of PyObject pointers in Python, each element can be in a completely different location.
In that case, the CPU cannot reuse cached data and must keep fetching from main memory, causing frequent cache misses and a large performance drop.
This is why Arrow is designed to store all values in contiguous buffers, so it can fully benefit from CPU cache and memory prefetching during sequential access.

## 3. Why store validity as a bitmap?
A Validity Bitmap stores missing-value information as a bitmap, i.e., at the bit level rather than the byte level.
For example, suppose we want to store missing-value information for 8 elements.

| Index        | 0  | 1 | 2  | 3  | 4 | 5  | 6  | 7  |
|--------------|----|---|----|----|---|----|----|----|
| Value        | 10 | - | 30 | 40 | - | 60 | 70 | 80 |
| Validity Bit | 1  | 0 | 1  | 1  | 0 | 1  | 1  | 1  |

If we store validity at the byte level, we need 8 bytes in total.
But since validity is represented only by 0 and 1, it can actually be stored at the bit level.
If we use bits for validity so that each bit represents the validity of one element, we can store the same information in 1 byte (=8 bits).
This allows us to store missing-value information using 8 times less space.

## 4. Implementation of `get_validity`

The full code is as follows.
```python
def get_validity(buffer: Buffer, i: int) -> bool:
    byte, bit = divmod(i, 8)
    b = buffer.data[byte]
    mask = (1 << bit)
    check = b & mask
    return check != 0
```

`i` is the index of the element whose validity we want to look up in the bitmap. For example, suppose we want to look up the validity of the 13th element.

First, we need to compute which byte and which bit within that byte correspond to 13. We can easily do this using `divmod`.

```python
byte, bit = divmod(13, 8)
# byte = 1, bit = 5
```
Then we read the corresponding byte (byte 1) from the bitmap.

```python
b = bitmap.data[byte]
# b = 0b00101101  # example value
```

Now we create a mask for the bit position we care about.

```python
mask = (1 << bit)  # shift 1 left by 5
# mask = 0b00100000
```

Using a bitwise AND, we check whether that bit is 1 or 0.

```python
check = b & mask

# b     = 0b00101101
# mask  = 0b00100000
# ------------------
# check = 0b00100000
```

If that bit corresponds to a missing value, the AND operation will yield all zeros and `check` will become `0b00000000`, i.e., 0.
But if it is a valid value, at least one bit will be 1, and `check` will become some non-zero value.

Therefore, we return validity by checking whether `check` is zero or not.

```python
return check != 0  # True (valid value)
```

## 5. Implementation of `set_validity`
The full code is as follows.
```python
def set_validity(buffer: Buffer, i: int, valid: bool):
    byte, bit = divmod(i, 8)
    b = buffer.data[byte]

    if valid:
        mask = (1 << bit)
        packed = b | mask
    else:
        mask = ~(1 << bit) & 0xFF
        packed = b & mask

    buffer.data[byte] = packed
```

`i` is the index of the element we want to set in the bitmap, and `valid` indicates whether to set that element as valid (True) or missing (False).

Suppose we want to set the 13th element to missing.
Then we need to compute which byte and which bit correspond to the 13th element. We use the same method as in missing-value lookup.

```python
byte, bit = divmod(13, 8)
# byte = 1, bit = 5
```

Then we read the corresponding byte (byte 1) from the bitmap.

```python
b = bitmap.data[byte]
# b = 0b00001101  # example value
```

If `valid` is True, we set the bit using a bitwise OR.

```python
mask = (1 << bit)  # shift 1 left by 5
packed = b | mask

# b     = 0b00001101
# mask  = 0b00100000
# ------------------
# packed= 0b00101101  # the 5th bit is set to 1
```

Conversely, if `valid` is False, we clear the bit using a bitwise AND.

```python
mask = ~(1 << bit)  # shift 1 left by 5, then invert bits
packed = b & mask

# b      = 0b00101101
# mask   = 0b11011111
# -------------------
# packed = 0b00001101  # the 5th bit is set to 0
```

One thing to be careful about is that bitwise NOT (~) can invert beyond the region we want.
In C-based systems like NumPy, integers have a fixed number of bits (e.g., 8-bit, 16-bit, 32-bit), so this problem does not occur,
but because we are handling bitmaps in Python, we cannot easily restrict integers to a fixed bit width.

For example, in a 16-bit integer, inverting the 5th bit yields `0b1111111111011111`. We want to use only the lower 8 bits in the AND operation,
so we need to remove the unused upper 8 bits so they do not affect the operation.
For reference, 0xFF is a value where only the lower 8 bits are set to 1; AND-ing any value with it leaves only the lower 8 bits.

```python
mask = ~(1 << bit)
mask = mask & 0xFF  # keep only the lower 8 bits

# mask = 0b1111111111011111
# 0xFF = 0b0000000011111111
# -------------------------
# mask = 0b0000000011011111
```

Reflecting this, we perform the AND operation to compute the value.

```python
mask = ~(1 << bit) & 0xFF
packed = b & mask

# b      = 0b0000000000101101
# mask   = 0b0000000011011111
# ---------------------------
# packed = 0b0000000000001101  # the 5th bit is set to 0
```

Finally, we store the modified packed value back into the bitmap.

```python
bitmap.data[byte] = packed
# bitmap.data[1] = 0b0000000000001101
```

## 6. Implementation of `slice`
The full code is as follows.
```python
import math


def slice(buffer: Buffer, offset: int, i: int, slice_length: int) -> Bitmap:
    """Create a new Bitmap view that references bits from i to i + slice_length."""

    abs_bit_position = offset + i
    byte, bit = divmod(abs_bit_position, 8)
    needed_bytes = math.ceil((bit + slice_length) / 8)
    sliced_buffer = buffer.data[byte:byte + needed_bytes]
    return Bitmap(slice_length, sliced_buffer, offset=bit)
```

The key idea of bitmap slicing is not to copy the original bitmap and create a new bitmap,
but, in the spirit of zero-copy, to create a new view that points to a specific range of the original bitmap.

Therefore, while keeping the original bitmap unchanged,
we implement it so that metadata (offset) manages which range of this bitmap we are currently viewing.

To implement slicing this way, we need the concept of absolute bit position.
Absolute bit position is the value obtained by converting the logical index `i` into the bit position in the actual buffer.
In this document, we define it as follows.

- logical index: the index `i` visible to the user (starting from 0)
- offset: the value indicating from which bit this Bitmap view should start reading
- absolute bit position: `abs_bit = offset + i`

In other words, even for the same bitmap, if it has already been sliced somewhere, we should not read from bit 0.
We store that as the value offset, and use logical index `i` plus offset as the absolute bit position.
For example, if `offset = 3` and the user wants logical index `i = 2`, we should not look up the 2nd element of the original bitmap,
but rather the bit at `abs_bit_position = 3 + 2 = 5`.

Now let's look at the code.

As mentioned above, we convert logical index `i` into absolute bit position.

```python
abs_bit_position = offset + i
```

Using absolute bit position, we compute the starting position of the new view.
If `i` is 2 and `offset` is 3, absolute bit position is 5,
and byte is 0 and bit is 5.

```python
byte, bit = divmod(abs_bit_position, 8)
```

We compute how many bytes are needed to represent this view.
Pay close attention to why we divide `bit + slice_length` by 8 and then take the ceiling.

If `bit` is 3 and `slice_length` is 10,
- bit + slice_length = 3 + 10 = 13
- 13 / 8 = 1.625
- ceiling(1.625) = 2

We need a total of 2 bytes.

This is because, to read 10 bits starting from the 3rd bit of the first byte in the buffer, we must read 5 bits from the 3rd bit to the end of the first byte,
and the remaining 5 bits must be read from the second byte.

```python
needed_bytes = math.ceil((bit + slice_length) / 8)
```

We slice the required byte range from the original buffer to create a new buffer view.
For reference, Buffer internally uses `memoryview`, so no new copy occurs in this process.

```python
sliced_buffer = buffer.data[byte:byte + needed_bytes]
```

Finally, we create and return the new Bitmap view.

```python
return Bitmap(slice_length, sliced_buffer, offset=bit)
```

And the existing `get_validity` and `set_validity` functions must be modified to use this new offset metadata.
Then, even in a sliced bitmap view, we can read and write bits at the correct positions.

```python
def get_validity(buffer: Buffer, offset: int, i: int) -> bool:
    abs_bit_position = offset + i  # <--- apply offset
    byte, bit = divmod(abs_bit_position, 8)
    b = buffer.data[byte]
    mask = (1 << bit)
    check = b & mask
    return check != 0


def set_validity(buffer: Buffer, offset: int, i: int, valid: bool):
    abs_bit_position = offset + i  # <--- apply offset
    byte, bit = divmod(abs_bit_position, 8)
    b = buffer.data[byte]

    if valid:
        mask = (1 << bit)
        packed = b | mask
    else:
        mask = ~(1 << bit) & 0xFF
        packed = b & mask

    buffer.data[byte] = packed
```

## Validity Bitmap: 3-line summary
- Validity Bitmap stores missing-value information at the bit level, reducing memory usage while maintaining a contiguous memory layout.
- A contiguous memory layout improves SIMD optimization and cache efficiency, contributing to better performance.
- A bitmap can efficiently manage missing values by implementing bit-level get/set operations and slicing.