# struct module

Python's `struct` module is a standard library that **converts Python values (integers, floats, etc.) into binary (bytes) in a specified format (pack)** or **interprets binary data back into values (unpack)**. It is especially useful when dealing with "byte-level fixed layouts" such as file formats, network protocols, and binary logs.

## 1. Situations where the struct module is needed

Text (string)-based data is easy for humans to read and write, but binary formats are preferred when storage space, processing speed, or compatibility with external systems (C/C++/embedded/network specifications) is important. `struct` helps you create and interpret such binary formats reliably in Python.

## 2. Main functions

- `struct.pack(fmt, v1, v2, ...)`  
  Converts values into `bytes` according to the `fmt` (format string) specification.

- `struct.unpack(fmt, buffer)`  
  Interprets `buffer` (bytes/bytearray, etc.) according to the `fmt` specification and returns a tuple.

- `struct.calcsize(fmt)`  
  Returns the size in bytes occupied by `fmt`.

- `struct.pack_into(fmt, buffer, offset, v1, v2, ...)`  
  Packs values into an existing buffer (such as `bytearray`) starting at the `offset` position.

- `struct.unpack_from(fmt, buffer, offset=0)`  
  Unpacks values from the buffer starting at the `offset` position.

- `struct.Struct(fmt)`  
  Creates an object that "compiles" the format in advance, improving performance and readability when repeatedly packing/unpacking with the same format.

## 3. Understanding the format string (format string)

A format string is largely divided into the following two parts.

1) **Byte order (endianness) and alignment rule specifier** (optional, 1 character at the beginning)  
2) **Data type codes** (required)

### Byte order and alignment rules (1 character at the beginning)

- `@` : native byte order + native size + native alignment (padding may occur)  
- `=` : native byte order + standard size + no alignment  
- `<` : little-endian + standard size + no alignment  
- `>` : big-endian + standard size + no alignment  
- `!` : network byte order (big-endian) + standard size + no alignment

In general, if you want to create a binary format "in the same way every time", it is recommended to explicitly specify `<` or `>`.

### Commonly used type codes

- `?` : boolean (1 byte)
- `x` : 1-byte padding (padding means a space that is not data)
- `c` : 1-byte character (bytes of length 1)  
- `s` : N-byte string (you must specify the length in front like `Ns`)  
- `p` : Pascal string (in the form `Np`, the first 1 byte stores the length, followed by the data)

Integer types:
- `b` / `B` : 1-byte signed/unsigned  
- `h` / `H` : 2-byte signed/unsigned  
- `i` / `I` : 4-byte signed/unsigned  
- `q` / `Q` : 8-byte signed/unsigned  
- `n` / `N` : ssize_t / size_t (platform-dependent size)

Floating-point types:
- `e` : 2-byte floating point (float16)  
- `f` : 4-byte floating point (float32)  
- `d` : 8-byte floating point (float64)

Other:
- `P` : pointer size (platform-dependent, usually used together with `@`)

Repetition/length specification:
- A numeric prefix specifies the repetition count or string length.  
  Example: `3I` means three `I`s, and `16s` means a 16-byte string

## 4. Example code

### 1) Packing/unpacking integers and floats

```python
import struct

fmt = "<if"  # little-endian: int32 + float32
data = struct.pack(fmt, 1, 2.3)

print(data)         # print bytes
print(data.hex())   # print in hex for readability

unpacked = struct.unpack(fmt, data)
print(unpacked)     # it may look like (1, 2.299999952316284).
```

Due to limitations of binary representation, floating-point values may appear slightly different from the original value.

### 2) Checking byte differences by endianness

```python
import struct

x = 42

print(struct.pack(">I", x).hex())  # big-endian
print(struct.pack("<I", x).hex())  # little-endian
```

### 3) Checking byte size with calcsize

```python
import struct

fmt = "<Ih?"
print(struct.calcsize(fmt))  # 4 + 2 + 1 = 7 (no alignment)
```

### 4) Writing to and reading from a buffer with pack_into / unpack_from

```python
import struct

buf = bytearray(16)

# Pack the 32-bit integer 42 starting from offset 4, i.e., at buf[4:8]
struct.pack_into("<I", buf, 4, 42) 
print(buf.hex())

value = struct.unpack_from("<I", buf, 4)
print(value)  # (42,)
```

## 5. Little-endian and big-endian

Endianness (Endian) refers to **the order in which bytes are arranged** when storing **a value consisting of multiple bytes** in memory.  

- Little Endian: stores the **least significant byte (LSB)** first.  
- Big Endian: stores the **most significant byte (MSB)** first.

## 6. Endianness examples

A 32-bit integer `42` is `0x0000002A` in hexadecimal.  
If we split it into 4 bytes, we get the following.

- bytes: `00 00 00 2A`

When stored in memory:

- big-endian: `00 00 00 2A`  
- little-endian: `2A 00 00 00`

You can confirm it in Python as follows.

```python
import struct

x = 42
print(struct.pack(">I", x).hex())  # 0000002a
print(struct.pack("<I", x).hex())  # 2a000000
```

Now consider the value `0x12345678` (32-bit). If we split it into bytes, it is `12 34 56 78`.

- big-endian storage: `12 34 56 78`  
- little-endian storage: `78 56 34 12`

Here, the lower 8 bits (the least significant part) is `0x78`.  
In little-endian, the least significant byte is located at **the lowest address (the first byte)**.

```python
import struct

x = 0x12345678

b_le = struct.pack("<I", x)
b_be = struct.pack(">I", x)

print(b_le.hex())  # 78563412
print(b_be.hex())  # 12345678

print(hex(b_le[0]))  # 0x78
print(hex(b_be[0]))  # 0x12
```

In little-endian, it is convenient because you can look at the first byte and immediately get the lower 8 bits.

This time, suppose we extend a 16-bit value `0x1234` to a 32-bit value `0x00001234`.

- bytes of the 16-bit value:
  - little-endian (16-bit): `34 12`
  - big-endian (16-bit): `12 34`

- bytes when extended to 32-bit:
  - little-endian (32-bit): `34 12 00 00`
  - big-endian (32-bit): `00 00 12 34`

In other words, in little-endian it can look like "zeros are appended at the back", so the extension may feel intuitive, while in big-endian it becomes "zeros are appended at the front".
Therefore, most modern CPUs (Intel, AMD, ARM, etc.) adopt little-endian as the default.

However, network protocols usually standardize on big-endian and call it **network byte order**.  
So in `struct`, you can use `!` to pack/unpack in network byte order (big-endian).

```python
import struct

x = 0x12345678
print(struct.pack("!I", x).hex())  # it prints the same as big-endian.
```

## 7. struct: 3-line summary

- `struct` converts between values and byte strings (bytes) according to a **specified format (format string)**.
- Endianness refers to **the byte arrangement order of multi-byte values**.
- If you want to fix a binary format, it is a good habit to explicitly specify an endianness specifier such as `<` or `>`.
