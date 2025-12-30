# IPC (Inter-Process Communication)
This document explains the IPC (Inter-Process Communication) file format used in nanosets. The goal is to handle data as close to Zero-copy as possible when serializing/deserializing a `Table` to/from a file. In particular, the key idea is to reduce unnecessary copies in the read path by using `mmap`-based memory mapping.

## 1. Minimal background to understand `mmap`

### RAM is divided into "kernel space" and "user space"
In modern operating systems, you can think of RAM as being conceptually divided into two regions.

- Kernel space: the memory region used by the operating system (OS). System-wide code such as disk caches, I/O buffers, and device control runs here.
- User space: the memory region used by regular programs we run (including Python). Each process has its own user space, and it is generally isolated from other processes' user spaces.

The purpose of this separation is safety and isolation. If a program accidentally corrupts OS memory, the entire system can be at risk, so the OS prevents regular programs from directly touching kernel space.

### Why does reading a file normally cause two "copies"?
When you read a file in Python (e.g., using `fp.read()`), conceptually the flow looks like this.

```text
[Disk] → Copy → [Kernel space] → Copy → [User space]
```

- First Copy: data read from disk is brought into kernel space (the kernel page cache/buffers).
- Second Copy: that data is copied again into the user space memory used by the program.

This "double copy" consumes extra CPU and memory bandwidth. The larger the data, the higher the cost.

### What does `mmap` reduce?
`mmap` (memory mapping) is a technique that reduces the "second Copy (kernel → user)" in the flow above.

With `mmap`, instead of the OS "copying the file data into user space," it maps the file pages in kernel space into the user-space address space (virtual address space).

```text
[Disk] → Copy → [Kernel space] ↔ [User virtual address space]
```

Key points:
- The file data itself lives in the kernel page cache.
- The user program accesses it through "addresses that point to that data."
- So it does not create "another large duplicate copy" in user space.

### What is a page?
The OS does not manage memory byte-by-byte; it manages memory in fixed-size blocks called pages.
On most systems, the page size is 4 KB.

So when data is brought from disk into RAM, it is not fetched one byte at a time, but in "page-sized chunks."
In kernel space, there is a region that stores these pages, called the page cache.

### If you call `mmap`, does the whole file get loaded into RAM at once?
No. `mmap` is not "immediately reading the entire file," but rather "mapping addresses."

In reality, when the program "reads from the mapped address for the first time," the OS loads the corresponding page from disk into the page cache. The event that occurs at that moment is called a page fault.

This behavior is called demand paging or lazy loading.

```text
Before 1st access:
    [User virtual address space] → Page Fault → OS → [Disk] → Copy → [Kernel space]

After 1st access:
    [User virtual address space] ↔ [Kernel space]
```

In summary:
- When `mmap()` is called: "only the address space is prepared"; the actual file data may not have been read yet
- On first access: a page fault occurs; the OS loads only the needed pages from disk
- On later access: pages already in cache can be accessed quickly

| Stage            | User Space (Process Memory)               | Kernel Space (Page Cache)           |
|------------------|-------------------------------------------|-------------------------------------|
| `mmap()` called  | Space for virtual addresses reserved      | No file data loaded (still on disk) |
| After 1st access | Address → Kernel page mapping established | File page loaded into page cache    |
| Later accesses   | Reads from mapped addresses               | File page remains in page cache     |

### How are `mmap` and `memoryview` different?
Both enable Zero-copy access, but at different levels.

- `mmap`: operates at the OS level.
  - It is connected to disk I/O, page cache, and page faults.
  - The core benefit is reducing copies between kernel and user space.

- `memoryview`: operates at the Python level.
  - It provides a Zero-copy "view" over bytes/buffer objects already in memory (e.g., `bytes`, `bytearray`, `mmap` objects).
  - It does not itself handle disk I/O or paging.

Summary:
- `mmap`: OS-level Zero-copy (disk ↔ virtual memory)
- `memoryview`: Python-level Zero-copy (RAM ↔ Python object)

## 2. What is IPC and why is it needed?
IPC usually means inter-process communication, but here it is used in a broader sense: a binary format for exchanging data across runtimes/processes/languages.

In nanosets, IPC aims to solve the following:

- Save a `Table` (schema + batches + column arrays) to a file
- When loading back, reconstruct Python objects while reusing value buffers (`values`/`offsets`/`validity`/`indices`) as close to Zero-copy as possible
- Maintain a layout independent of language/runtime, aligned with an Arrow-style (buffer-based, `validity` bitmap, `offsets`-based) structure

In other words, it is not a "human-friendly rows-only (JSON)" format, but a format that "preserves buffer layout and reattaches quickly."

## 3. File layout
The file layout produced by `write_table(fp, table)` is as follows.

- `MAGIC` (5 bytes)
- Header Length (4 bytes, little-endian `uint32`)
- Header (JSON bytes)
- Buffers (raw binary blobs; `values`/`offsets`/bitmaps/`indices`, etc.)

### The role of MAGIC
`MAGIC` is a fixed byte sequence used to quickly identify the file format.

- In this implementation, `MAGIC = b"NANO0"`
- Meaning: this is a nanosets IPC file, version 0

### The role of the Header (JSON)
The header is metadata that contains:

- `Schema`: field names, dtype, nullable
- `RecordBatch`es: batch lengths, and per-column (`array`) type (`kind`) and configuration
- `buffers`: the list of blobs as (`offset`, `length`)

JSON is used because it is human-readable, simple to implement, and easy to parse across languages. The actual data body is stored as raw blobs, and only metadata is JSON, so overhead is minimized.

## 4. Overall flow of `write_table`
`write_table(fp, table)` performs two major steps.

### Collecting blobs (buffers)
It iterates arrays and accumulates all required `Buffer`s into `blobs: List[memoryview]`.  
`add_buffer(b: Buffer) -> int` appends `b.data` to the blob list and returns its index.

Here, blobs are the physical buffers inside each `Array`.

- `PrimitiveArray`: `values`
- `StringArray`: `offsets` + `values`
- `ListArray`: `offsets` + `child` (recursive)
- `StructArray`: `children` (recursive)
- `TensorArray`: special handling (see below)
- validity bitmap: `validity.buffer`
- indices buffer (view): `indices`

### Writing the header and then writing the file
It computes offsets based on the cumulative lengths of buffers and records them into `header["buffers"]`.

Then it writes the following in order:

```python
fp.write(MAGIC)
fp.write(struct.pack("<I", len(header_bytes)))
fp.write(header_bytes)
for blob in blobs:
    fp.write(blob)
```

## 5. Why is `read_table` (mmap-based deserialization) important?
`read_table(path)` opens the file, memory-maps the entire file with `mmap`, reads the header, and reconstructs buffers by creating views via slicing `memoryview(mm)`.

The key points are:
- it does not rebuild buffers as Python `bytearray`/`bytes`
- it creates only "views" using `memoryview` slices on top of the mapped `mm`
- as a result, the value buffers (`values`/`offsets`/`validity`/`indices`) sit directly on top of the file mapping (Zero-copy oriented)

### File parsing order
1) validate `MAGIC`  
2) read header length (4 bytes, `"<I"`)  
3) read header JSON bytes and `json.loads`  
4) create a `memoryview` for the raw blob region according to `header["buffers"]`, and slice each blob with `Buffer.from_memoryview`

```python
data_start = mm.tell()
base_view = memoryview(mm)[data_start : data_start + total]

buffers: List[Buffer] = []
for buffer in header["buffers"]:
    start = buffer["offset"]
    end = start + buffer["length"]
    buffers.append(Buffer.from_memoryview(base_view[start:end]))
```

Since `Buffer` internally holds a `memoryview`, the entire read path becomes `mmap` + `memoryview`, operating close to Zero-copy.

## 6. `decode_array`: reconstruct arrays from metadata
`decode_array(inputs)` reads the header metadata to decide which `Array` to create, then attaches the required buffers by retrieving them from `buffers[idx]`.

The important perspective is:
- you create new `Array` Python objects
- but the value buffers they hold are not "new memory copied from the file"
- they reuse the views on top of `mmap` (`Buffer(memoryview(mm)[...])`) (Zero-copy oriented)

### Validity reconstruction is also Zero-copy oriented
If validity exists, `Bitmap` is reconstructed by receiving a `Buffer`:

```python
validity_buffer = buffers[inputs["validity"]]
validity = Bitmap(validity_length, validity_buffer)
```

That is, the validity bitmap also references the file-mapped memory directly.

## 7. `TensorArray` and `torch.frombuffer`: the key point when combined with `mmap`
For IPC, `TensorArray` does not "store each row tensor separately." Instead, it stores the raw bytes from stacking tensors into one blob (`values`).

When reading, the crucial function is `torch.frombuffer`.

### What does `torch.frombuffer` do?
`torch.frombuffer(buffer, dtype=..., count=...)` is an API that "creates a tensor from a byte buffer."
When possible, it does not copy bytes into a new allocation; it creates a **tensor view** on top of the existing buffer (Zero-copy oriented).

In this document, that means:

- `values_buffer.data` may be a `memoryview` of the `mmap` file
- calling `torch.frombuffer(values_buffer.data, ...)` can produce a tensor that lives "on top of the file-mapped memory" (as a view)
- i.e., the design aims to avoid copying large tensor data into a new allocation

### What the code actually does
```python
base_1d = torch.frombuffer(values_buffer.data, dtype=scalar_dtype, count=total_elems)
base_block = base_1d.view(base_length, *elem_shape) if elem_shape else base_1d.view(base_length)

base_tensors: List[torch.Tensor] = [base_block[i] for i in range(base_length)]
return TensorArray(base_tensors, validity, indices)
```

- create a 1D tensor with `torch.frombuffer(...)`
- reshape it to `(base_length, *shape)` via `.view(...)`
- slice per-row as `base_block[i]`, then wrap them in `TensorArray`

## 8. IPC: 3-line summary
- `mmap` maps file data from the kernel page cache into the user virtual address space, reducing the "second Copy (kernel → user)" (Zero-copy oriented).
- `read_table` creates `Buffer`s via `mmap` + `memoryview` slices, aiming to re-reference `values`/`offsets`/`validity`/`indices` on top of the file mapping with Zero-copy.
- `TensorArray` is stored as one raw-bytes blob, and on read it uses `torch.frombuffer` to build tensor views on top of the file-mapped buffer when possible (Zero-copy oriented).