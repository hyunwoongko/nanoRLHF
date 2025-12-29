# IPC
This document explains the IPC (Inter-Process Communication) file format used in nanosets. 
The goal is to handle data as close to zero-copy as possible when serializing/deserializing a `Table` to/from a file, and in particular to reduce unnecessary copies on the read path by using `mmap`-based memory mapping.

## 1. What is IPC and why is it needed?
IPC usually refers to communication between processes, but here it is used in a broader sense to mean a binary format that can move data across runtimes, processes, and languages.

The problems this IPC implementation aims to solve are:

- Save a `Table` (schema + batches + column arrays) to a file
- When loading again, reconstruct Python objects while reusing value buffers (`values`/`offsets`/`validity`/`indices`, etc.) with as little copying as possible
- Maintain an Arrow-style (buffer-based, `validity` bitmap, `offsets`-based) layout that is independent of language/runtime

## 2. File layout
The file layout produced by `write_table(fp, table)` is:

- `MAGIC` (5 bytes)
- Header Length (4 bytes, little-endian `uint32`)
- Header (JSON bytes)
- Buffers (raw binary blobs; `values`/`offsets`/bitmaps/`indices`, etc.)

### Role of MAGIC
`MAGIC` is a fixed byte sequence used to quickly identify the file format.

- In this implementation, `MAGIC = b"NANO0"`
- Meaning: this is a nanosets IPC file, version 0

### Role of the Header (JSON)
The header contains metadata such as:

- `Schema`: field name, dtype, nullable
- `RecordBatch`es: batch length, and each column (`array`)’s type (`kind`) and construction info
- `buffers`: the list of blobs’ (`offset`, `length`)

JSON is used because it is human-readable, simple to implement, and easy to parse across languages. The actual data body is kept as raw blobs, and only the metadata is JSON, minimizing overhead.

## 3. Overall flow of `write_table`
`write_table(fp, table)` performs two major steps.

### Collecting blobs (buffers)
It traverses arrays and accumulates all required `Buffer`s into `blobs: List[memoryview]`.  
`add_buffer(b: Buffer) -> int` appends `b.data` to the blobs array and returns its index.

The important point here is that blobs are internal buffers of individual `Array`s:

- `PrimitiveArray`: `values`
- `StringArray`: `offsets` + `values`
- `ListArray`: `offsets` + `child` (recursive)
- `StructArray`: `children` (recursive)
- `TensorArray`: special handling (see below)
- validity bitmap: `validity.buffer`
- indices buffer (view): `indices`

### Creating the header and writing to the file
It computes offsets based on the cumulative lengths of buffers and records them into `header["buffers"]`.

Then it writes the following to the file in order:

'''python
fp.write(MAGIC)
fp.write(struct.pack("<I", len(header_bytes)))
fp.write(header_bytes)
for blob in blobs:
    fp.write(blob)
'''

## 4. `encode_array`: array metadata serialization rules
`encode_array(array: Array)` creates the metadata that goes into the header depending on the array type, and registers the necessary buffers into blobs.

Common metadata fields:

- `dtype`: `{"kind": dtype.name}`
- `length`: `array.length`
- if `validity` exists:
  - `validity`: blob index of the validity buffer
  - `validity_length`: `len(array.validity)`
- if `indices` exists:
  - `indices`: blob index of the indices buffer

### PrimitiveArray
- `kind = "primitive"`
- `values = add_buffer(array.values)`

### StringArray
- `kind = "string"`
- `offsets = add_buffer(array.offsets)`
- `values = add_buffer(array.values)`

### ListArray
- `kind = "list"`
- `offsets = add_buffer(array.offsets)`
- `child = encode_array(array.child)` (the child array metadata is included recursively)

### StructArray
- `kind = "struct"`
- `names = array.field_names`
- `children = [encode_array(ch) for ch in array.children]`

### TensorArray
`TensorArray` does not end with a single `values` buffer, so it is handled by a separate function `encode_tensor_array`.

## 5. IPC representation of `TensorArray`
Because `TensorArray` internally is `List[Optional[torch.Tensor]]`, storing each row tensor as a separate blob would be inefficient.  
Instead, this design is used:

- `base_length`: the base storage length of the `TensorArray` (= `len(array.tensors)`)
- store dtype/shape/device as metadata
- stack `base_length` tensors into one large contiguous block via `stack`
- store its raw bytes as a single `values` blob

### dtype/shape/device consistency checks
It picks the first non-`None` tensor as the prototype and checks that all tensors in `base_tensors` have the same dtype/shape/device.

This implementation also supports only CPU tensors in IPC:

- error if `prototype.device.type != "cpu"`

### How `None` is handled
Because rows may be `None`, a placeholder tensor is needed to build the stack.

- if `None`, insert `torch.zeros(elem_shape, dtype=scalar_dtype, device=device)`
- for non-contiguous tensors, call `.contiguous()`
- finally build one block via `torch.stack(...).contiguous()`

Then:

- `raw_bytes = stacked_tensor.numpy().tobytes(order="C")`
- wrap it with `Buffer` and add it as the `values` blob

### Fields stored in the `TensorArray` metadata
- `kind = "tensor"`
- `base_length`
- `tensor_dtype` (string, e.g. `float32`)
- `tensor_shape` (list)
- `device` (string)
- `values` (blob index)

If `base_length` is 0 or no prototype exists, set `values` to `None`.

## 6. `read_table`: `mmap`-based deserialization
`read_table(path)` opens the file, memory-maps the entire file via `mmap`, reads the header, and reconstructs buffers by slicing views from `memoryview(mm)`.

The key point is that it does not copy buffers into a Python `bytearray`; it creates `Buffer`s as slices of the `memoryview` over the `mmap`.

### File parsing order
1) validate `MAGIC`  
2) read header length (4 bytes, `"<I"`)  
3) read header JSON bytes and `json.loads`  
4) according to `header["buffers"]`, create a `memoryview` over the raw blob region and slice each blob via `Buffer.from_memoryview`

'''python
data_start = mm.tell()
base_view = memoryview(mm)[data_start : data_start + total]

buffers: List[Buffer] = []
for buffer in header["buffers"]:
    start = buffer["offset"]
    end = start + buffer["length"]
    buffers.append(Buffer.from_memoryview(base_view[start:end]))
'''

Because `Buffer` internally holds a `memoryview`, the entire read path becomes `mmap` + `memoryview`, operating close to zero-copy.

## 7. `decode_array`: reconstructing arrays from metadata
`decode_array(inputs)` performs the inverse of `encode_array`.

### Restoring dtype
'''python
data_type = DataType(inputs["dtype"]["kind"])
'''

### Restoring validity
If `validity` exists:

- `validity_buffer = buffers[inputs["validity"]]`
- `validity_length = inputs.get("validity_length", logical_length)`
- `validity = Bitmap(validity_len, validity_buffer)`

Here, `Bitmap` can also be initialized with a `Buffer` in a zero-copy way.

### Restoring indices
If `indices` exists:

- `indices = buffers[inputs["indices"]]`

### Reconstruction rules by kind
`ListArray` and `StringArray` use an `offsets` buffer, 
so `base_length` is computed as `(len(offsets) // 4) - 1`. (`offsets` is an `int32` array, so it is 4-byte aligned.)

- primitive:
  - `values_buf = buffers[inputs["values"]]`
  - `item_size = FMT[data_type][1]`
  - `base_length = len(values_buf) // item_size`
  - `PrimitiveArray(data_type, base_length, values_buf, validity, indices)`

- string:
  - `offsets = buffers[inputs["offsets"]]`
  - `values = buffers[inputs["values"]]`
  - `base_length = (len(offsets) // 4) - 1`
  - `StringArray(offsets, base_length, values, validity, indices)`

- list:
  - `offsets = buffers[inputs["offsets"]]`
  - `child = decode_array(inputs["child"])`
  - `base_length = (len(offsets) // 4) - 1`
  - `ListArray(offsets, base_length, child, validity, indices)`

- struct:
  - `names = inputs["names"]`
  - `children = [decode_array(cm) for cm in inputs["children"]]`
  - `StructArray(names, children, validity)`

- tensor:
  - `decode_tensor_array(inputs, validity, indices)`

## 8. `decode_tensor_array`: reconstructing `TensorArray`
Because `TensorArray` is stored as stacked bytes, reconstruction follows these steps.

An important point is that `torch.frombuffer` can create a tensor view without copying the buffer when possible. 
That means the tensor can be placed on top of the `mmap`-mapped file data (the internal behavior can vary by conditions/environment, but the design intent is zero-copy oriented).

- read `base_length`, `tensor_dtype`, `tensor_shape`, `values_idx`
- obtain the `values` buffer
- create a 1D tensor via `torch.frombuffer(values_buf.data, dtype=scalar_dtype, count=total_elems)`
- view it into a `(base_length, *elem_shape)` block
- build a per-row tensor list and create a `TensorArray`

'''python
base_1d = torch.frombuffer(values_buf.data, dtype=scalar_dtype, count=total_elems)
base_block = base_1d.view(base_length, *elem_shape) if elem_shape else base_1d.view(base_length)

base_tensors: List[torch.Tensor] = [base_block[i] for i in range(base_length)]
return TensorArray(base_tensors, validity, indices)
'''

## 9. Reconstructing `Schema` / `RecordBatch` / `Table`
It creates `Field`s from `schema.fields` in the header and constructs a `Schema`.
Then it iterates over `batches`, reconstructs each batch’s `columns` metadata via `decode_array`, builds `RecordBatch`, and finally returns `Table(batches)`.

## 10. IPC: 3-line summary
- The file consists of `MAGIC` + JSON header + raw buffers (blobs), and the header contains each buffer’s `offset`/`length` and array metadata.
- Write collects blobs such as `values`/`offsets`/`validity`/`indices`, and read re-references blobs via `mmap` + `memoryview` close to zero-copy.
- `TensorArray` stores row tensors as a single raw-bytes block made by `stack`, and read reconstructs it via `torch.frombuffer` to build tensor views on top of the file buffer.