import json
import math
import mmap
import struct
from typing import List

import torch

from nanorlhf.nanosets.core.bitmap import Bitmap
from nanorlhf.nanosets.core.buffer import Buffer
from nanorlhf.nanosets.dtype.array import Array
from nanorlhf.nanosets.dtype.dtype import DataType, FMT
from nanorlhf.nanosets.dtype.list_array import ListArray
from nanorlhf.nanosets.dtype.primitive_array import PrimitiveArray
from nanorlhf.nanosets.dtype.string_array import StringArray
from nanorlhf.nanosets.dtype.struct_array import StructArray
from nanorlhf.nanosets.dtype.tensor_array import TensorArray
from nanorlhf.nanosets.table.field import Field
from nanorlhf.nanosets.table.record_batch import RecordBatch
from nanorlhf.nanosets.table.schema import Schema
from nanorlhf.nanosets.table.table import Table

MAGIC = b"NANO0"

TORCH_DTYPE_TO_STR = {
    torch.float32: "float32",
    torch.float64: "float64",
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.int64: "int64",
    torch.int32: "int32",
    torch.int16: "int16",
    torch.int8: "int8",
    torch.uint8: "uint8",
    torch.bool: "bool",
}

STR_TO_TORCH_DTYPE = {v: k for k, v in TORCH_DTYPE_TO_STR.items()}


def write_table(fp, table: Table):
    """
    Serialize a `Table` into a compact binary IPC format

    Args:
        fp (file-like): writable binary file-like object
        table (Table): Table to serialize

    Notes:
        The file layout is as follows:
        - MAGIC (5 bytes)
        - Header Length (4 bytes)
        - Header (JSON)
        - Buffers (Blobs)

    Examples:
        >>> with open("table.nano", "wb") as f:
        ...    write_table(f, table)

    Discussion:
        Q. What is a magic string?
            A magic string (or "magic number") is a short fixed sequence of bytes
            written at the beginning of a file to uniquely identify its format.
            Many file formats use it (e.g., PNG = b"\\x89PNG", ZIP = b"PK\\x03\\x04").
            It allows quick detection of file type and prevents misinterpretation.
            We use `b"NANO0"` as our magic string, meaning "This is a Nanoset IPC file, version 0".

        Q. Why header in Json?
            The header is stored as a JSON object for human readability and cross-language interoperability,
            while the actual data (buffers) are written as raw binary blobs for performance and memory efficiency.
            The header is very small compared to the data, so the overhead of JSON is negligible.

        Q. What is blob?
            The term ‘blob’ refers to the raw binary data buffers (e.g. values, offsets, validity bitmaps)
            written sequentially after the header.
    """
    blobs: List[memoryview] = []

    def add_buffer(b: Buffer) -> int:
        """
        Add a buffer to the blobs list and return its index.
        """
        length = len(blobs)
        blobs.append(b.data)
        return length

    def dtype_meta(dtype: DataType):
        """
        Serialize DataType into metadata dictionary.
        """
        return {"kind": dtype.name}

    def encode_tensor_array(array: TensorArray, meta: dict) -> None:
        """
        Serialize TensorArray into metadata and buffers.
        """
        base_tensors = array.tensors
        base_length = len(base_tensors)
        meta["kind"] = "tensor"
        meta["base_length"] = base_length

        if base_length == 0:
            meta["values"] = None
            meta["tensor_dtype"] = None
            meta["tensor_shape"] = []
            meta["device"] = "cpu"
            return

        prototype = None
        for tensor in base_tensors:
            if tensor is not None:
                prototype = tensor
                break

        if prototype is None:
            meta["values"] = None
            meta["tensor_dtype"] = "float32"
            meta["tensor_shape"] = []
            meta["device"] = "cpu"
            return

        if prototype.device.type != "cpu":
            raise ValueError("TensorArray IPC currently supports only CPU tensors")

        if prototype.dtype not in TORCH_DTYPE_TO_STR:
            raise ValueError(f"Unsupported torch dtype for TensorArray IPC: {prototype.dtype}")

        scalar_dtype = prototype.dtype
        scalar_name = TORCH_DTYPE_TO_STR[scalar_dtype]
        elem_shape = list(prototype.shape)
        device = prototype.device

        for tensor in base_tensors:
            if tensor is None:
                continue
            if tensor.dtype != scalar_dtype:
                raise ValueError("All tensors in TensorArray must have the same dtype for IPC")
            if list(tensor.shape) != elem_shape:
                raise ValueError("All tensors in TensorArray must have the same shape for IPC")
            if tensor.device != device:
                raise ValueError("All tensors in TensorArray must have the same device for IPC")

        meta["tensor_dtype"] = scalar_name
        meta["tensor_shape"] = elem_shape
        meta["device"] = str(device)

        stacked_list = []
        for tensor in base_tensors:
            if tensor is None:
                stacked_list.append(torch.zeros(elem_shape, dtype=scalar_dtype, device=device))
            else:
                stacked_list.append(tensor.contiguous() if not tensor.is_contiguous() else tensor)

        stacked_tensor = torch.stack(stacked_list, dim=0).contiguous()
        raw_bytes = stacked_tensor.numpy().tobytes(order="C")

        buffer = Buffer.from_memoryview(memoryview(raw_bytes))
        meta["values"] = add_buffer(buffer)

    def encode_array(array: Array):
        """
        Serialize array objects into metadata and buffers.
        """
        meta = {
            "dtype": dtype_meta(array.dtype),
            "length": array.length,
        }

        if array.validity is not None:
            meta["validity"] = add_buffer(array.validity.buffer)
            meta["validity_length"] = len(array.validity)

        if isinstance(array, PrimitiveArray):
            meta["kind"] = "primitive"
            meta["values"] = add_buffer(array.values)

        elif isinstance(array, StringArray):
            meta["kind"] = "string"
            meta["offsets"] = add_buffer(array.offsets)
            meta["values"] = add_buffer(array.values)

        elif isinstance(array, ListArray):
            meta["kind"] = "list"
            meta["offsets"] = add_buffer(array.offsets)
            meta["child"] = encode_array(array.child)

        elif isinstance(array, StructArray):
            meta["kind"] = "struct"
            meta["names"] = array.field_names
            meta["children"] = [encode_array(ch) for ch in array.children]

        elif isinstance(array, TensorArray):
            encode_tensor_array(array, meta)

        else:
            raise TypeError(f"unsupported array type for IPC: {type(array).__name__}")

        if array.indices is not None:
            meta["indices"] = add_buffer(array.indices)

        return meta

    header = {
        "schema": {
            "fields": [
                {"name": f.name, "dtype": dtype_meta(f.dtype), "nullable": f.nullable} for f in table.schema.fields
            ]
        },
        "batches": [
            {
                "length": b.length,
                "columns": [encode_array(arr) for arr in b.columns],
            }
            for b in table.batches
        ],
        "buffers": [],
    }

    offset = 0
    for blob in blobs:
        header["buffers"].append({"offset": offset, "length": len(blob)})
        offset += len(blob)

    header_bytes = json.dumps(header).encode("utf-8")

    fp.write(MAGIC)
    fp.write(struct.pack("<I", len(header_bytes)))
    fp.write(header_bytes)
    for blob in blobs:
        fp.write(blob)


def read_table(path: str) -> Table:
    """
    Memory-map a serialized `Table` from a file written using `write_table()`.

    Args:
        path (str): Path to the `.nano` file.

    Returns:
        Table: A fully reconstructed `Table` backed by memory-mapped buffers.

    Examples:
        >>> table = read_table("table.nano")

    Discussion:
        Q. What is `mmap`?
            Before understanding `mmap`, it helps to know how RAM is structured in an operating system.

            In modern OSes, RAM is conceptually divided into:
              - Kernel space: managed by the OS; used for disk caches, I/O buffers, and device control.
              - User space: used by individual programs; isolated from the kernel for safety.

            Normally, reading a file with standard I/O (e.g., `fp.read()`) follows this path:
                [Disk] → Copy → [Kernel space] → Copy → [User space]

            The first copy loads data from disk into the kernel space,
            and the second copy moves it into the user space.
            This double-copy increases CPU overhead and wastes memory bandwidth.

            `mmap` (memory mapping) removes that second copy.
            User space has its own memory area called the virtual address space (or virtual memory).
            When we use `mmap`, instead of copying data into user space,
            it maps the addresses of kernel space directly into the user virtual address space.

                [Disk] → Copy → [Kernel space] ↔ [User virtual address space]

            So the file data itself remains in the kernel space and user space just has pointers to it.

        Q. What is a page in memory?
            A page is the smallest fixed-size block of memory managed by the operating system’s
            virtual memory system. Most systems use 4 KB per page.

            When the OS copies data from disk into kernel space,
            it doesn't load individual bytes; it loads an entire page (4KB) at a time.
            These pages are stored in the kernel space in a structure called the page cache.

        Q. Is the entire file copied into the kernel page cache at once?
            No. When the user program first reads from a mapped region, a page fault occurs,
            prompting the OS to copy the **only needed page** from the disk into the kernel space.
            This is called **demand paging** or **lazy loading**.

            Before 1st access:
                [User virtual address space] → Page Fault → OS → [Disk] → Copy → [Kernel space]

            After 1st access:
                [User virtual address space] ↔ [Kernel space]

            This can be summarized in the following table:
                +----------------------------------------------------------------------------------------------------+
                |                    Memory Mapping States                     |                                     |
                +------------------+-------------------------------------------+-------------------------------------+
                | Stage            | User Space (Process Memory)               | Kernel Space (Page Cache)           |
                +------------------+-------------------------------------------+-------------------------------------+
                | `mmap()` called  | Space for virtual addresses reserved      | No file data loaded (still on disk) |
                | After 1st access | Address → Kernel page mapping established | File page loaded into page cache    |
                | Later accesses   | Reads from mapped addresses               | File page remains in page cache     |
                +------------------+-------------------------------------------+-------------------------------------+

        Q. How is `mmap` different from `memoryview`?
            Both `mmap` and `memoryview` provide zero-copy access, but at different levels:
            The key difference is where the zero-copy happens, OS-level vs Python-level.

            `mmap`: works at the OS level.
                Maps a page address directly into virtual memory.
                Avoids copying data between kernel and user space.

            `memoryview`: works at the Python level.
                Provides a zero-copy view into existing memory
                (e.g., bytes, NumPy arrays, or `mmap` objects)
                but doesn't handle disk I/O or paging.

            In short:
                - `mmap`: OS-level zero-copy (disk ↔ virtual memory)
                - `memoryview`: Python-level zero-copy (RAM ↔ Python object)
    """
    # Open the file and create a read-only memory map.
    # On some platforms (e.g., Windows), keeping the file handle alive is safer while the map is in use.
    fp = open(path, "rb")
    mm = mmap.mmap(fp.fileno(), 0, access=mmap.ACCESS_READ)

    try:
        if mm.read(len(MAGIC)) != MAGIC:
            raise ValueError("Invalid file format: missing magic string")

        (len_header,) = struct.unpack("<I", mm.read(4))
        header_bytes = mm.read(len_header)
        header = json.loads(header_bytes.decode("utf-8"))

        total = sum(buffer["length"] for buffer in header["buffers"])
        data_start = mm.tell()
        base_view = memoryview(mm)[data_start : data_start + total]

        buffers: List[Buffer] = []
        for buffer in header["buffers"]:
            start = buffer["offset"]
            end = start + buffer["length"]
            buffers.append(Buffer.from_memoryview(base_view[start:end]))

        def meta_to_dtype(inputs):
            """
            Reconstruct DataType from metadata.
            """
            return DataType(inputs["kind"])

        def decode_tensor_array(inputs, validity, indices):
            """
            Reconstruct TensorArray from metadata and buffer indices.

            Discussion:
                Q. How are tensors stored in the IPC format?
                    Tensors are stored as a single contiguous block of raw bytes.
                    The metadata specifies the tensor dtype and shape,
                    allowing reconstruction of individual tensors.

                Q. How to restore individual tensors?
                    There's a method named `tensor.frombuffer` in PyTorch,
                    So we can use the method to create a 1D tensor from the raw bytes,
                    then reshape it into the original tensor shapes.
            """
            base_length = inputs.get("base_length", inputs["length"])
            tensor_dtype_name = inputs["tensor_dtype"]
            tensor_shape = inputs["tensor_shape"]
            values_idx = inputs.get("values", None)

            if base_length == 0 or values_idx is None:
                base_tensors: List[torch.Tensor] = []
                return TensorArray(base_tensors, validity, indices)

            if tensor_dtype_name is None:
                raise ValueError("TensorArray IPC metadata missing tensor_dtype")

            if tensor_dtype_name not in STR_TO_TORCH_DTYPE:
                raise ValueError(f"Unknown tensor dtype in IPC: {tensor_dtype_name}")

            scalar_dtype = STR_TO_TORCH_DTYPE[tensor_dtype_name]
            elem_shape = list(tensor_shape)

            values_buffer = buffers[values_idx]
            num_elems_per_tensor = 1 if not elem_shape else math.prod(elem_shape)
            total_elems = base_length * num_elems_per_tensor

            base_1d = torch.frombuffer(values_buffer.data, dtype=scalar_dtype, count=total_elems)
            base_block = base_1d.view(base_length, *elem_shape) if elem_shape else base_1d.view(base_length)

            base_tensors: List[torch.Tensor] = [base_block[i] for i in range(base_length)]
            return TensorArray(base_tensors, validity, indices)

        def decode_array(inputs):
            """
            Reconstruct array objects from metadata and buffer indices.
            """
            data_type = meta_to_dtype(inputs["dtype"])
            logical_length = inputs["length"]

            validity = None
            if "validity" in inputs:
                validity_buffer = buffers[inputs["validity"]]
                validity_length = inputs.get("validity_length", logical_length)
                validity = Bitmap(validity_length, validity_buffer)

            indices = None
            if "indices" in inputs:
                indices = buffers[inputs["indices"]]

            kind = inputs["kind"]

            if kind == "primitive":
                values_buffer = buffers[inputs["values"]]
                _, item_size = FMT[data_type]
                base_length = len(values_buffer) // item_size
                return PrimitiveArray(data_type, base_length, values_buffer, validity, indices)

            if kind == "string":
                offsets = buffers[inputs["offsets"]]
                values = buffers[inputs["values"]]
                base_length = (len(offsets) // 4) - 1
                return StringArray(offsets, base_length, values, validity, indices)

            if kind == "list":
                offsets = buffers[inputs["offsets"]]
                child = decode_array(inputs["child"])
                base_length = (len(offsets) // 4) - 1
                return ListArray(offsets, base_length, child, validity, indices)

            if kind == "struct":
                names = inputs["names"]
                children = [decode_array(cm) for cm in inputs["children"]]
                return StructArray(names, children, validity)

            if kind == "tensor":
                return decode_tensor_array(inputs, validity, indices)

            raise TypeError(f"unsupported array kind in IPC: {kind!r}")

        fields = tuple(
            Field(
                field["name"],
                meta_to_dtype(field["dtype"]),
                field.get("nullable", True),
            )
            for field in header["schema"]["fields"]
        )
        schema = Schema(fields)
        batches: List[RecordBatch] = []
        for buffer in header["batches"]:
            columns = [decode_array(col_meta) for col_meta in buffer["columns"]]
            batches.append(RecordBatch(schema, columns))

        return Table(batches)

    except Exception:
        mm.close()
        fp.close()
        raise
