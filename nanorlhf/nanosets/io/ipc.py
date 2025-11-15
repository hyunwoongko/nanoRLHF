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

_TORCH_DTYPE_TO_STR = {
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

_STR_TO_TORCH_DTYPE = {v: k for k, v in _TORCH_DTYPE_TO_STR.items()}


def write_table(fp, table: Table):
    blobs: List[memoryview] = []

    def add_buf(b: Buffer) -> int:
        i = len(blobs)
        blobs.append(b.data)
        return i

    def dtype_meta(dt: DataType):
        return {"kind": dt.name}

    def encode_tensor_array(arr: TensorArray, meta: dict) -> None:
        base_tensors = arr._tensors
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
        for t in base_tensors:
            if t is not None:
                prototype = t
                break

        if prototype is None:
            meta["values"] = None
            meta["tensor_dtype"] = "float32"
            meta["tensor_shape"] = []
            meta["device"] = "cpu"
            return

        if prototype.device.type != "cpu":
            raise ValueError("TensorArray IPC currently supports only CPU tensors")

        if prototype.dtype not in _TORCH_DTYPE_TO_STR:
            raise ValueError(f"Unsupported torch dtype for TensorArray IPC: {prototype.dtype}")

        scalar_dtype = prototype.dtype
        scalar_name = _TORCH_DTYPE_TO_STR[scalar_dtype]
        elem_shape = list(prototype.shape)
        device = prototype.device

        for t in base_tensors:
            if t is None:
                continue
            if t.dtype != scalar_dtype:
                raise ValueError("All tensors in TensorArray must have the same dtype for IPC")
            if list(t.shape) != elem_shape:
                raise ValueError("All tensors in TensorArray must have the same shape for IPC")
            if t.device != device:
                raise ValueError("All tensors in TensorArray must have the same device for IPC")

        meta["tensor_dtype"] = scalar_name
        meta["tensor_shape"] = elem_shape
        meta["device"] = str(device)

        stacked_list = []
        for t in base_tensors:
            if t is None:
                stacked_list.append(torch.zeros(elem_shape, dtype=scalar_dtype, device=device))
            else:
                stacked_list.append(t.contiguous() if not t.is_contiguous() else t)

        big = torch.stack(stacked_list, dim=0).contiguous()
        raw_bytes = big.numpy().tobytes(order="C")

        buf = Buffer.from_memoryview(memoryview(raw_bytes))
        meta["values"] = add_buf(buf)

    def encode_array(arr: Array):
        meta = {
            "dtype": dtype_meta(arr.dtype),
            "length": arr.length,
        }

        if arr.validity is not None:
            meta["validity"] = add_buf(arr.validity.buffer)
            meta["validity_length"] = len(arr.validity)

        if isinstance(arr, PrimitiveArray):
            meta["kind"] = "primitive"
            meta["values"] = add_buf(arr.values)

        elif isinstance(arr, StringArray):
            meta["kind"] = "string"
            meta["offsets"] = add_buf(arr.offsets)
            meta["values"] = add_buf(arr.values)

        elif isinstance(arr, ListArray):
            meta["kind"] = "list"
            meta["offsets"] = add_buf(arr.offsets)
            meta["child"] = encode_array(arr.child)

        elif isinstance(arr, StructArray):
            meta["kind"] = "struct"
            meta["names"] = arr.field_names
            meta["children"] = [encode_array(ch) for ch in arr.children]

        elif isinstance(arr, TensorArray):
            encode_tensor_array(arr, meta)

        else:
            raise TypeError(f"unsupported array type for IPC: {type(arr).__name__}")

        if arr.indices is not None:
            meta["indices"] = add_buf(arr.indices)

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
    f = open(path, "rb")
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

    try:
        if mm.read(len(MAGIC)) != MAGIC:
            raise ValueError("Invalid file format: missing magic string")

        (len_header,) = struct.unpack("<I", mm.read(4))
        header_bytes = mm.read(len_header)
        header = json.loads(header_bytes.decode("utf-8"))

        total = sum(b["length"] for b in header["buffers"])
        data_start = mm.tell()
        base_view = memoryview(mm)[data_start : data_start + total]

        buf_views: List[Buffer] = []
        for b in header["buffers"]:
            start = b["offset"]
            end = start + b["length"]
            buf_views.append(Buffer.from_memoryview(base_view[start:end]))

        def meta_to_dtype(m):
            return DataType(m["kind"])

        def decode_tensor_array(m, validity, indices):
            base_length = m.get("base_length", m["length"])
            tensor_dtype_name = m["tensor_dtype"]
            tensor_shape = m["tensor_shape"]
            values_idx = m.get("values", None)

            if base_length == 0 or values_idx is None:
                base_tensors: List[torch.Tensor] = []
                return TensorArray(base_tensors, validity, indices)

            if tensor_dtype_name is None:
                raise ValueError("TensorArray IPC metadata missing tensor_dtype")

            if tensor_dtype_name not in _STR_TO_TORCH_DTYPE:
                raise ValueError(f"Unknown tensor dtype in IPC: {tensor_dtype_name}")

            scalar_dtype = _STR_TO_TORCH_DTYPE[tensor_dtype_name]
            elem_shape = list(tensor_shape)

            values_buf = buf_views[values_idx]
            num_elems_per_tensor = 1 if not elem_shape else math.prod(elem_shape)
            total_elems = base_length * num_elems_per_tensor

            base_1d = torch.frombuffer(values_buf.data, dtype=scalar_dtype, count=total_elems)
            base_block = base_1d.view(base_length, *elem_shape) if elem_shape else base_1d.view(base_length)

            base_tensors: List[torch.Tensor] = [base_block[i] for i in range(base_length)]
            return TensorArray(base_tensors, validity, indices)

        def decode_array(m):
            """
            Reconstruct array objects from metadata and buffer indices.
            """
            dt = meta_to_dtype(m["dtype"])
            logical_length = m["length"]

            validity = None
            if "validity" in m:
                validity_buf = buf_views[m["validity"]]
                validity_len = m.get("validity_length", logical_length)
                validity = Bitmap(validity_len, validity_buf)

            indices = None
            if "indices" in m:
                indices = buf_views[m["indices"]]

            kind = m["kind"]

            if kind == "primitive":
                values_buf = buf_views[m["values"]]
                _, item_size = FMT[dt]
                base_length = len(values_buf) // item_size
                return PrimitiveArray(dt, base_length, values_buf, validity, indices)

            if kind == "string":
                offsets = buf_views[m["offsets"]]
                values = buf_views[m["values"]]
                base_length = (len(offsets) // 4) - 1
                return StringArray(offsets, base_length, values, validity, indices)

            if kind == "list":
                offsets = buf_views[m["offsets"]]
                child = decode_array(m["child"])
                base_length = (len(offsets) // 4) - 1
                return ListArray(offsets, base_length, child, validity, indices)

            if kind == "struct":
                names = m["names"]
                children = [decode_array(cm) for cm in m["children"]]
                return StructArray(names, children, validity)

            if kind == "tensor":
                return decode_tensor_array(m, validity, indices)

            raise TypeError(f"unsupported array kind in IPC: {kind!r}")

        fields = tuple(
            Field(
                fld["name"],
                meta_to_dtype(fld["dtype"]),
                fld.get("nullable", True),
            )
            for fld in header["schema"]["fields"]
        )
        schema = Schema(fields)
        batches: List[RecordBatch] = []
        for b in header["batches"]:
            cols = [decode_array(col_meta) for col_meta in b["columns"]]
            batches.append(RecordBatch(schema, cols))

        return Table(batches)

    except Exception:
        mm.close()
        f.close()
        raise
