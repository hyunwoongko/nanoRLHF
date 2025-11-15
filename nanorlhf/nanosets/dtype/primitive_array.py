import struct
from typing import Optional, Union, Sequence, List

from nanorlhf.nanosets.core.bitmap import Bitmap
from nanorlhf.nanosets.core.buffer import Buffer
from nanorlhf.nanosets.dtype.array import Array, ArrayBuilder
from nanorlhf.nanosets.dtype.dtype import (
    DataType,
    FMT,
    PrimitiveType,
    BOOL,
    FLOAT32,
    FLOAT64,
    INT32,
    INT64,
    INT32_MIN,
    INT32_MAX,
)
from nanorlhf.nanosets.dtype.dtype_inference import infer_primitive_dtype
from nanorlhf.nanosets.utils import normalize_index, unpack_int32, pack_int32


class PrimitiveArray(Array):

    def __init__(
        self,
        dtype: DataType,
        length: int,
        values: Optional[Buffer],
        validity: Optional[Bitmap] = None,
        indices: Optional[Buffer] = None,
    ):
        assert dtype in FMT, f"Unsupported primitive dtype {dtype}"
        self.fmt, self.item_size = FMT[dtype]
        logical_length = (len(indices) // 4) if indices is not None else length

        super().__init__(dtype, logical_length, values, validity, indices)

        if self.is_contiguous():
            expected_length = self.length * self.item_size
            if len(values) != expected_length:
                raise ValueError(f"Values size mismatch: expected {expected_length} bytes, got {len(values)} bytes")
        else:
            if len(self.indices) % 4 != 0:
                raise ValueError("indices buffer size must be a multiple of 4 (int32)")

    def __getitem__(self, key: Union[int, slice]):
        if isinstance(key, int):
            if self.is_null(key):
                return None
            offset = self.base_index(key) * self.item_size
            return struct.unpack_from(self.fmt, self.values.data, offset)[0]

        if isinstance(key, slice):
            start, stop, step = key.indices(self.length)
            return self.take(range(start, stop, step))

    def take(self, indices: Sequence[int]):
        num_items = len(indices)
        if num_items == 0:
            return PrimitiveArray(self.dtype, 0, values=Buffer.from_bytearray(bytearray(0)))

        normalized = [normalize_index(i, self.length) for i in indices]
        is_contiguous_slice = all(normalized[k] + 1 == normalized[k + 1] for k in range(num_items - 1))

        if is_contiguous_slice:
            start = normalized[0]
            length = num_items
            if self.is_contiguous():
                byte_offset = start * self.item_size
                byte_length = length * self.item_size
                sub_values = self.values.slice(byte_offset, byte_length)
                sub_validity = self.validity.slice(start, length) if self.validity else None
                return PrimitiveArray(self.dtype, length, sub_values, sub_validity)
            else:
                index_offset = start * 4
                index_length = length * 4
                sub_indices = self.indices.slice(index_offset, index_length)
                return PrimitiveArray(self.dtype, length, self.values, self.validity, sub_indices)

        base_indices = normalized if self.is_contiguous() else [unpack_int32(self.indices, i) for i in normalized]
        new_indices = pack_int32(base_indices)
        return PrimitiveArray(self.dtype, len(base_indices), self.values, self.validity, new_indices)

    def to_list(self) -> List[Optional[PrimitiveType]]:
        output = []
        for i in range(self.length):
            if self.is_null(i):
                output.append(None)
            else:
                offset = self.base_index(i) * self.item_size
                value = struct.unpack_from(self.fmt, self.values.data, offset)[0]
                output.append(value)
        return output

    @classmethod
    def from_list(cls, data: list, dtype: Optional[DataType] = None) -> "PrimitiveArray":
        target = dtype if dtype is not None else infer_primitive_dtype(data)
        if target not in FMT:
            raise ValueError(f"Unsupported data type for PrimitiveArray: {target}")

        builder = PrimitiveArrayBuilder(target)

        if target is BOOL:
            for v in data:
                if v is None:
                    builder.append(None)
                elif isinstance(v, bool):
                    builder.append(v)
                else:
                    raise TypeError("BOOL dtype expects `bool` or `None`.")

        elif target in (INT32, INT64):
            for v in data:
                if v is None:
                    builder.append(None)
                    continue
                if isinstance(v, bool):
                    builder.append(int(v))
                elif isinstance(v, int):
                    if target is INT32 and not (INT32_MIN <= v <= INT32_MAX):
                        raise OverflowError(f"Value {v} out of int32 range")
                    builder.append(v)
                elif isinstance(v, float):
                    raise TypeError(
                        "Float value provided for integer dtype. Use FLOAT32/FLOAT64 or omit dtype for inference."
                    )
                else:
                    raise TypeError(f"Integer dtype expects int/bool/None, got {type(v).__name__}.")

        elif target in (FLOAT32, FLOAT64):
            for v in data:
                if v is None:
                    builder.append(None)
                elif isinstance(v, (bool, int, float)):
                    builder.append(float(v))
                else:
                    raise TypeError(f"Float dtype expects float/int/bool/None, got {type(v).__name__}.")

        else:
            raise TypeError(f"Unsupported dtype: {target}")

        return builder.finish()


class PrimitiveArrayBuilder(ArrayBuilder):
    def __init__(self, dtype: DataType):
        assert dtype in FMT, f"Unsupported data type: {dtype}"
        self.dtype = dtype
        self.fmt, self.item_size = FMT[dtype]
        self.values = []
        self.validity = []

    def append(self, value: Optional[PrimitiveType]) -> "PrimitiveArrayBuilder":
        if value is None:
            self.validity.append(0)
            self.values.append(False if self.dtype is BOOL else 0)
        else:
            self.validity.append(1)
            self.values.append(value)
        return self

    def finish(self) -> PrimitiveArray:
        num_items = len(self.values)
        raw_buffer = bytearray(num_items * self.item_size)

        offset = 0
        for value in self.values:
            struct.pack_into(self.fmt, raw_buffer, offset, value)
            offset += self.item_size

        buffer = Buffer.from_bytearray(raw_buffer)
        validity = Bitmap.from_list(self.validity)
        return PrimitiveArray(self.dtype, num_items, buffer, validity, indices=None)
