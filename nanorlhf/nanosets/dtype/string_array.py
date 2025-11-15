from typing import Optional, Union, Sequence, List

from nanorlhf.nanosets.core.bitmap import Bitmap
from nanorlhf.nanosets.core.buffer import Buffer
from nanorlhf.nanosets.dtype.array import Array, ArrayBuilder
from nanorlhf.nanosets.dtype.dtype import STRING
from nanorlhf.nanosets.utils import normalize_index, unpack_int32, pack_int32


class StringArray(Array):
    def __init__(
        self,
        offsets: Buffer,
        length: int,
        values: Buffer,
        validity: Optional[Bitmap] = None,
        indices: Optional[Buffer] = None,
    ):
        if len(offsets) % 4 != 0:
            raise ValueError("offsets buffer size must be a multiple of 4 (int32)")

        physical_length = len(offsets) // 4 - 1
        if physical_length < 0:
            raise ValueError("offsets buffer must contain at least one entry")

        if indices is None:
            logical_length = length
            if logical_length != physical_length:
                raise ValueError(f"length mismatch: base_length={physical_length}, length argument={length}")
        else:
            if len(indices) % 4 != 0:
                raise ValueError("indices buffer size must be a multiple of 4 (int32)")
            logical_length = len(indices) // 4

        super().__init__(STRING, logical_length, values, validity, indices)

        self.offsets = offsets
        self.physical_length = physical_length

    def __getitem__(self, key: Union[int, slice]):
        if isinstance(key, int):
            if self.is_null(key):
                return None

            index = self.base_index(key)
            if not (0 <= index < self.physical_length):
                raise IndexError(f"base index {index} out of range [0, {self.physical_length})")

            start = unpack_int32(self.offsets, index)
            end = unpack_int32(self.offsets, index + 1)

            if start < 0 or end < start or end > len(self.values):
                raise ValueError(
                    f"Invalid string slice range: start={start}, end={end}, values_size={len(self.values)}"
                )

            length = end - start
            if length == 0:
                return ""

            sub_buffer = self.values.slice(start, length)
            return bytes(sub_buffer.data).decode("utf-8")

        if isinstance(key, slice):
            start, stop, step = key.indices(self.length)
            return self.take(range(start, stop, step))

        raise TypeError(f"Invalid index type: {type(key).__name__}")

    def take(self, indices: Sequence[int]) -> "StringArray":
        num_items = len(indices)
        if num_items == 0:
            empty_offsets = pack_int32([0])
            empty_values = Buffer.from_bytearray(bytearray())
            return StringArray(empty_offsets, 0, empty_values, validity=None, indices=None)

        normalized = [normalize_index(i, self.length) for i in indices]
        is_contiguous_slice = all(normalized[k] + 1 == normalized[k + 1] for k in range(num_items - 1))

        if is_contiguous_slice:
            start = normalized[0]
            length = num_items

            if self.is_contiguous():
                base_start = start
                base_end = start + length

                byte_start = unpack_int32(self.offsets, base_start)
                byte_end = unpack_int32(self.offsets, base_end)
                byte_length = byte_end - byte_start

                sub_values = self.values.slice(byte_start, byte_length)

                local_offsets: List[int] = []
                for i in range(base_start, base_end + 1):
                    off = unpack_int32(self.offsets, i)
                    local_offsets.append(off - byte_start)

                sub_offsets = pack_int32(local_offsets)
                sub_validity = self.validity.slice(start, length) if self.validity else None
                return StringArray(
                    offsets=sub_offsets,
                    length=length,
                    values=sub_values,
                    validity=sub_validity,
                    indices=None,
                )

            else:
                index_offset = start * 4
                index_length = length * 4
                sub_indices = self.indices.slice(index_offset, index_length)  # type: ignore[arg-type]
                sub_validity = self.validity.slice(start, length) if self.validity else None
                return StringArray(
                    offsets=self.offsets,
                    length=length,
                    values=self.values,
                    validity=sub_validity,
                    indices=sub_indices,
                )

        base_indices = normalized if self.is_contiguous() else [unpack_int32(self.indices, i) for i in normalized]
        new_indices = pack_int32(base_indices)
        return StringArray(
            offsets=self.offsets,
            length=len(base_indices),
            values=self.values,
            validity=self.validity,
            indices=new_indices,
        )

    def to_list(self) -> List[Optional[str]]:
        output = []
        for i in range(self.length):
            if self.is_null(i):
                output.append(None)
            else:
                output.append(self[i])
        return output

    @classmethod
    def from_list(cls, data: List[Optional[str]]) -> "StringArray":
        builder = StringArrayBuilder()
        for value in data:
            builder.append(value)
        return builder.finish()


class StringArrayBuilder(ArrayBuilder):
    def __init__(self):
        self.offsets: List[int] = [0]
        self.data_bytes = bytearray()
        self.validity: List[int] = []

    def append(self, value: Optional[str]) -> "StringArrayBuilder":
        if value is None:
            self.validity.append(0)
            self.offsets.append(self.offsets[-1])
        else:
            if not isinstance(value, str):
                raise TypeError(f"StringArray expects str or None, got {type(value).__name__}")
            encoded = value.encode("utf-8")
            self.data_bytes.extend(encoded)
            self.validity.append(1)
            self.offsets.append(len(self.data_bytes))
        return self

    def finish(self) -> StringArray:
        num_items = len(self.validity)
        if len(self.offsets) != num_items + 1:
            raise ValueError(
                f"offsets length must be num_items + 1, got offsets={len(self.offsets)}, num_items={num_items}"
            )

        offsets_buffer = pack_int32(self.offsets)
        values_buffer = Buffer.from_bytearray(self.data_bytes)
        validity_bitmap = Bitmap.from_list(self.validity)

        return StringArray(
            offsets=offsets_buffer,
            length=num_items,
            values=values_buffer,
            validity=validity_bitmap,
            indices=None,
        )
