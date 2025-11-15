from typing import Optional, Union, Sequence, List, Iterable, Any

from nanorlhf.nanosets.core.bitmap import Bitmap
from nanorlhf.nanosets.core.buffer import Buffer
from nanorlhf.nanosets.dtype.array import Array, ArrayBuilder
from nanorlhf.nanosets.dtype.dtype import LIST
from nanorlhf.nanosets.dtype.dtype_inference import infer_child_builder
from nanorlhf.nanosets.utils import normalize_index, unpack_int32, pack_int32


class ListArray(Array):

    def __init__(
        self,
        offsets: Buffer,
        length: int,
        child: Array,
        validity: Optional[Bitmap] = None,
        indices: Optional[Buffer] = None,
    ):
        if len(offsets) % 4 != 0:
            raise ValueError("offsets buffer size must be a multiple of 4 (int32)")

        base_length = len(offsets) // 4 - 1
        if base_length < 0:
            raise ValueError("offsets buffer must contain at least one entry")

        total_elems = unpack_int32(offsets, base_length)
        if total_elems > len(child):
            raise ValueError(f"offsets refer to {total_elems} child elements, but child length is {len(child)}")

        if indices is None:
            logical_length = length
            if logical_length != base_length:
                raise ValueError(f"length mismatch: base_length={base_length}, length argument={length}")
        else:
            if len(indices) % 4 != 0:
                raise ValueError("indices buffer size must be a multiple of 4 (int32)")
            logical_length = len(indices) // 4

        super().__init__(LIST, logical_length, child.values, validity, indices)

        self.offsets = offsets
        self.child = child
        self.base_length = base_length

    def __getitem__(self, key: Union[int, slice]):
        if isinstance(key, int):
            if self.is_null(key):
                return None

            idx = self.base_index(key)
            if not (0 <= idx < self.base_length):
                raise IndexError(f"base index {idx} out of range [0, {self.base_length})")

            start = unpack_int32(self.offsets, idx)
            end = unpack_int32(self.offsets, idx + 1)

            if start < 0 or end < start or end > len(self.child):
                raise ValueError(f"Invalid child range: start={start}, end={end}, child_length={len(self.child)}")

            if start == end:
                return []

            sub_array = self.child.take(range(start, end))
            return sub_array.to_list()

        if isinstance(key, slice):
            start, stop, step = key.indices(self.length)
            return self.take(range(start, stop, step))

        raise TypeError(f"Invalid index type: {type(key).__name__}")

    def take(self, indices: Sequence[int]) -> "ListArray":
        num_items = len(indices)
        if num_items == 0:
            empty_offsets = pack_int32([0])
            return ListArray(
                offsets=empty_offsets,
                length=0,
                child=self.child,
                validity=None,
                indices=None,
            )

        normalized = [normalize_index(i, self.length) for i in indices]
        is_contiguous_slice = all(normalized[k] + 1 == normalized[k + 1] for k in range(num_items - 1))

        if is_contiguous_slice:
            start = normalized[0]
            length = num_items

            if self.is_contiguous():
                base_start = start
                base_end = start + length

                child_start = unpack_int32(self.offsets, base_start)
                child_end = unpack_int32(self.offsets, base_end)
                new_child = self.child.take(range(child_start, child_end))

                local_offsets: List[int] = []
                for i in range(base_start, base_end + 1):
                    off = unpack_int32(self.offsets, i)
                    local_offsets.append(off - child_start)

                new_offsets = pack_int32(local_offsets)
                new_validity = self.validity.slice(start, length) if self.validity else None
                return ListArray(
                    offsets=new_offsets,
                    child=new_child,
                    length=length,
                    validity=new_validity,
                    indices=None,
                )

            else:
                index_offset = start * 4
                index_length = length * 4

                sub_indices = self.indices.slice(index_offset, index_length)  # type: ignore[arg-type]
                new_validity = self.validity.slice(start, length) if self.validity else None
                return ListArray(
                    offsets=self.offsets,
                    length=length,
                    child=self.child,
                    validity=new_validity,
                    indices=sub_indices,
                )

        base_indices = normalized if self.is_contiguous() else [unpack_int32(self.indices, i) for i in normalized]
        new_indices = pack_int32(base_indices)
        return ListArray(
            offsets=self.offsets,
            length=len(base_indices),
            child=self.child,
            validity=self.validity,
            indices=new_indices,
        )

    def to_list(self) -> List[Optional[list]]:
        out: List[Optional[list]] = []
        for i in range(self.length):
            if self.is_null(i):
                out.append(None)
            else:
                out.append(self[i])
        return out

    @classmethod
    def from_list(cls, data: List[Optional[Iterable[Any]]]) -> "ListArray":
        child_builder = infer_child_builder(data)
        builder = ListArrayBuilder(child_builder)
        for row in data:
            builder.append(row)

        return builder.finish()


class ListArrayBuilder(ArrayBuilder):

    def __init__(self, child_builder: ArrayBuilder):
        self.child_builder = child_builder
        self.offsets: List[int] = [0]
        self.validity: List[int] = []
        self.length: int = 0

    def append(self, value: Optional[Iterable[Any]]) -> "ListArrayBuilder":
        if value is None:
            self.validity.append(0)
            self.offsets.append(self.offsets[-1])
            self.length += 1
            return self

        if isinstance(value, (str, bytes, bytearray)) or not hasattr(value, "__iter__"):
            raise TypeError(
                f"ListArrayBuilder.append expects an iterable (non-string) or None, got {type(value).__name__}"
            )

        self.validity.append(1)
        start_count = self.offsets[-1]
        count = 0
        for elem in value:
            self.child_builder.append(elem)
            count += 1

        self.offsets.append(start_count + count)
        self.length += 1
        return self

    def finish(self) -> ListArray:

        num_items = self.length
        if len(self.validity) != num_items:
            raise ValueError(f"validity length {len(self.validity)} does not match number of items {num_items}")
        if len(self.offsets) != num_items + 1:
            raise ValueError(
                f"offsets length must be num_items + 1, got offsets={len(self.offsets)}, num_items={num_items}"
            )

        offsets_buffer = pack_int32(self.offsets)
        child_array = self.child_builder.finish()
        validity_bitmap = Bitmap.from_list(self.validity)

        return ListArray(
            offsets=offsets_buffer,
            length=num_items,
            child=child_array,
            validity=validity_bitmap,
            indices=None,
        )
