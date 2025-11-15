from typing import List, Optional, Sequence

import torch

from nanorlhf.nanosets.core.bitmap import Bitmap
from nanorlhf.nanosets.core.buffer import Buffer
from nanorlhf.nanosets.dtype.array import Array, ArrayBuilder
from nanorlhf.nanosets.dtype.dtype import TENSOR
from nanorlhf.nanosets.utils import normalize_index, unpack_int32, pack_int32


class TensorArray(Array):
    def __init__(
        self,
        tensors: List[Optional[torch.Tensor]],
        validity: Optional[Bitmap] = None,
        indices: Optional[Buffer] = None,
    ):
        base_length = len(tensors)
        logical_length = (len(indices) // 4) if indices is not None else base_length
        super().__init__(TENSOR, logical_length, values=None, validity=validity, indices=indices)

        self._tensors: List[Optional[torch.Tensor]] = tensors
        self.base_length = base_length

        if validity is not None and len(validity) != base_length:
            raise ValueError(
                f"Validity bitmap length ({len(validity)}) does not match number of base rows ({base_length})"
            )

        if not self.is_contiguous():
            if len(indices) % 4 != 0:
                raise ValueError("indices buffer size must be a multiple of 4 (int32)")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, key):
        if isinstance(key, int):
            i = normalize_index(key, self.length)
            if self.is_null(i):
                return None
            base_i = self.base_index(i)
            if not (0 <= base_i < self.base_length):
                raise IndexError(f"base index {base_i} out of range [0, {self.base_length})")
            return self._tensors[base_i]

        if isinstance(key, slice):
            start, stop, step = key.indices(self.length)
            return self.take(range(start, stop, step))

        raise TypeError(f"Invalid index type for TensorArray: {type(key).__name__}")

    def take(self, indices: Sequence[int]) -> "TensorArray":
        num_items = len(indices)
        if num_items == 0:
            return TensorArray([], None, None)

        normalized = [normalize_index(i, self.length) for i in indices]
        is_contiguous_slice = all(normalized[k] + 1 == normalized[k + 1] for k in range(num_items - 1))

        if is_contiguous_slice:
            start = normalized[0]
            length = num_items

            if self.is_contiguous():
                base_start = start
                base_end = start + length

                sub_tensors = self._tensors[base_start:base_end]
                sub_validity = self.validity.slice(base_start, length) if self.validity is not None else None
                return TensorArray(sub_tensors, sub_validity, None)
            else:
                index_offset = start * 4
                index_length = length * 4
                sub_indices = self.indices.slice(index_offset, index_length)
                return TensorArray(self._tensors, self.validity, sub_indices)

        if self.is_contiguous():
            base_indices = normalized
        else:
            base_indices = [unpack_int32(self.indices, i) for i in normalized]

        new_indices = pack_int32(base_indices)
        return TensorArray(self._tensors, self.validity, new_indices)

    def to_list(self) -> List[Optional[torch.Tensor]]:
        output: List[Optional[torch.Tensor]] = []
        for i in range(self.length):
            if self.is_null(i):
                output.append(None)
            else:
                output.append(self[i])
        return output

    @classmethod
    def from_list(cls, data: List[Optional[torch.Tensor]]) -> "TensorArray":
        builder = TensorArrayBuilder()
        for x in data:
            builder.append(x)
        return builder.finish()


class TensorArrayBuilder(ArrayBuilder):
    def __init__(self):
        self._tensors: List[Optional[torch.Tensor]] = []
        self.validity: List[int] = []
        self._prototype: Optional[torch.Tensor] = None

    def _check_and_set_prototype(self, value: torch.Tensor):
        if self._prototype is None:
            self._prototype = value
            return

        if value.dtype != self._prototype.dtype:
            raise TypeError(f"Inconsistent tensor dtype: expected {self._prototype.dtype}, got {value.dtype}")
        if value.device != self._prototype.device:
            raise TypeError(f"Inconsistent tensor device: expected {self._prototype.device}, got {value.device}")
        if value.shape != self._prototype.shape:
            raise TypeError(
                f"Inconsistent tensor shape: expected {tuple(self._prototype.shape)}, got {tuple(value.shape)}"
            )

    def append(self, value: Optional[torch.Tensor]) -> "TensorArrayBuilder":
        if value is None:
            self.validity.append(0)
            self._tensors.append(None)
            return self

        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"TensorArrayBuilder.append expects torch.Tensor or None, got {type(value).__name__}"
            )

        self._check_and_set_prototype(value)
        self.validity.append(1)
        self._tensors.append(value)
        return self

    def finish(self) -> TensorArray:
        if len(self._tensors) != len(self.validity):
            raise ValueError(
                f"TensorArrayBuilder internal length mismatch: "
                f"tensors={len(self._tensors)}, validity={len(self.validity)}"
            )

        validity_bitmap = Bitmap.from_list(self.validity)
        return TensorArray(self._tensors, validity_bitmap, None)
