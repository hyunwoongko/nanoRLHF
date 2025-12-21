from abc import ABC, abstractmethod
from typing import Optional, Sequence

from nanorlhf.nanosets.core.bitmap import Bitmap
from nanorlhf.nanosets.core.buffer import Buffer
from nanorlhf.nanosets.dtype.dtype import DataType
from nanorlhf.nanosets.utils import normalize_index, unpack_int32


class Array(ABC):

    def __init__(
        self,
        dtype: DataType,
        length: int,
        values: Optional[Buffer] = None,
        validity: Optional[Bitmap] = None,
        indices: Optional[Buffer] = None,
    ):
        self.dtype = dtype
        self.length = length
        self.values = values
        self.validity = validity
        self.indices = indices

    def is_contiguous(self):
        return self.indices is None

    def is_null(self, idx: int) -> bool:
        if self.validity is None:
            return False

        normalized_idx = normalize_index(idx, self.length)
        if self.is_contiguous():
            return not self.validity[normalized_idx]

        base_idx = unpack_int32(self.indices, normalized_idx)
        return not self.validity[base_idx]

    def base_index(self, idx: int) -> int:
        normalized_idx = normalize_index(idx, self.length)
        if self.is_contiguous():
            return normalized_idx
        return unpack_int32(self.indices, normalized_idx)

    def __len__(self):
        return self.length

    def __setitem__(self, key, value):
        raise RuntimeError(f"`{self.__class__.__name__}` is immutable")

    @abstractmethod
    def __getitem__(self, item):
        raise NotImplementedError

    @abstractmethod
    def take(self, indices: Sequence):
        raise NotImplementedError

    @abstractmethod
    def to_list(self):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_list(cls, data: list):
        raise NotImplementedError


class ArrayBuilder(ABC):

    @abstractmethod
    def append(self, value):
        raise NotImplementedError

    @abstractmethod
    def finish(self) -> Array:
        raise NotImplementedError
