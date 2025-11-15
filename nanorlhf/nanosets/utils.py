import os
import struct
from operator import index as _index
from typing import Sequence

from nanorlhf.nanosets.core.buffer import Buffer


def normalize_index(i: int, n: int) -> int:
    if n < 0:
        raise ValueError(f"length must be >= 0, got {n}")
    i = _index(i)
    if i < 0:
        i += n
    if i < 0 or i >= n:
        raise IndexError(f"index {i} out of range for length {n}")
    return i


def is_bool_seq(seq) -> bool:
    if isinstance(seq, (str, bytes, bytearray)):
        return False
    try:
        for v in seq:
            if v is None:
                continue
            if not isinstance(v, bool):
                return False
        return True
    except TypeError:
        return False


def unpack_int32(buffer: Buffer, position: int) -> int:
    return struct.unpack_from("<i", buffer.data, position * 4)[0]


def pack_int32(indices: Sequence[int]) -> Buffer:
    byte_array = bytearray(len(indices) * 4)
    offset = 0
    for i in indices:
        struct.pack_into("<i", byte_array, offset, int(i))
        offset += 4
    return Buffer.from_bytearray(byte_array)


def ext(path: str) -> str:
    base = os.path.basename(path)
    if "." not in base:
        return ""
    return base.rsplit(".", 1)[1].lower()
