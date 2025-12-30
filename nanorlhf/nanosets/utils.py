import os
import struct
from operator import index as _index
from typing import Sequence

from nanorlhf.nanosets.core.buffer import Buffer

DEFAULT_BATCH_SIZE = 1000


def normalize_index(i: int, n: int) -> int:
    """
    Normalize a possibly negative index `i` for a sequence of length `n`.

    Args:
        i (int): The index to normalize. Can be negative.
        n (int): The length of the sequence.

    Returns:
        int: The normalized index, guaranteed to be in the range [0, n).

    Examples:
        >>> normalize_index(2, 5)
        2
        >>> normalize_index(-1, 5)
        4
        >>> normalize_index(-5, 5)
        0
    """
    if n < 0:
        raise ValueError(f"length must be >= 0, got {n}")
    i = _index(i)
    if i < 0:
        i += n
    if i < 0 or i >= n:
        raise IndexError(f"index {i} out of range for length {n}")
    return i


def unpack_int32(buffer: Buffer, position: int) -> int:
    """
    Unpack a little-endian int32 from the given Buffer at the specified position.

    Args:
        buffer (Buffer): The Buffer containing the data.
        position (int): The index of the int32 to unpack.

    Returns:
        int: The unpacked int32 value.

    Examples:
        >>> from nanorlhf.nanosets.core.buffer import Buffer
        >>> byte_array = bytearray(8)
        >>> struct.pack_into("<2i", byte_array, 0, 42, -7)
        >>> buffer = Buffer.from_bytearray(byte_array)
        >>> unpack_int32(buffer, 0)
        42
        >>> unpack_int32(buffer, 1)
        -7
    """
    return struct.unpack_from("<i", buffer.data, position * 4)[0]


def pack_int32(indices: Sequence[int]) -> Buffer:
    """
    Pack a sequence of integers into a Buffer as little-endian int32.

    Args:
        indices (Sequence[int]): A sequence of integers to pack.

    Returns:
        Buffer: A Buffer containing the packed int32 values.

    Examples:
        >>> from nanorlhf.nanosets.core.buffer import Buffer
        >>> packed_buffer = pack_int32([1, 2, 3, 4])
        >>> list(struct.unpack("<4i", packed_buffer.data))
        [1, 2, 3, 4]
    """
    byte_array = bytearray(len(indices) * 4)
    offset = 0
    for idx in indices:
        struct.pack_into("<i", byte_array, offset, int(idx))
        offset += 4
    return Buffer.from_bytearray(byte_array)


def ext(path: str) -> str:
    """
    Get the file extension from a given file path.

    Args:
        path (str): The file path.

    Returns:
        str: The file extension in lowercase without the leading dot.
             Returns an empty string if there is no extension.
    """
    base = os.path.basename(path)
    if "." not in base:
        return ""
    return base.rsplit(".", 1)[1].lower()
