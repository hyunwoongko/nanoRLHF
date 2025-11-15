import math
from typing import Optional, Union

from nanorlhf.nanosets.core.buffer import Buffer


class Bitmap:
    def __init__(self, num_bits: int, buffer: Optional[Buffer] = None, bit_offset: int = 0):
        if num_bits < 0:
            raise ValueError(f"Number of bits must be non-negative, got {num_bits}")
        if not (0 <= bit_offset < 8):
            raise ValueError(f"Bit offset must be in [0, 8), got {bit_offset}")

        self.num_bits = num_bits
        self.num_bytes = 0 if num_bits == 0 else math.ceil((bit_offset + num_bits) / 8)
        self.bit_offset = bit_offset

        if buffer is None:
            self.buffer = Buffer.from_bytearray(bytearray(self.num_bytes))
        else:
            assert len(buffer) == self.num_bytes, (
                f"Buffer length {len(buffer)} does not match required size {self.num_bytes}"
            )
            self.buffer = buffer  # for zero-copy initialization

    def _check_bound(self, i: int) -> None:
        if not (0 <= i < self.num_bits):
            raise IndexError(f"Bitmap index {i} out of range [0, {self.num_bits})")

    def _absolute_bit(self, i: int) -> int:
        return self.bit_offset + i

    def __len__(self):
        return self.num_bits

    def __getitem__(self, key: int):
        self._check_bound(key)
        abs_bit = self._absolute_bit(key)
        byte, bit = divmod(abs_bit, 8)
        b = self.buffer.data[byte]
        check = b & (1 << bit)
        return check != 0

    def __setitem__(self, key: int, value: Union[int, bool]):
        self._check_bound(key)
        assert isinstance(value, (int, bool)), f"Bitmap value must be int or bool, got {type(value)}"

        abs_bit = self._absolute_bit(key)
        byte, bit = divmod(abs_bit, 8)
        b = self.buffer.data[byte]
        packed = (b | (1 << bit)) if bool(value) else (b & ~(1 << bit) & 0xFF)
        self.buffer.data[byte] = packed

    @classmethod
    def from_list(cls, bits: list[int]) -> Optional["Bitmap"]:
        if not bits:
            return None
        if 0 not in bits:
            return None
        bitmap = cls(len(bits))
        for i, v in enumerate(bits):
            bitmap[i] = v
        return bitmap

    def slice(self, offset: int, length: int):
        if offset < 0 or length < 0:
            raise ValueError("Offset and length must be non-negative")
        if offset + length > self.num_bits:
            raise ValueError(f"slice [{offset}:{offset+length}) out of range for num_bits={self.num_bits}")
        if length == 0:
            return Bitmap(0)

        abs_offset_bit = self.bit_offset + offset
        start_byte = abs_offset_bit // 8
        new_bit_offset = abs_offset_bit % 8

        needed_bytes = math.ceil((new_bit_offset + length) / 8)
        sliced_buffer = self.buffer.slice(start_byte, needed_bytes)
        return Bitmap(length, sliced_buffer, new_bit_offset)
