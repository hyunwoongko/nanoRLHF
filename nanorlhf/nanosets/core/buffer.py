from dataclasses import dataclass


@dataclass
class Buffer:
    data: memoryview

    @classmethod
    def from_bytearray(cls, data: bytearray):
        return cls(memoryview(data))

    @classmethod
    def from_memoryview(cls, data: memoryview):
        return cls(data)

    def to_bytearray(self) -> bytearray:
        return bytearray(self.data)

    def to_memoryview(self) -> memoryview:
        return self.data

    def __len__(self) -> int:
        return len(self.data)

    def slice(self, offset: int, length: int) -> "Buffer":
        if offset < 0 or length < 0 or offset + length > len(self.data):
            raise ValueError("slice out of bounds")
        return Buffer(self.data[offset:offset + length])
