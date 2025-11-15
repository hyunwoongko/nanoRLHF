from dataclasses import dataclass
from typing import Union


@dataclass
class DataType:
    """
    A simple data type class to represent primitive data types.
    """
    name: str

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return f"DataType({self.name!r})"


# Predefined primitive data types
BOOL = DataType("bool")
INT32 = DataType("int32")
INT64 = DataType("int64")
FLOAT32 = DataType("float32")
FLOAT64 = DataType("float64")

# Additional complex data types
STRING = DataType("string")
LIST = DataType("list")
STRUCT = DataType("struct")
TENSOR = DataType("tensor")

# Struct format (little-endian)
#   '<' means little-endian
#   '?' means bool (1 byte)
#   'i' means int32 (4 bytes)
#   'q' means int64 (8 bytes)
#   'f' means float32 (4 bytes)
#   'd' means float64 (8 bytes)
FMT = {
    BOOL: ('<?', 1),
    INT32: ('<i', 4),
    INT64: ('<q', 8),
    FLOAT32: ('<f', 4),
    FLOAT64: ('<d', 8),
}

PrimitiveType = Union[bool, int, float]
INT32_MIN, INT32_MAX = -2_147_483_648, 2_147_483_647

