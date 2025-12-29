import math
from typing import Optional, Union

from nanorlhf.nanosets.core.buffer import Buffer


class Bitmap:
    """
    Bitmap is a simple class that stores the validity (nullness) of elements.
    Each bit in the bitmap represents whether the corresponding element is valid or null.

    Args:
        num_bits (int): Number of bits (elements) in the bitmap.
        buffer (Optional[Buffer]): Optional buffer to initialize the bitmap from.

    Notes:
        It stores 1 for valid values and 0 for nulls.
        e.g. [1, None, 1, None] -> 1010 -> 0b00001010 (in binary).

    Examples:
        >>> bm = Bitmap(10)
        >>> bm[0] = True
        >>> print(bm[0])  # True
        >>> bm[1] = False
        >>> print(bm[1])  # False

    Discussion:
        Q. Why do we need to store null (missing) information separately?
            In real-world datasets, some elements may be missing (e.g. None or NaN).
            If nulls are stored directly inside the data array, it mixes different Python object types,
            such as integers and NoneType, breaking the uniform memory layout.

            >>> data = [10, None, 30, 40]

            In Python, this list internally stores object pointers (PyObject*), not raw integers.
            Each element in the list is a pointer to a Python object allocated in a different memory location like:

            | Index | Value | Actual Storage Type            |
            |------:|-------|--------------------------------|
            | 0     | 10    | PyObject* (points to int)      |
            | 1     | None  | PyObject* (points to NoneType) |
            | 2     | 30    | PyObject* (points to int)      |
            | 3     | 40    | PyObject* (points to int)      |

            This scattered memory layout leads to several issues:
            - No SIMD vectorization: The CPU cannot process multiple elements at once.
            - Cache inefficiency: Values are distributed across non-contiguous memory.
            - High overhead: Each access goes through a Python object wrapper.

            By separating values and their validity information,
            the numeric data can be stored as a compact, contiguous memory block (e.g. int32, float64, etc.).
            while the validity is tracked by a lightweight bitmap that uses only 1 bit per element.

            >>> values  = bytearray([10, 0, 30, 40])   # int32 array
            >>> bitmap  = bytearray([0b00001011])      # 1 means valid, 0 means null

            This design provides three major advantages:
            1. Speed: Contiguous memory enables SIMD and vectorized operations.
            2. Memory efficiency: Validity information requires only 1 bit per value.
            3. Interoperability: The structure is language-agnostic and can be shared
               across C, Python, Java, Rust, and others without serialization overhead.

            In short, separating null validity from data values allows Arrow-style arrays
            to achieve both high performance and flexibility.

        Q. What is SIMD vectorization, and why does it matter?
            SIMD stands for Single Instruction, Multiple Data.
            It allows the CPU to perform the same operation on multiple elements simultaneously.
            For example, instead of executing four separate additions:
            >>> # Scalar operations (no SIMD)
            >>> [1+1, 2+2, 3+3, 4+4]  # processed one by one

            With SIMD, the CPU can execute them all at once using vectorized instructions:
            >>> # Vectorized operations (with SIMD)
            >>> [1, 2, 3, 4] + [1, 2, 3, 4]  # computed together internally

            SIMD works only when data is stored contiguously in memory.
            If data is scattered (like Python objects), the CPU cannot load or process multiple values efficiently.
            This is why libraries like Arrow require contiguous memory layouts.
            They allow low-level vectorized operations that fully utilize CPU hardware capabilities.

        Q. What is cache inefficiency, and why does it matter?
            Modern CPUs are much faster than main memory, so they use a small, high-speed memory
            called the CPU cache to temporarily store recently accessed data.

            When data is stored contiguously (for example, in a NumPy or Arrow array),
            the CPU can load an entire sequence of elements into the cache in a single loading operation.
            This allows fast sequential processing because nearby elements are already cached.
            >>> # Contiguous memory (cache-friendly)
            >>> [10, 20, 30, 40]  # loaded into cache together

            However, when data is scattered in memory (like a list of PyObject* pointers),
            each element may reside in a completely different memory location.
            The CPU must repeatedly fetch data from main memory instead of reusing cached data,
            causing frequent "cache misses" that slow down processing dramatically.
            >>> # Scattered memory (cache-unfriendly)
            >>> [10, None, 30, 40]  # each object stored separately

            This is why Arrow stores all values in contiguous buffers
            ensuring that sequential reads fully benefit from CPU caching and memory prefetching.
        """

    def __init__(self, num_bits: int, buffer: Optional[Buffer] = None, bit_offset: int = 0):
        if num_bits < 0:
            raise ValueError(f"Number of bits must be non-negative, got {num_bits}")
        if not (0 <= bit_offset < 8):
            raise ValueError(f"Bit offset must be in [0, 8), got {bit_offset}")

        # 1 byte is 8 bites, and `ceil` computes the number of bytes needed to store the bits
        self.num_bytes = 0 if num_bits == 0 else math.ceil((bit_offset + num_bits) / 8)
        self.num_bits = num_bits
        self.bit_offset = bit_offset

        if buffer is None:
            self.buffer: Buffer = Buffer.from_bytearray(bytearray(self.num_bytes))
        else:
            assert len(buffer) == self.num_bytes, (
                f"Buffer length {len(buffer)} does not match required size {self.num_bytes}"
            )
            # for zero-copy initialization
            self.buffer: Buffer = buffer

    def check_bound(self, i: int):
        """
        Check if the given index is within the valid range of the bitmap.

        Args:
            i (int): Index to check.
        """
        if not (0 <= i < self.num_bits):
            raise IndexError(f"Bitmap index {i} out of range [0, {self.num_bits})")

    def absolute_bit(self, i: int) -> int:
        """
        Calculate the absolute bit position in the underlying buffer.

        Args:
            i (int): Index in the bitmap.

        Returns:
            int: Absolute bit position in the buffer.

        Discussion:
            Q. What is absolute bit position?
                The absolute bit position accounts for any initial bit offset in the bitmap.
                It is calculated by adding the bitmap's bit_offset to the given index i.

            Q. What is bit offset and why does it matter?
                Bit offset indicates how many bits to skip at the start of the bitmap.
                This is important when the bitmap does not start exactly at a byte boundary.
                For example, if bit_offset is 3, the first 3 bits of the first byte are unused,
                and the valid bits start from the 4th bit. This is very useful when slicing bitmaps.

                Example:
                - Original bitmap (with bit offset 3):
                  Byte 0: [x x x 1 0 1 0 1]  (x = unused bits)
                  Byte 1: [1 1 0 0 1 0 0 0]

                - The first valid bit corresponds to index 0 in the bitmap,
                  which is actually the 4th bit of Byte 0 (absolute bit position 3).
        """
        return self.bit_offset + i

    def __len__(self) -> int:
        """
        Get the number of bits in the bitmap.

        Returns:
            int: Number of bits.
        """
        return self.num_bits

    def __getitem__(self, key: int):
        """
        Check if the i-th element is valid.

        Args:
            key (int): index of the element

        Returns:
            bool: True if valid, False if invalid (null)

        Notes:
            - The method runs in O(1) time, regardless of the bitmap size.
            - Using bitmaps avoids storing one boolean per element (which would use 1 byte each),
              saving up to 8× memory.

        Discussion:
            Q. How does this function check the bit?
                It locates which byte and bit represent the given element’s validity,
                then uses a bitwise AND (&) to test whether the target bit is set.

                >>> byte, bit = divmod(key, 8)
                >>> mask = (1 << bit)
                >>> (self.buffer.data[byte] & mask) != 0

                If the result is nonzero, the bit is 1 (valid).
                If it is zero, the bit is 0 (null).

            Q. Example walkthrough
                Suppose the byte is 00100100 (binary):

                - For bit=2:
                    mask = 00000100
                    00100100 & 00000100 = 00000100  -> nonzero -> True

                - For bit=1:
                    mask = 00000010
                    00100100 & 00000010 = 00000000  -> zero -> False
        """
        self.check_bound(key)
        abs_bit = self.absolute_bit(key)

        byte, bit = divmod(abs_bit, 8)
        b = self.buffer.data[byte]
        mask = (1 << bit)
        check = b & mask
        return check != 0

    def __setitem__(self, key: int, value: Union[int, bool]):
        """
        Set the validity of the i-th element.

        Args:
            key (int): index of the element
            value (Union[int, bool]): True for valid, False for invalid (null)

        Discussion:
            Q. How does this method work internally?
                Each bit in the bitmap represents the validity of one element.
                A bit value of 1 means valid, and 0 means null.
                The bitmap is stored as a NumPy array of bytes (uint8),
                so each byte holds the validity of 8 elements.

                >>> i = 10
                >>> byte, bit = divmod(i, 8)
                Here, `byte` is the index of the byte, and `bit` is the position inside that byte.

                For example:
                    i = 0  -> (byte=0, bit=0)
                    i = 7  -> (byte=0, bit=7)
                    i = 8  -> (byte=1, bit=0)
                    i = 9  -> (byte=1, bit=1)

            Q. How are bitwise operations used here?
                - To set a bit to 1 (mark valid):
                    self.buffer.data[byte] |= (1 << bit)
                    The OR operator (|) turns that bit to 1 while keeping other bits unchanged.

                - To clear a bit to 0 (mark null):
                    self.buffer.data &= ~(1 << bit) & 0xFF
                    The NOT (~) flips the mask bits (so only the target bit becomes 0),
                    and AND (&) keeps all other bits intact.

                - Why `& 0xFF` when clearing a bit?
                    In Python, integers can grow beyond 8 bits, so when we use the NOT operator (~),
                    it produces a negative number with an infinite series of leading 1s in binary.
                    This can lead to unintended consequences when performing bitwise operations on bytes.

                    By applying `& 0xFF`, we ensure that only the lowest 8 bits are kept,
                    effectively masking out any higher bits that could interfere with our byte-level operation.
                    This guarantees that the result remains within the valid range of a byte (0-255).

            Q. Example walkthrough
                Suppose we have byte = 00100100 (binary) and bit = 3.

                mask = (1 << 3)  # 00001000 (binary)

                - When setting 3rd bit valid (True):
                    00100100
                  | 00001000
                  = 00101100  (bit 3 set to 1)

                - When setting 3rd bit invalid (False):
                    00101100
                  & 11110111 (which is ~00001000)
                  = 00100100 (bit 3 cleared to 0)
                  & 11111111 (=0xFF, to keep it within a byte)
                  = 00100100
        """
        self.check_bound(key)
        assert isinstance(value, (int, bool)), f"Bitmap value must be int or bool, got {type(value)}"

        abs_bit = self.absolute_bit(key)
        byte, bit = divmod(abs_bit, 8)
        b = self.buffer.data[byte]

        if bool(value):
            mask = (1 << bit)
            packed = b | mask
        else:
            mask = ~(1 << bit) & 0xFF
            packed = b & mask

        self.buffer.data[byte] = packed

    @classmethod
    def from_list(cls, bits: list[int]) -> Optional["Bitmap"]:
        """
        Build a bitmap from a 0/1 list.
        This method packs and pads the bits into bytes.

        Args:
            bits: 0/1 per element (1 = valid, 0 = null).

        Returns:
            Optional[Bitmap]: Bitmap instance or None if all bits are 1 (no nulls).
        """
        if not bits:
            return None
        if 0 not in bits:
            return None
        bitmap = cls(len(bits))
        for i, v in enumerate(bits):
            bitmap[i] = v
        return bitmap

    def slice(self, offset: int, length: int) -> "Bitmap":
        """
        Slice the bitmap from the given offset with the specified length.

        Args:
            offset (int): Starting index of the slice.
            length (int): Number of bits in the slice.

        Returns:
            Bitmap: A new Bitmap instance representing the sliced portion.

        Discussion:
            Q. How does slicing work with bit offsets?
                When slicing a bitmap, we need to account for both the byte and bit offsets.
                The absolute bit position is calculated by adding the bitmap's bit_offset to the given offset.
                This determines where the slice starts in the underlying buffer.

                For example, if the original bitmap has a bit_offset of 3, and we want to slice
                starting from offset 5, the absolute starting bit position is 3 + 5 = 8.
                This means the slice starts at the beginning of the second byte.
        """
        if offset < 0 or length < 0:
            raise ValueError("Offset and length must be non-negative")
        if offset + length > self.num_bits:
            raise ValueError(f"slice [{offset}:{offset + length}) out of range for num_bits={self.num_bits}")
        if length == 0:
            return Bitmap(0)

        abs_bit_position = self.bit_offset + offset
        byte, bit = divmod(abs_bit_position, 8)

        needed_bytes = math.ceil((bit + length) / 8)
        sliced_buffer = self.buffer.slice(byte, needed_bytes)
        return Bitmap(length, sliced_buffer, bit)
