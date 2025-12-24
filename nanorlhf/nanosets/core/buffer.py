from dataclasses import dataclass


@dataclass
class Buffer:
    """
    A simple buffer class that wraps a `memoryview`, for zero-copy data handling.

    Discussion:
        Q. What is zero-copy?
            Typically, when passing or converting data to another variable,
            Python often copies the data to a new memory location.

            For example:
            >>> a = b"hello world"
            >>> b = a[0:5]  # This creates a copy of `a` in a new memory location.

            In this case, `b` contains the part of data from `a`,
            but different copies exist in memory, so duplicate memory is used.

            But `memoryview` doesn't copy the data, it provides a view of the original data.
            So it shares the same memory location as the original data.

            >>> data = bytearray(b"hello world")
            >>> view = memoryview(data)[0:5]  # No copy!

        Q. Why is this important for Arrow-like implementation?
            Libraries like Arrow aim to minimize unnecessary data copies for speed and memory efficiency.
            By using zero-copy techniques, they can handle large datasets more efficiently.
    """

    data: memoryview

    def __len__(self) -> int:
        """
        Returns the length of the buffer.

        Returns:
            int: The length of the buffer.
        """
        return len(self.data)

    @classmethod
    def from_bytearray(cls, data: bytearray) -> "Buffer":
        """
        Creates a Buffer from a bytearray.

        Args:
            data (bytearray): The bytearray to create the Buffer from.

        Returns:
            Buffer: A new Buffer instance.
        """
        return cls(memoryview(data))

    @classmethod
    def from_memoryview(cls, data: memoryview) -> "Buffer":
        """
        Creates a Buffer from a memoryview.

        Args:
            data (memoryview): The memoryview to create the Buffer from.

        Returns:
            Buffer: A new Buffer instance.
        """
        return cls(data)

    def to_bytearray(self) -> bytearray:
        """
        Converts the Buffer to a bytearray.

        Returns:
            bytearray: The bytearray representation of the Buffer.
        """
        return bytearray(self.data)

    def to_memoryview(self) -> memoryview:
        """
        Converts the Buffer to a memoryview.

        Returns:
            memoryview: The memoryview representation of the Buffer.
        """
        return self.data

    def slice(self, offset: int, length: int) -> "Buffer":
        """
        Slices the Buffer and returns a new Buffer.

        Args:
            offset (int): The starting offset for the slice.
            length (int): The length of the slice.

        Returns:
            Buffer: A new Buffer instance representing the slice.

        Discussion:
            Q. What happens inside memoryview when slicing?
                When you slice a memoryview, it creates a new memoryview object
                that references the same underlying data but with adjusted start and length.
                This means no data is copied; both the original and sliced memoryviews
                point to the same memory location.

                For example:
                >>> data = bytearray(b"hello world")
                >>> view = memoryview(data)
                >>> slice_view = view[0:5]
                >>> print(id(view.obj) == id(slice_view.obj))
                True
        """
        if offset < 0 or length < 0 or offset + length > len(self.data):
            raise ValueError("slice out of bounds")
        return Buffer(self.data[offset:offset + length])
