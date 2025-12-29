import struct
from typing import Optional, Union, Sequence, List

from nanorlhf.nanosets.core.bitmap import Bitmap
from nanorlhf.nanosets.core.buffer import Buffer
from nanorlhf.nanosets.dtype.array import Array, ArrayBuilder
from nanorlhf.nanosets.dtype.dtype import (
    DataType,
    FMT,
    PrimitiveType,
    BOOL,
    FLOAT32,
    FLOAT64,
    INT32,
    INT64,
    INT32_MIN,
    INT32_MAX,
)
from nanorlhf.nanosets.dtype.dtype_inference import infer_primitive_dtype
from nanorlhf.nanosets.utils import normalize_index, unpack_int32, pack_int32


class PrimitiveArray(Array):
    """
    An Arrow-style primitive array backed by a contiguous binary buffer, optionally accompanied by
    a validity bitmap and an indices buffer for zero-copy views.

    This class stores primitive values (e.g., int32/int64/float32/float64/bool) in a `Buffer`
    as a tightly packed byte sequence. Missingness (null) is tracked separately by `Bitmap`,
    and view semantics (non-contiguous indexing) are achieved through an optional `indices` buffer.

    Attributes:
        fmt (str): A `struct` format string for the array dtype (looked up from `FMT`).
        item_size (int): Number of bytes per element for the dtype (looked up from `FMT`).

    Notes:
        - Values are interpreted through `struct` using `fmt`, and each element occupies `item_size` bytes.
        - If `indices` is present, this array is a non-contiguous view. Logical indices map to base indices
          in the values buffer through the int32 `indices` buffer.
        - This array intentionally does not implement `__setitem__` to keep an immutable, Arrow-like usage model.

    Examples:
        >>> arr = PrimitiveArray.from_list([1, None, 3], dtype=INT32)
        >>> arr[0]
        1
        >>> arr[1] is None
        True
        >>> arr.take([2, 0]).to_list()
        [3, 1]

    Discussion:
        Q. Why store values as packed bytes instead of Python objects?
            Python lists store PyObject* pointers, which leads to scattered memory access.
            Packing primitives into a contiguous byte buffer enables cache-friendly access and easier
            interoperability with low-level systems.

        Q. Why separate validity (nullness) from values?
            Nullness is orthogonal metadata. A bitmap can represent validity using 1 bit per element,
            keeping the values buffer purely numeric and contiguous.

        Q. What is the purpose of the indices buffer?
            It enables view semantics without copying values. The array can reference arbitrary positions
            in the underlying values buffer by mapping logical indices -> base indices.

        Q. Why does this class omit __setitem__?
            Arrow-style arrays are typically treated as immutable. If multiple views share the same buffers,
            in-place mutation can invalidate other views and complicate concurrency and reasoning. By omitting
            __setitem__, we encourage creating new arrays (or new views) instead of mutating existing ones.
    """

    def __init__(
        self,
        dtype: DataType,
        length: int,
        values: Optional[Buffer],
        validity: Optional[Bitmap] = None,
        indices: Optional[Buffer] = None,
    ):
        """
        Initialize a PrimitiveArray.

        Args:
            dtype (DataType): Primitive dtype. Must exist in `FMT`.
            length (int): Length of the array when `indices` is None.
            values (Optional[Buffer]): Packed values buffer.
            validity (Optional[Bitmap]): Optional validity bitmap (1 = valid, 0 = null).
            indices (Optional[Buffer]): Optional int32 buffer mapping logical indices -> base indices.

        Discussion:
            Q. How is the logical length determined?
                If `indices` is present, the logical length is `len(indices) // 4`, because `indices` stores
                int32 entries (4 bytes each). Otherwise the logical length is the provided `length`.

            Q. Why does a contiguous array require values size == length * item_size?
                In the contiguous case, every logical element corresponds to a fixed-size slot in `values`.
                If the buffer size mismatches, indexing would read out of bounds or interpret unrelated bytes.

            Q. Why is the values-size check skipped for non-contiguous views?
                A non-contiguous array is typically a view over a larger base buffer. The values buffer can be
                larger than the logical slice because the view references only some positions.

            Q. Why must indices be a multiple of 4 bytes?
                Indices are stored as int32 values, so the underlying bytes must be divisible by 4 to form a
                valid sequence of int32 entries.
        """
        assert dtype in FMT, f"Unsupported primitive dtype {dtype}"
        self.fmt, self.item_size = FMT[dtype]
        logical_length = (len(indices) // 4) if indices is not None else length

        super().__init__(dtype, logical_length, values, validity, indices)

        if self.is_contiguous():
            expected_length = self.length * self.item_size
            if len(values) != expected_length:
                raise ValueError(f"Values size mismatch: expected {expected_length} bytes, got {len(values)} bytes")
        else:
            if len(self.indices) % 4 != 0:
                raise ValueError("indices buffer size must be a multiple of 4 (int32)")

    def __getitem__(self, key: Union[int, slice]):
        """
        Read an element (int index) or return a selection (slice).

        Args:
            key (Union[int, slice]):
                - int: returns a single element (Python scalar) or None if null.
                - slice: returns a new PrimitiveArray corresponding to the slice, implemented via `take`.

        Returns:
            Union[Optional[PrimitiveType], "PrimitiveArray"]:
                - If key is int: returns a primitive Python value or None.
                - If key is slice: returns a PrimitiveArray view/selection.

        Discussion:
            Q. What happens when key is an integer?
                - If `self.is_null(key)` is True, return None.
                - Otherwise compute the base index (position in the values buffer) as:
                    base = self.base_index(key)
                - Convert it to a byte offset:
                    offset = base * self.item_size
                - Read the value using:
                    struct.unpack_from(self.fmt, self.values.data, offset)[0]

            Q. Why use base_index instead of key directly?
                If the array is contiguous, base_index(key) == key (after normalization).
                If the array is a view (indices exists), base_index(key) reads indices[key] to find where the
                actual value lives in the shared values buffer.

            Q. What happens when key is a slice?
                The slice is normalized via `key.indices(self.length)` and then delegated to:
                    self.take(range(start, stop, step))
                This makes slicing and take share the same selection logic.

            Q. Why use struct.unpack_from instead of slicing bytes?
                `unpack_from` reads directly from the underlying buffer at an offset without creating an
                intermediate slice object, which avoids extra allocations.
        """
        if isinstance(key, int):
            if self.is_null(key):
                return None
            offset = self.base_index(key) * self.item_size
            # why indexing [0]? -> unpack_from returns a tuple of (value1, value2, ...)
            return struct.unpack_from(self.fmt, self.values.data, offset)[0]

        if isinstance(key, slice):
            start, stop, step = key.indices(self.length)
            return self.take(range(start, stop, step))

    def take(self, indices: Sequence[int]):
        """
        Select elements by logical indices and return a new PrimitiveArray.

        Args:
            indices (Sequence[int]):
                - A sequence of logical indices to take.
                - Negative indices are allowed and are normalized.

        Returns:
            PrimitiveArray: A new PrimitiveArray representing the selected elements.

        Notes:
            - This method aims to avoid copying the values buffer.
            - Contiguous logical selections may become buffer slices (zero-copy).
            - Arbitrary selections become indices-based views (values shared).

        Discussion:
            Q. What are "logical indices" in this method?
                The indices provided to `take` are interpreted in the coordinate system of this array:
                i.e., 0..self.length-1. If this array is itself a view, these logical indices are first mapped
                to base indices into the shared values buffer.

            Q. Why normalize indices first?
                Python-style negative indexing (e.g., -1 means last) is convenient. Normalization ensures every
                index falls into [0, self.length) before further logic runs.

                >>> normalized = [normalize_index(i, self.length) for i in indices]

            Q. How does it detect a contiguous logical slice?
                It checks whether normalized indices are consecutive:
                    normalized[k+1] == normalized[k] + 1 for all k.
                If so, the selection can often be represented more compactly as a slice-like view.

            Q. What happens when the selection is a contiguous logical slice and the array is contiguous?
                In that case the selected region is physically contiguous in the values buffer, so we can slice
                the values buffer by bytes:

                >>> byte_offset = start * self.item_size
                >>> byte_length = length * self.item_size
                >>> sub_values = self.values.slice(byte_offset, byte_length)

                Validity (if present) is sliced in element coordinates:

                >>> sub_validity = self.validity.slice(start, length) if self.validity else None

                The result is a new contiguous PrimitiveArray sharing the same underlying memory.

            Q. What happens when the selection is a contiguous logical slice but the array is non-contiguous?
                Even if logical indices are consecutive, the base indices in the values buffer may not be.
                Therefore we slice only the indices buffer (each entry is int32 = 4 bytes):

                >>> index_offset = start * 4
                >>> index_length = length * 4
                >>> sub_indices = self.indices.slice(index_offset, index_length)

                The result is a smaller non-contiguous view that still shares values/validity with the original.

            Q. What happens when the selection is not a contiguous logical slice?
                For arbitrary patterns (jumps, duplicates, reordering), we create a new indices buffer that
                directly stores base indices into `values`.

                - If this array is contiguous:
                    base_indices = normalized
                - If this array is non-contiguous:
                    base_indices = [unpack_int32(self.indices, i) for i in normalized]

                Then we pack base_indices into an int32 buffer:

                >>> new_indices = pack_int32(base_indices)

                The returned array shares the same values/validity buffers and uses new_indices as its view mapping.

            Q. Does take ever copy the values buffer?
                No. In all branches, values is either sliced as a view (contiguous case) or shared via indices
                (non-contiguous case). The only newly allocated memory is for the indices buffer when needed.
        """
        num_items = len(indices)
        if num_items == 0:
            return PrimitiveArray(self.dtype, 0, values=Buffer.from_bytearray(bytearray(0)))

        normalized = [normalize_index(i, self.length) for i in indices]
        is_contiguous_slice = all(normalized[k] + 1 == normalized[k + 1] for k in range(num_items - 1))

        if is_contiguous_slice:
            start = normalized[0]
            length = num_items
            if self.is_contiguous():
                byte_offset = start * self.item_size
                byte_length = length * self.item_size
                sub_values = self.values.slice(byte_offset, byte_length)
                sub_validity = self.validity.slice(start, length) if self.validity else None
                return PrimitiveArray(self.dtype, length, sub_values, sub_validity)
            else:
                index_offset = start * 4
                index_length = length * 4
                sub_indices = self.indices.slice(index_offset, index_length)
                return PrimitiveArray(self.dtype, length, self.values, self.validity, sub_indices)

        else:
            base_indices = normalized if self.is_contiguous() else [unpack_int32(self.indices, i) for i in normalized]
            new_indices = pack_int32(base_indices)
            return PrimitiveArray(self.dtype, len(base_indices), self.values, self.validity, new_indices)

    def to_list(self) -> List[Optional[PrimitiveType]]:
        """
        Convert the array into a Python list of scalars (and None for nulls).

        Returns:
            List[Optional[PrimitiveType]]: Python values (int/float/bool) and None for nulls.

        Discussion:
            Q. How does to_list interpret nulls?
                For each logical index i:
                - If `self.is_null(i)` is True, it appends None.
                - Otherwise it reads the value from the values buffer.

            Q. How does it read each value from the buffer?
                It computes the element’s base index and byte offset:
                    offset = self.base_index(i) * self.item_size
                Then reads the scalar with:
                    struct.unpack_from(self.fmt, self.values.data, offset)[0]

            Q. Why might to_list be considered a "materialization" step?
                The array stores data as packed bytes + bitmap metadata. Converting to a Python list creates
                Python objects for each element, which is often desirable for debugging or interop, but it loses
                the compact, contiguous representation.
        """
        output = []
        for i in range(self.length):
            if self.is_null(i):
                output.append(None)
            else:
                offset = self.base_index(i) * self.item_size
                value = struct.unpack_from(self.fmt, self.values.data, offset)[0]
                output.append(value)
        return output

    @classmethod
    def from_list(cls, data: list, dtype: Optional[DataType] = None) -> "PrimitiveArray":
        """
        Build a contiguous PrimitiveArray from a Python list, optionally with an explicit dtype.

        Args:
            data (list): A list containing primitive values and/or None for nulls.
            dtype (Optional[DataType]):
                - If provided, forces the dtype.
                - If None, dtype is inferred by `infer_primitive_dtype(data)`.

        Returns:
            PrimitiveArray: A newly constructed contiguous PrimitiveArray (indices=None).

        Discussion:
            Q. How is dtype chosen?
                - If dtype is given, it is used as-is.
                - Otherwise, dtype is inferred from the input list via `infer_primitive_dtype(data)`.

            Q. How are nulls represented?
                Nulls are represented by validity=0 and a placeholder in the values buffer:
                - BOOL uses False
                - Numeric types use 0
                The placeholder is never observed when validity marks an element as null.

            Q. Why reject floats for integer dtypes?
                Silent float->int conversion can lose information and hide bugs. This implementation requires
                explicit intent: use a float dtype or omit dtype and let inference select a float dtype.

            Q. Why allow bools in integer/float arrays but not in BOOL arrays?
                Python bool is a subclass of int, so treating it as 0/1 can be convenient for numeric arrays.
                For BOOL dtype, this implementation enforces strict bool-or-None inputs.
        """
        target = dtype if dtype is not None else infer_primitive_dtype(data)
        if target not in FMT:
            raise ValueError(f"Unsupported data type for PrimitiveArray: {target}")

        builder = PrimitiveArrayBuilder(target)

        if target is BOOL:
            for v in data:
                if v is None:
                    builder.append(None)
                elif isinstance(v, bool):
                    builder.append(v)
                else:
                    raise TypeError("BOOL dtype expects `bool` or `None`.")

        elif target in (INT32, INT64):
            for v in data:
                if v is None:
                    builder.append(None)
                    continue
                if isinstance(v, bool):
                    builder.append(int(v))
                elif isinstance(v, int):
                    if target is INT32 and not (INT32_MIN <= v <= INT32_MAX):
                        raise OverflowError(f"Value {v} out of int32 range")
                    builder.append(v)
                elif isinstance(v, float):
                    raise TypeError(
                        "Float value provided for integer dtype. Use FLOAT32/FLOAT64 or omit dtype for inference."
                    )
                else:
                    raise TypeError(f"Integer dtype expects int/bool/None, got {type(v).__name__}.")

        elif target in (FLOAT32, FLOAT64):
            for v in data:
                if v is None:
                    builder.append(None)
                elif isinstance(v, (bool, int, float)):
                    builder.append(float(v))
                else:
                    raise TypeError(f"Float dtype expects float/int/bool/None, got {type(v).__name__}.")

        else:
            raise TypeError(f"Unsupported dtype: {target}")

        return builder.finish()


class PrimitiveArrayBuilder(ArrayBuilder):
    """
    A builder for constructing a contiguous PrimitiveArray.

    The builder collects:
        - values: primitive values (including placeholders for nulls)
        - validity: 0/1 flags (1 = valid, 0 = null)

    When `finish()` is called, it allocates a single bytearray of size (num_items * item_size),
    packs all values into it using `struct.pack_into`, and builds a `Bitmap` from the validity list.

    Notes:
        - Null handling:
            If a value is None, validity stores 0 and values stores a placeholder:
            - BOOL -> False
            - Numeric -> 0
            The validity bitmap decides whether the placeholder is interpreted or treated as null.

    Discussion:
        Q. Why use a builder instead of constructing the buffer incrementally?
            Incrementally appending packed bytes can cause repeated reallocations and copies.
            A builder can allocate the final buffer once and write into it by offset.

        Q. Why store placeholders for nulls in values?
            The values buffer must have a fixed-width slot for every element to keep offsets simple:
                offset = index * item_size
            Nullness is represented by validity, so the placeholder never needs semantic meaning.

        Q. Why can Bitmap.from_list return None?
            If there are no nulls (all validity bits are 1), storing a bitmap is redundant.
            Returning None saves memory and signals that every element is valid.
    """

    def __init__(self, dtype: DataType):
        """
        Initialize the builder for a specific primitive dtype.

        Args:
            dtype (DataType): Target dtype to build. Must exist in `FMT`.

        Discussion:
            Q. How does the builder know how many bytes each element needs?
                It reads (fmt, item_size) from `FMT[dtype]`. `item_size` controls how large the final buffer is,
                and `fmt` controls how values are packed/unpacked.

            Q. Why does the builder keep Python lists before packing?
                This implementation prioritizes clarity. The builder stage separates concerns:
                collect values + validity first, then pack once at the end into a contiguous buffer.
        """
        assert dtype in FMT, f"Unsupported data type: {dtype}"
        self.dtype = dtype
        self.fmt, self.item_size = FMT[dtype]
        self.values = []
        self.validity = []

    def append(self, value: Optional[PrimitiveType]) -> "PrimitiveArrayBuilder":
        """
        Append a single element (or null) to the builder.

        Args:
            value (Optional[PrimitiveType]): A primitive value, or None to represent null.

        Returns:
            PrimitiveArrayBuilder: Returns self to allow chaining.

        Discussion:
            Q. What exactly is stored when value is None?
                The builder appends:
                - validity += [0]
                - values += [placeholder]
                The placeholder is:
                - False for BOOL dtype
                - 0 for all numeric dtypes
                This ensures the values buffer still has one fixed-width slot per logical element.

            Q. What exactly is stored when value is not None?
                The builder appends:
                - validity += [1]
                - values += [value]
                The value will later be packed into bytes using `struct.pack_into`.

            Q. Why doesn't append enforce dtype-specific type rules?
                This builder is a low-level helper. In this codebase, `PrimitiveArray.from_list` enforces
                dtype-specific rules and conversions before calling append.
        """
        if value is None:
            self.validity.append(0)
            self.values.append(False if self.dtype is BOOL else 0)
        else:
            self.validity.append(1)
            self.values.append(value)
        return self

    def finish(self) -> PrimitiveArray:
        """
        Finalize the builder and produce a contiguous PrimitiveArray.

        Returns:
            PrimitiveArray: A contiguous PrimitiveArray (indices=None) constructed from the builder state.

        Discussion:
            Q. How is the final values buffer created?
                It allocates a bytearray sized for all elements:
                    raw_buffer = bytearray(num_items * item_size)
                Then it packs each element sequentially using `struct.pack_into`, advancing the offset by item_size.

            Q. Why use struct.pack_into instead of struct.pack repeatedly?
                `pack_into` writes directly into a preallocated buffer at a given offset, which avoids creating many
                small temporary bytes objects.

            Q. How is validity represented in the final array?
                The builder converts the list of 0/1 flags into a Bitmap:
                    validity = Bitmap.from_list(self.validity)
                If there are no nulls, Bitmap.from_list may return None, which means every element is valid.

            Q. Does finish create a view or does it materialize data?
                It materializes a new contiguous values buffer. This is the step that converts Python-level values
                into an Arrow-like binary representation.
        """
        num_items = len(self.values)
        raw_buffer = bytearray(num_items * self.item_size)

        offset = 0
        for value in self.values:
            struct.pack_into(self.fmt, raw_buffer, offset, value)
            offset += self.item_size

        buffer = Buffer.from_bytearray(raw_buffer)
        validity = Bitmap.from_list(self.validity)
        return PrimitiveArray(self.dtype, num_items, buffer, validity, indices=None)
