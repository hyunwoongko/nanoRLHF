from typing import Optional, Union, Sequence, List

from nanorlhf.nanosets.core.bitmap import Bitmap
from nanorlhf.nanosets.core.buffer import Buffer
from nanorlhf.nanosets.dtype.array import Array, ArrayBuilder
from nanorlhf.nanosets.dtype.dtype import STRING
from nanorlhf.nanosets.utils import normalize_index, unpack_int32, pack_int32


class StringArray(Array):
    """
    An Arrow-style UTF-8 string array backed by a contiguous byte buffer and an int32 offsets buffer,
    optionally accompanied by a validity bitmap and an indices buffer for zero-copy views.

    This class stores all string bytes concatenated in a single `values: Buffer` (UTF-8 encoded),
    and uses `offsets: Buffer` (int32 entries) to define the start/end boundary of each string:
        - i-th string occupies bytes in [offsets[i], offsets[i + 1]) within `values`.

    Missingness (null) is tracked separately by `Bitmap`, and view semantics (non-contiguous indexing)
    are achieved through an optional `indices` buffer (int32 entries) mapping logical indices
    to physical indices in the base (offsets/values) storage.

    Example: offsets + values for ["Hello.", "I am Kevin.", "How are you?"]
        | i (string idx) | string | UTF-8 bytes | byte_len | start = offsets[i] | end = offsets[i+1] | values[start:end] (hex) |
        |---:|---|---|---:|---:|---:|---|
        | 0 | `Hello.` | `b"Hello."` | 6 | 0 | 6 | `48 65 6C 6C 6F 2E` |
        | 1 | `I am Kevin.` | `b"I am Kevin."` | 10 | 6 | 16 | `49 20 61 6D 20 4B 65 76 69 6E 2E` |
        | 2 | `How are you?` | `b"How are you?"` | 12 | 16 | 28 | `48 6F 77 20 61 72 65 20 79 6F 75 3F` |

        >>> offsets = [0, 6, 16, 28]
        >>> validity = None  # all values are valid
        >>> values (utf-8) = b"Hello.I am Kevin.How are you?"
        >>> values (hex)   = 48 65 6C 6C 6F 2E 49 20 61 6D 20 4B 65 76 69 6E 2E 48 6F 77 20 61 72 65 20 79 6F 75 3F

    Notes:
        - `offsets` must contain `physical_length + 1` int32 entries.
        - If `indices` is present, this array is a non-contiguous view:
            - logical index -> base_index (physical index)
            - physical index -> [byte_start, byte_end) via offsets
        - This array intentionally does not implement `__setitem__` to keep an immutable, Arrow-like usage model.

    Examples:
        >>> arr = StringArray.from_list(["a", None, "bc"])
        >>> arr[0]
        'a'
        >>> arr[1] is None
        True
        >>> arr.take([2, 0]).to_list()
        ['bc', 'a']
        >>> arr.take([0, 2])[0]
        'a'

    Discussion:
        Q. Why store strings as (values bytes + offsets) instead of Python objects?
            Python lists store object pointers, and each string is a separate Python object that can live
            anywhere in memory. An Arrow-style layout stores raw UTF-8 bytes contiguously and keeps boundaries
            in an offsets array, which enables a compact representation and makes it easier to share across
            languages and runtimes.

        Q. What do offsets represent, exactly?
            Offsets are int32 positions into the `values` byte buffer. For an element at physical index i:
            - start = offsets[i]
            - end   = offsets[i + 1]
            - string bytes are values[start:end]
            Offsets length must be physical_length + 1 so that the last element has a valid end offset.

        Q. Why track nulls with a bitmap instead of storing None in the data buffer?
            Nullness is metadata. By separating validity from values, the values buffer can remain a pure,
            contiguous byte sequence. A bitmap encodes validity using 1 bit per element, and `None` never needs
            to appear inside the packed bytes representation.

        Q. What is the purpose of the indices buffer?
            It enables view semantics without copying base buffers. A view can select or reorder elements by
            storing an int32 mapping from logical indices to physical indices. The view reuses the same
            offsets/values buffers from the base array.

        Q. Why does this class omit __setitem__?
            Arrow-style arrays are typically treated as immutable. If multiple views share the same buffers,
            in-place mutation can invalidate other views and complicate concurrency and reasoning. By omitting
            __setitem__, we encourage creating new arrays (via builders) or new views (via take) instead of
            mutating existing ones.
    """

    def __init__(
        self,
        offsets: Buffer,
        length: int,
        values: Buffer,
        validity: Optional[Bitmap] = None,
        indices: Optional[Buffer] = None,
    ):
        """
        Initialize a StringArray.

        Args:
            offsets (Buffer): An int32 buffer of offsets into the `values` byte buffer.
                - Must be a multiple of 4 bytes (int32).
                - Must contain (physical_length + 1) entries where physical_length = (#entries - 1).
            length (int): Length of the array when `indices` is None (contiguous base array).
            values (Buffer): A contiguous byte buffer containing all UTF-8 encoded string bytes concatenated.
            validity (Optional[Bitmap]): Optional validity bitmap (1 = valid, 0 = null).
            indices (Optional[Buffer]): Optional int32 buffer mapping logical indices -> physical indices.

        Discussion:
            Q. How is physical_length computed from offsets?
                Offsets store int32 entries. If offsets contains N entries, then it can describe N-1 strings,
                because each string needs a pair (offsets[i], offsets[i+1]). Therefore:
                - physical_length = (len(offsets) // 4) - 1

            Q. How is the logical length determined?
                - If indices is None (contiguous), logical length is the provided `length`, and it must match
                  physical_length because offsets/values represent exactly that many strings.
                - If indices is present (non-contiguous view), logical length is (#indices entries) because the view
                  defines how many logical elements exist:
                    logical_length = len(indices) // 4

            Q. Why allow logical_length to differ from physical_length when indices exists?
                In a view, offsets/values belong to a base storage that can be larger than the view. The view
                references only selected physical indices, so it can be shorter (or reordered) without changing
                the base buffers.
        """
        if len(offsets) % 4 != 0:
            raise ValueError("offsets buffer size must be a multiple of 4 (int32)")

        physical_length = len(offsets) // 4 - 1
        if physical_length < 0:
            raise ValueError("offsets buffer must contain at least one entry")

        if indices is None:
            logical_length = length
            if logical_length != physical_length:
                raise ValueError(f"length mismatch: base_length={physical_length}, length argument={length}")
        else:
            if len(indices) % 4 != 0:
                raise ValueError("indices buffer size must be a multiple of 4 (int32)")
            logical_length = len(indices) // 4

        super().__init__(STRING, logical_length, values, validity, indices)

        self.offsets = offsets
        self.physical_length = physical_length

    def __getitem__(self, key: Union[int, slice]):
        """
        Read an element (int index) or return a selection (slice).

        Args:
            key (Union[int, slice]):
                - int: returns a single string (Python str) or None if null.
                - slice: returns a new StringArray corresponding to the slice, implemented via `take`.

        Returns:
            Union[Optional[str], "StringArray"]:
                - If key is int: returns a Python str or None.
                - If key is slice: returns a StringArray view/selection.

        Discussion:
            Q. What happens when key is an integer?
                - If `self.is_null(key)` is True, return None.
                - Otherwise compute the base (physical) index via:
                    index = self.base_index(key)
                - Validate that the base index is within the physical range:
                    0 <= index < self.physical_length
                - Read byte boundaries from offsets:
                    start = unpack_int32(self.offsets, index)
                    end   = unpack_int32(self.offsets, index + 1)
                - Validate [start, end) against the values buffer size and ordering.
                - Slice the byte buffer and decode:
                    sub = self.values.slice(start, end - start)
                    return bytes(sub.data).decode("utf-8")

            Q. Why is base_index needed instead of using key directly?
                If the array is contiguous, base_index(key) matches the normalized key.
                If the array is a view (indices exists), base_index(key) reads indices[key] so the logical index
                maps to a physical string position in the shared offsets/values storage.

            Q. Why check physical_length for the base index?
                Offsets/values represent the base storage with `physical_length` strings. Even if the logical
                length is smaller (view), every logical element must map to a valid physical index; otherwise
                offsets lookups would be invalid.

            Q. What happens when the string length is zero?
                If end - start == 0, this returns the empty string "" immediately. This is distinct from null:
                - null is represented by validity=0 and returns None
                - empty string is a valid value and returns "".

            Q. What happens when key is a slice?
                The slice is normalized via `key.indices(self.length)` and then delegated to:
                    self.take(range(start, stop, step))
                This makes slicing reuse the same selection semantics as take.

            Q. Why raise TypeError for non-int and non-slice keys?
                It matches typical Python container behavior and keeps the indexing contract explicit.
        """
        if isinstance(key, int):
            if self.is_null(key):
                return None

            index = self.base_index(key)
            if not (0 <= index < self.physical_length):
                raise IndexError(f"base index {index} out of range [0, {self.physical_length})")

            start = unpack_int32(self.offsets, index)
            end = unpack_int32(self.offsets, index + 1)

            if start < 0 or end < start or end > len(self.values):
                raise ValueError(
                    f"Invalid string slice range: start={start}, end={end}, values_size={len(self.values)}"
                )

            length = end - start
            if length == 0:
                return ""

            sub_buffer = self.values.slice(start, length)
            return bytes(sub_buffer.data).decode("utf-8")

        if isinstance(key, slice):
            start, stop, step = key.indices(self.length)
            return self.take(range(start, stop, step))

        raise TypeError(f"Invalid index type: {type(key).__name__}")

    def take(self, indices: Sequence[int]) -> "StringArray":
        """
        Select elements by logical indices and return a new StringArray.

        Args:
            indices (Sequence[int]):
                - A sequence of logical indices to take.
                - Negative indices are allowed and are normalized.

        Returns:
            StringArray: A new StringArray representing the selected elements.

        Notes:
            - This method aims to avoid copying the `values` buffer whenever possible.
            - Contiguous logical selections from a contiguous base may become a (sub_values + rebuilt sub_offsets)
              contiguous array.
            - Selections over a non-contiguous base, or arbitrary patterns, become indices-based views that share
              offsets/values (and typically validity) with the original.

        Discussion:
            Q. What are "logical indices" in this method?
                The indices passed to `take` are interpreted in the coordinate system of this array:
                0..self.length-1. If this array is a view, those logical indices will ultimately map to
                physical indices in the base offsets/values storage.

            Q. Why normalize indices first?
                Python-style negative indexing is convenient. Normalization ensures each index lands in
                [0, self.length) before any contiguity checks or mapping logic runs:
                    normalized = [normalize_index(i, self.length) for i in indices]

            Q. How does it detect a contiguous logical slice?
                It checks whether the normalized indices are consecutive:
                    normalized[k+1] == normalized[k] + 1 for all k
                If true, the selection corresponds to a logical slice that can often be represented more
                compactly than an arbitrary gather.

            Q. What happens when indices is empty?
                It returns an empty StringArray with:
                - offsets = [0] (one entry, so physical_length = 0)
                - values  = empty byte buffer
                - validity=None, indices=None
                This matches the offsets invariant: offsets length must be physical_length + 1.

            Q. What happens for a contiguous logical slice when the array is contiguous?
                In a contiguous base, physical indices match logical indices. Therefore the selected strings
                occupy a contiguous byte span in `values`, but offsets must be rebased to the new sub_values.

                The algorithm:
                - Determine base_start/base_end in physical coordinates.
                - Compute byte_start/byte_end from offsets at those boundaries.
                - Slice values to sub_values = values[byte_start:byte_end].
                - Rebuild offsets locally by subtracting byte_start from each original offset in [base_start, base_end].
                - Slice validity (if present) in logical coordinates.
                - Return a new contiguous StringArray (indices=None) with sub_offsets/sub_values.

            Q. Why does the contiguous branch rebuild offsets, instead of slicing offsets directly?
                Offsets store absolute positions into the base `values`. After slicing `values` to `sub_values`,
                the start of `sub_values` must be treated as 0. Rebuilding offsets by subtracting byte_start
                produces a consistent local offsets buffer for the sliced values.

            Q. What happens for a contiguous logical slice when the array is non-contiguous?
                Logical contiguity does not imply physical contiguity. The underlying physical indices could be
                arbitrary. Therefore we do not attempt to rebuild offsets/values; instead we slice only the
                indices buffer (int32 entries) to produce a smaller view, and slice validity for the logical range.

            Q. What happens when the selection is not a contiguous logical slice?
                For arbitrary patterns (jumps, duplicates, reordering), we create a new indices buffer storing
                physical indices into the shared offsets/values storage:
                - If this array is contiguous: base_indices = normalized
                - If this array is non-contiguous: base_indices = [unpack_int32(self.indices, i) for i in normalized]
                Then pack them into int32:
                    new_indices = pack_int32(base_indices)
                The returned array shares offsets/values (and validity) and uses new_indices as the view mapping.

            Q. Does take ever copy the values buffer?
                It does not copy the underlying byte data when it slices `values`, because `Buffer.slice` is
                memoryview-based (zero-copy). The main allocation in take is:
                - a new offsets buffer for the contiguous+contiguous slice case (to rebase offsets)
                - a new indices buffer for arbitrary selections (view mapping)
        """
        num_items = len(indices)
        if num_items == 0:
            empty_offsets = pack_int32([0])
            empty_values = Buffer.from_bytearray(bytearray())
            return StringArray(empty_offsets, 0, empty_values, validity=None, indices=None)

        normalized = [normalize_index(i, self.length) for i in indices]
        is_contiguous_slice = all(normalized[k] + 1 == normalized[k + 1] for k in range(num_items - 1))

        if is_contiguous_slice:
            start = normalized[0]
            length = num_items

            if self.is_contiguous():
                base_start = start
                base_end = start + length

                byte_start = unpack_int32(self.offsets, base_start)
                byte_end = unpack_int32(self.offsets, base_end)
                byte_length = byte_end - byte_start

                sub_values = self.values.slice(byte_start, byte_length)

                local_offsets: List[int] = []
                for i in range(base_start, base_end + 1):
                    offset = unpack_int32(self.offsets, i)
                    local_offsets.append(offset - byte_start)

                sub_offsets = pack_int32(local_offsets)
                sub_validity = self.validity.slice(start, length) if self.validity else None
                return StringArray(
                    offsets=sub_offsets,
                    length=length,
                    values=sub_values,
                    validity=sub_validity,
                    indices=None,
                )

            else:
                index_offset = start * 4
                index_length = length * 4
                sub_indices = self.indices.slice(index_offset, index_length)  # type: ignore[arg-type]
                sub_validity = self.validity.slice(start, length) if self.validity else None
                return StringArray(
                    offsets=self.offsets,
                    length=length,
                    values=self.values,
                    validity=sub_validity,
                    indices=sub_indices,
                )

        base_indices = normalized if self.is_contiguous() else [unpack_int32(self.indices, i) for i in normalized]
        new_indices = pack_int32(base_indices)
        return StringArray(
            offsets=self.offsets,
            length=len(base_indices),
            values=self.values,
            validity=self.validity,
            indices=new_indices,
        )

    def to_list(self) -> List[Optional[str]]:
        """
        Convert the array into a Python list of strings (and None for nulls).

        Returns:
            List[Optional[str]]: Python str values and None for nulls.

        Discussion:
            Q. How does to_list interpret nulls?
                It iterates over logical indices i:
                - If `self.is_null(i)` is True, it appends None.
                - Otherwise it appends `self[i]`, which reads bytes via offsets and decodes UTF-8.

            Q. Why does it delegate string reading to self[i]?
                Centralizing decoding logic in `__getitem__` ensures the same checks and mapping rules
                (base_index, offsets bounds, UTF-8 decoding) are applied consistently.

            Q. Why might to_list be considered a "materialization" step?
                The array stores data as bytes + offsets + bitmap metadata. Converting to a Python list creates
                Python string objects for each element, which is convenient for interop and debugging but loses
                the compact Arrow-style representation.
        """
        output = []
        for i in range(self.length):
            if self.is_null(i):
                output.append(None)
            else:
                output.append(self[i])
        return output

    @classmethod
    def from_list(cls, data: List[Optional[str]]) -> "StringArray":
        """
        Build a contiguous StringArray from a Python list of strings (and/or None).

        Args:
            data (List[Optional[str]]): Input values where each element is either a Python str or None.

        Returns:
            StringArray: A newly constructed contiguous StringArray (indices=None).

        Discussion:
            Q. Why use a builder here?
                Strings have variable byte lengths, so building requires simultaneously growing a byte buffer
                (values) and recording cumulative offsets. The builder keeps these concerns together:
                - append bytes to data_bytes
                - append cumulative length to offsets
                - append 0/1 to validity
                Then finish packs offsets into int32 bytes and constructs the final array.

            Q. What happens for None values?
                None is represented by validity=0 and an offsets step that does not advance (offsets repeats
                the previous value). This is consistent with the idea that null has no bytes in the values buffer.
        """
        builder = StringArrayBuilder()
        for value in data:
            builder.append(value)
        return builder.finish()


class StringArrayBuilder(ArrayBuilder):
    """
    A builder for constructing a contiguous StringArray.

    The builder collects:
        - offsets: cumulative byte positions (starts with 0; length is num_items + 1)
        - data_bytes: concatenated UTF-8 encoded bytes for all non-null strings
        - validity: 0/1 flags (1 = valid, 0 = null)

    When `finish()` is called, it packs offsets into an int32 `Buffer`, wraps `data_bytes` into `values`,
    builds a `Bitmap` from validity, and returns a contiguous StringArray (indices=None).

    Notes:
        - Null handling:
            If a value is None:
            - validity appends 0
            - offsets appends the same last offset (no bytes added)
            If a value is a string:
            - UTF-8 bytes are appended to data_bytes
            - validity appends 1
            - offsets appends the new cumulative length

    Discussion:
        Q. Why does offsets start with [0]?
            Offsets represent cumulative byte positions. The first element must start at byte 0 in the values buffer,
            so offsets[0] is always 0. Every append then pushes a new end position, so offsets length becomes
            num_items + 1.

        Q. Why do nulls append the same last offset?
            A null value contributes no bytes to the values buffer. Keeping offsets unchanged means the byte span
            [offsets[i], offsets[i+1]) is empty for that position, while validity marks it as null so decoding
            is bypassed and None is returned instead.

        Q. Why validate offsets length in finish?
            Offsets must contain exactly one more entry than the number of elements, because each element needs
            a start and an end boundary. The check ensures the offsets invariants hold before creating the array.

        Q. Why can Bitmap.from_list return None?
            If there are no nulls (all validity bits are 1), storing a bitmap is redundant. Returning None saves
            memory and signals that every element is valid.
    """

    def __init__(self):
        """
        Initialize the StringArrayBuilder.

        Args:
            None

        Discussion:
            Q. What state does the builder maintain?
                - offsets starts as [0] and grows by one entry per appended element.
                - data_bytes is a bytearray accumulating UTF-8 bytes of all non-null strings.
                - validity is a list of 0/1 flags aligned with appended elements.

            Q. Why use a bytearray for data_bytes?
                A bytearray supports efficient incremental appends. This is convenient during building, and it can
                be wrapped into a Buffer at the end without requiring per-string allocations in the final layout.
        """
        self.offsets: List[int] = [0]
        self.data_bytes = bytearray()
        self.validity: List[int] = []

    def append(self, value: Optional[str]) -> "StringArrayBuilder":
        """
        Append a single string (or null) to the builder.

        Args:
            value (Optional[str]): A Python str value, or None to represent null.

        Returns:
            StringArrayBuilder: Returns self to allow chaining.

        Discussion:
            Q. What exactly is stored when value is None?
                The builder appends:
                - validity += [0]
                - offsets  += [offsets[-1]]
                No bytes are added to data_bytes. The element is considered null by validity.

            Q. What exactly is stored when value is a string?
                The builder:
                - encodes it as UTF-8 bytes
                - extends data_bytes with the encoded bytes
                - appends validity += [1]
                - appends offsets += [len(data_bytes)] (the new cumulative end offset)

            Q. Why enforce that non-null values must be str?
                This builder defines the string array contract: elements are UTF-8 strings or null. Enforcing
                types early prevents accidental insertion of incompatible objects that cannot be encoded.
        """
        if value is None:
            self.validity.append(0)
            self.offsets.append(self.offsets[-1])
        else:
            if not isinstance(value, str):
                raise TypeError(f"StringArray expects str or None, got {type(value).__name__}")
            encoded = value.encode("utf-8")
            self.data_bytes.extend(encoded)
            self.validity.append(1)
            self.offsets.append(len(self.data_bytes))
        return self

    def finish(self) -> StringArray:
        """
        Finalize the builder and produce a contiguous StringArray.

        Returns:
            StringArray: A contiguous StringArray (indices=None) constructed from the builder state.

        Discussion:
            Q. Why must offsets length be num_items + 1?
                Each element needs a start and an end offset. With num_items elements, there must be
                num_items + 1 offsets so that offsets[i+1] is defined for the last element.

            Q. How are final buffers produced?
                - offsets_buffer = pack_int32(self.offsets)
                - values_buffer  = Buffer.from_bytearray(self.data_bytes)
                - validity_bitmap = Bitmap.from_list(self.validity)

            Q. Does finish create a view or does it materialize data?
                It materializes a new contiguous values buffer for the concatenated bytes and a new offsets buffer.
                This is the step that converts Python-level strings into an Arrow-like (values + offsets + validity)
                binary representation.
        """
        num_items = len(self.validity)
        if len(self.offsets) != num_items + 1:
            raise ValueError(
                f"offsets length must be num_items + 1, got offsets={len(self.offsets)}, num_items={num_items}"
            )

        offsets_buffer = pack_int32(self.offsets)
        values_buffer = Buffer.from_bytearray(self.data_bytes)
        validity_bitmap = Bitmap.from_list(self.validity)

        return StringArray(
            offsets=offsets_buffer,
            length=num_items,
            values=values_buffer,
            validity=validity_bitmap,
            indices=None,
        )