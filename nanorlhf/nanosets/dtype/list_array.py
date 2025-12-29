from typing import Optional, Union, Sequence, List, Iterable, Any

from nanorlhf.nanosets.core.bitmap import Bitmap
from nanorlhf.nanosets.core.buffer import Buffer
from nanorlhf.nanosets.dtype.array import Array, ArrayBuilder
from nanorlhf.nanosets.dtype.dtype import LIST
from nanorlhf.nanosets.dtype.dtype_inference import infer_child_builder
from nanorlhf.nanosets.utils import normalize_index, unpack_int32, pack_int32


class ListArray(Array):
    """
    An Arrow-style list array backed by an int32 offsets buffer and a single child array,
    optionally accompanied by a validity bitmap and an indices buffer for zero-copy views.

    This class represents a column of variable-length lists by:
        - Storing all list elements in one contiguous `child: Array`.
        - Using `offsets: Buffer` (int32 entries) to define each list boundary in child:
            - i-th list occupies child[offsets[i] : offsets[i + 1]]

    Missingness (null) is tracked separately by `Bitmap`, and view semantics (non-contiguous indexing)
    are achieved through an optional `indices` buffer (int32 entries) mapping logical indices
    to physical indices in the base (offsets/child) storage.

    Notes:
        - `offsets` must contain `base_length + 1` int32 entries.
        - `base_length = (#offset entries - 1)` is the number of lists described by offsets.
        - The final offsets entry (offsets[base_length]) must be <= len(child).
        - If `indices` is present, this array is a non-contiguous view:
            - logical index -> base_index (physical list index)
            - physical list index -> [child_start, child_end) via offsets
        - This array intentionally does not implement `__setitem__` to keep an immutable, Arrow-like usage model.

    Examples:
        >>> arr = ListArray.from_list([[1, 2], [], None, [3]])
        >>> arr[0]
        [1, 2]
        >>> arr[1]
        []
        >>> arr[2] is None
        True
        >>> arr.take([3, 0]).to_list()
        [[3], [1, 2]]

    Discussion:
        Q. Why store lists as (child array + offsets) instead of Python nested lists?
            Python nested lists store many separate objects and pointers, leading to scattered memory access.
            An Arrow-style layout stores all elements in a single child array and keeps boundaries in offsets,
            enabling a compact representation and making it easier to share across languages and runtimes.

        Q. What do offsets represent, exactly?
            Offsets are int32 positions into the child array. For an element at physical index i:
            - start = offsets[i]
            - end   = offsets[i + 1]
            - list elements are child[start:end]
            Offsets length must be base_length + 1 so that the last element has a valid end offset.

        Q. Why track nulls with a bitmap instead of storing None in child?
            Nullness is metadata. By separating validity from child storage, the child array can remain a pure
            array of elements. A bitmap encodes validity using 1 bit per element, and the list payload does not
            need placeholder objects to represent nulls.

        Q. What is the purpose of the indices buffer?
            It enables view semantics without copying offsets/child. A view can select or reorder lists by storing
            an int32 mapping from logical indices to physical list indices. The view reuses the same offsets and
            child array from the base array.

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
        child: Array,
        validity: Optional[Bitmap] = None,
        indices: Optional[Buffer] = None,
    ):
        """
        Initialize a ListArray.

        Args:
            offsets (Buffer): An int32 buffer of offsets into the `child` array.
                - Must be a multiple of 4 bytes (int32).
                - Must contain (base_length + 1) entries where base_length = (#entries - 1).
            length (int): Length of the array when `indices` is None (contiguous base array).
            child (Array): The child array that stores all list elements contiguously.
            validity (Optional[Bitmap]): Optional validity bitmap (1 = valid, 0 = null).
            indices (Optional[Buffer]): Optional int32 buffer mapping logical indices -> physical list indices.

        Discussion:
            Q. How is base_length computed from offsets?
                Offsets store int32 entries. If offsets contains N entries, then it can describe N-1 lists,
                because each list needs a pair (offsets[i], offsets[i+1]). Therefore:
                - base_length = (len(offsets) // 4) - 1

            Q. Why validate the last offsets entry against the child length?
                The final offset offsets[base_length] equals the total number of child elements referenced by
                this offsets buffer. If it exceeds len(child), then at least one list would read beyond the child
                array, which would make __getitem__ invalid.

            Q. How is the logical length determined?
                - If indices is None (contiguous), logical length is the provided `length`, and it must match
                  base_length because offsets/child represent exactly that many lists.
                - If indices is present (non-contiguous view), logical length is (#indices entries) because the view
                  defines how many logical elements exist:
                    logical_length = len(indices) // 4

            Q. Why allow logical_length to differ from base_length when indices exists?
                In a view, offsets/child belong to a base storage that can be larger than the view. The view
                references only selected physical list indices, so it can be shorter (or reordered) without changing
                the base buffers.
        """
        if len(offsets) % 4 != 0:
            raise ValueError("offsets buffer size must be a multiple of 4 (int32)")

        base_length = len(offsets) // 4 - 1
        if base_length < 0:
            raise ValueError("offsets buffer must contain at least one entry")

        total_elems = unpack_int32(offsets, base_length)
        if total_elems > len(child):
            raise ValueError(f"offsets refer to {total_elems} child elements, but child length is {len(child)}")

        if indices is None:
            logical_length = length
            if logical_length != base_length:
                raise ValueError(f"length mismatch: base_length={base_length}, length argument={length}")
        else:
            if len(indices) % 4 != 0:
                raise ValueError("indices buffer size must be a multiple of 4 (int32)")
            logical_length = len(indices) // 4

        super().__init__(LIST, logical_length, child.values, validity, indices)

        self.offsets = offsets
        self.child = child
        self.base_length = base_length

    def __getitem__(self, key: Union[int, slice]):
        """
        Read an element (int index) or return a selection (slice).

        Args:
            key (Union[int, slice]):
                - int: returns a single list value (as a Python list) or None if null.
                - slice: returns a new ListArray corresponding to the slice, implemented via `take`.

        Returns:
            Union[Optional[list], "ListArray"]:
                - If key is int: returns a Python list (materialized) or None.
                - If key is slice: returns a ListArray view/selection.

        Discussion:
            Q. What happens when key is an integer?
                - If `self.is_null(key)` is True, return None.
                - Otherwise compute the base (physical) index via:
                    idx = self.base_index(key)
                - Validate that the base index is within the base range:
                    0 <= idx < self.base_length
                - Read child boundaries from offsets:
                    start = unpack_int32(self.offsets, idx)
                    end   = unpack_int32(self.offsets, idx + 1)
                - Validate [start, end) against the child length and ordering.
                - If start == end, return [] (empty list).
                - Otherwise slice the child via take and materialize:
                    sub = self.child.take(range(start, end))
                    return sub.to_list()

            Q. Why is base_index needed instead of using key directly?
                If the array is contiguous, base_index(key) matches the normalized key.
                If the array is a view (indices exists), base_index(key) reads indices[key] so the logical index
                maps to a physical list position in the shared offsets/child storage.

            Q. Why check base_length for the base index?
                Offsets describe the base storage with `base_length` lists. Even if the logical length is smaller
                (view), every logical element must map to a valid physical list index; otherwise offsets lookups
                would be invalid.

            Q. What happens when key is a slice?
                The slice is normalized via `key.indices(self.length)` and then delegated to:
                    self.take(range(start, stop, step))
                This makes slicing reuse the same selection semantics as take.

            Q. Why does integer indexing return a Python list instead of a child Array view?
                This implementation chooses to materialize each list value for ergonomic Python access. Internally,
                the payload is still stored as offsets + child; to_list provides a consistent Python-level output.
        """
        if isinstance(key, int):
            if self.is_null(key):
                return None

            idx = self.base_index(key)
            if not (0 <= idx < self.base_length):
                raise IndexError(f"base index {idx} out of range [0, {self.base_length})")

            start = unpack_int32(self.offsets, idx)
            end = unpack_int32(self.offsets, idx + 1)

            if start < 0 or end < start or end > len(self.child):
                raise ValueError(f"Invalid child range: start={start}, end={end}, child_length={len(self.child)}")

            if start == end:
                return []

            sub_array = self.child.take(range(start, end))
            return sub_array.to_list()

        if isinstance(key, slice):
            start, stop, step = key.indices(self.length)
            return self.take(range(start, stop, step))

        raise TypeError(f"Invalid index type: {type(key).__name__}")

    def take(self, indices: Sequence[int]) -> "ListArray":
        """
        Select elements by logical indices and return a new ListArray.

        Args:
            indices (Sequence[int]):
                - A sequence of logical indices to take.
                - Negative indices are allowed and are normalized.

        Returns:
            ListArray: A new ListArray representing the selected elements.

        Notes:
            - This method aims to avoid copying offsets/child whenever possible.
            - Contiguous logical selections from a contiguous base may produce:
                - a sliced child (via child.take on a contiguous range)
                - rebuilt offsets rebased to the new child
              and return a new contiguous ListArray (indices=None).
            - Selections over a non-contiguous base, or arbitrary patterns, become indices-based views that share
              offsets/child (and typically validity) with the original.

        Discussion:
            Q. What are "logical indices" in this method?
                The indices passed to `take` are interpreted in the coordinate system of this array:
                0..self.length-1. If this array is a view, those logical indices will ultimately map to
                physical list indices in the base offsets/child storage.

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
                It returns an empty ListArray with:
                - offsets = [0] (one entry, so base_length = 0)
                - child shared as-is
                - validity=None, indices=None
                This matches the offsets invariant: offsets length must be base_length + 1.

            Q. What happens for a contiguous logical slice when the array is contiguous?
                In a contiguous base, physical indices match logical indices. Therefore the selected lists span
                a contiguous region in the child coordinate system, bounded by:
                    child_start = offsets[base_start]
                    child_end   = offsets[base_end]
                The algorithm:
                - Slice child to new_child via child.take(range(child_start, child_end)).
                - Rebuild offsets locally by subtracting child_start from each original offsets entry in
                  [base_start, base_end].
                - Slice validity (if present) in logical coordinates.
                - Return a new contiguous ListArray (indices=None) with new_offsets/new_child.

            Q. Why does the contiguous branch rebuild offsets, instead of slicing offsets directly?
                Offsets store absolute positions into the base child. After producing new_child that starts at
                child_start, the first offset must become 0. Rebuilding offsets by subtracting child_start
                produces a consistent local offsets buffer for the sliced child.

            Q. What happens for a contiguous logical slice when the array is non-contiguous?
                Logical contiguity does not imply physical contiguity. The underlying physical list indices could
                be arbitrary, so we do not rebuild offsets/child. Instead we slice only the indices buffer
                (int32 entries) to produce a smaller view, and slice validity for the logical range.

            Q. What happens when the selection is not a contiguous logical slice?
                For arbitrary patterns (jumps, duplicates, reordering), we create a new indices buffer storing
                physical list indices into the shared offsets/child storage:
                - If this array is contiguous: base_indices = normalized
                - If this array is non-contiguous: base_indices = [unpack_int32(self.indices, i) for i in normalized]
                Then pack them into int32:
                    new_indices = pack_int32(base_indices)
                The returned array shares offsets/child (and validity) and uses new_indices as the view mapping.

            Q. Does take ever copy the child array?
                It does not copy underlying child buffers when the child supports zero-copy slicing/views.
                The main allocations in take are:
                - a new offsets buffer for the contiguous+contiguous slice case (to rebase offsets)
                - a new indices buffer for arbitrary selections (view mapping)
        """
        num_items = len(indices)
        if num_items == 0:
            empty_offsets = pack_int32([0])
            return ListArray(
                offsets=empty_offsets,
                length=0,
                child=self.child,
                validity=None,
                indices=None,
            )

        normalized = [normalize_index(i, self.length) for i in indices]
        is_contiguous_slice = all(normalized[k] + 1 == normalized[k + 1] for k in range(num_items - 1))

        if is_contiguous_slice:
            start = normalized[0]
            length = num_items

            if self.is_contiguous():
                base_start = start
                base_end = start + length

                child_start = unpack_int32(self.offsets, base_start)
                child_end = unpack_int32(self.offsets, base_end)
                new_child = self.child.take(range(child_start, child_end))

                local_offsets: List[int] = []
                for i in range(base_start, base_end + 1):
                    off = unpack_int32(self.offsets, i)
                    local_offsets.append(off - child_start)

                new_offsets = pack_int32(local_offsets)
                new_validity = self.validity.slice(start, length) if self.validity else None
                return ListArray(
                    offsets=new_offsets,
                    child=new_child,
                    length=length,
                    validity=new_validity,
                    indices=None,
                )

            else:
                index_offset = start * 4
                index_length = length * 4

                sub_indices = self.indices.slice(index_offset, index_length)  # type: ignore[arg-type]
                new_validity = self.validity.slice(start, length) if self.validity else None
                return ListArray(
                    offsets=self.offsets,
                    length=length,
                    child=self.child,
                    validity=new_validity,
                    indices=sub_indices,
                )

        base_indices = normalized if self.is_contiguous() else [unpack_int32(self.indices, i) for i in normalized]
        new_indices = pack_int32(base_indices)
        return ListArray(
            offsets=self.offsets,
            length=len(base_indices),
            child=self.child,
            validity=self.validity,
            indices=new_indices,
        )

    def to_list(self) -> List[Optional[list]]:
        """
        Convert the array into a Python list of lists (and None for nulls).

        Returns:
            List[Optional[list]]: Python list values and None for nulls.

        Discussion:
            Q. How does to_list interpret nulls?
                It iterates over logical indices i:
                - If `self.is_null(i)` is True, it appends None.
                - Otherwise it appends `self[i]`, which reads offsets, slices child, and materializes a Python list.

            Q. Why does it delegate list reading to self[i]?
                Centralizing range validation and view mapping in `__getitem__` ensures the same checks and mapping
                rules (base_index, offsets bounds, child slicing) are applied consistently.

            Q. Why might to_list be considered a "materialization" step?
                The array stores data as offsets + child + bitmap metadata. Converting to a Python structure
                creates Python list objects (and possibly Python scalars/strings inside), which is convenient for
                interop and debugging but loses the compact Arrow-style representation.
        """
        outputs: List[Optional[list]] = []
        for i in range(self.length):
            if self.is_null(i):
                outputs.append(None)
            else:
                outputs.append(self[i])
        return outputs

    @classmethod
    def from_list(cls, data: List[Optional[Iterable[Any]]]) -> "ListArray":
        """
        Build a contiguous ListArray from a Python list of iterables (and/or None).

        Args:
            data (List[Optional[Iterable[Any]]]): Input values where each element is either an iterable (non-string)
                representing a row of elements, or None to represent a null list.

        Returns:
            ListArray: A newly constructed contiguous ListArray (indices=None).

        Discussion:
            Q. Why infer a child builder?
                A ListArray stores its payload in a single child Array, so it must decide what ArrayBuilder to use
                for the child elements (primitive, string, nested list, struct-like, tensor-like, etc.). Inference
                chooses a builder based on a representative sample element and validates consistency.

            Q. Why use a builder here?
                Building requires simultaneously:
                - appending child elements into a child_builder
                - recording cumulative element counts into offsets
                - recording 0/1 validity per row
                The builder keeps these concerns together and produces offsets/child/validity in finish().

            Q. What happens for None rows?
                None rows are represented by validity=0 and an offsets step that does not advance (offset repeats
                the previous value). This is consistent with the idea that a null list contributes no elements to
                the child array.
        """
        child_builder = infer_child_builder(data)
        builder = ListArrayBuilder(child_builder)
        for row in data:
            builder.append(row)

        return builder.finish()


class ListArrayBuilder(ArrayBuilder):
    """
    A builder for constructing a contiguous ListArray.

    The builder collects:
        - child_builder: an ArrayBuilder accumulating all child elements
        - offsets: cumulative element positions into the eventual child array (starts with 0; length is num_items + 1)
        - validity: 0/1 flags (1 = valid, 0 = null)
        - length: number of appended rows

    When `finish()` is called, it packs offsets into an int32 `Buffer`, finalizes the child array via
    child_builder.finish(), builds a `Bitmap` from validity, and returns a contiguous ListArray (indices=None).

    Notes:
        - Null handling:
            If a value is None:
            - validity appends 0
            - offsets appends the same last offset (no elements added)
            If a value is an iterable:
            - each element is appended to child_builder
            - validity appends 1
            - offsets appends the new cumulative element count

        - Iterable handling:
            Strings/bytes are iterables but are treated as scalar-like in many APIs; this builder rejects
            (str, bytes, bytearray) inputs to avoid accidental character-wise iteration.

    Discussion:
        Q. Why does offsets start with [0]?
            Offsets represent cumulative element positions in the child array. The first list must start at
            element 0 in the child, so offsets[0] is always 0. Every append then pushes a new end position,
            so offsets length becomes num_items + 1.

        Q. Why do nulls append the same last offset?
            A null list contributes no elements to the child array. Keeping offsets unchanged means the element
            span [offsets[i], offsets[i+1]) is empty for that position, while validity marks it as null so
            __getitem__ returns None instead of [].

        Q. Why validate offsets length and validity length in finish?
            - validity must align 1:1 with rows (num_items).
            - offsets must have exactly one more entry than rows so that offsets[i+1] is defined for the last row.
            These checks ensure the offsets/validity invariants hold before creating the final array.

        Q. Why can Bitmap.from_list return None?
            If there are no nulls (all validity bits are 1), storing a bitmap is redundant. Returning None saves
            memory and signals that every element is valid.
    """

    def __init__(self, child_builder: ArrayBuilder):
        """
        Initialize the ListArrayBuilder.

        Args:
            child_builder (ArrayBuilder): Builder used to accumulate all child elements for this ListArray.

        Discussion:
            Q. What state does the builder maintain?
                - child_builder accumulates elements in the child coordinate system.
                - offsets starts as [0] and grows by one entry per appended row.
                - validity is a list of 0/1 flags aligned with appended rows.
                - length counts how many rows have been appended.

            Q. Why accept a child_builder instead of a dtype?
                The child can itself be a complex array (string, list, struct, tensor, etc.). Accepting a builder
                lets ListArrayBuilder remain generic and delegate child element handling to the appropriate builder.
        """
        self.child_builder = child_builder
        self.offsets: List[int] = [0]
        self.validity: List[int] = []
        self.length: int = 0

    def append(self, value: Optional[Iterable[Any]]) -> "ListArrayBuilder":
        """
        Append a single list (iterable) or a null list to the builder.

        Args:
            value (Optional[Iterable[Any]]): An iterable of elements for one row, or None to represent a null list.

        Returns:
            ListArrayBuilder: Returns self to allow chaining.

        Discussion:
            Q. What exactly is stored when value is None?
                The builder appends:
                - validity += [0]
                - offsets  += [offsets[-1]]
                - length   += 1
                No elements are appended to child_builder. The row is considered null by validity.

            Q. What exactly is stored when value is an iterable?
                The builder:
                - appends validity += [1]
                - iterates elements and appends each element into child_builder
                - counts how many elements were appended (count)
                - appends offsets += [offsets[-1] + count]
                - increments length by 1

            Q. Why reject (str, bytes, bytearray)?
                These are iterables, but treating them as list rows would iterate over characters/bytes, which is
                usually unintended. Rejecting them makes the API safer and more explicit.
        """
        if value is None:
            self.validity.append(0)
            self.offsets.append(self.offsets[-1])
            self.length += 1
            return self

        if isinstance(value, (str, bytes, bytearray)) or not hasattr(value, "__iter__"):
            raise TypeError(
                f"ListArrayBuilder.append expects an iterable (non-string) or None, got {type(value).__name__}"
            )

        self.validity.append(1)
        start_count = self.offsets[-1]
        count = 0
        for elem in value:
            self.child_builder.append(elem)
            count += 1

        self.offsets.append(start_count + count)
        self.length += 1
        return self

    def finish(self) -> ListArray:
        """
        Finalize the builder and produce a contiguous ListArray.

        Returns:
            ListArray: A contiguous ListArray (indices=None) constructed from the builder state.

        Discussion:
            Q. Why must validity length match num_items?
                Validity stores one 0/1 flag per row. If the lengths mismatch, null tracking would be misaligned
                with offsets and indexing.

            Q. Why must offsets length be num_items + 1?
                Each row needs a start and an end offset. With num_items rows, there must be num_items + 1 offsets
                so that offsets[i+1] is defined for the last row.

            Q. How are final buffers produced?
                - offsets_buffer = pack_int32(self.offsets)
                - child_array    = self.child_builder.finish()
                - validity_bitmap = Bitmap.from_list(self.validity)

            Q. Does finish create a view or does it materialize data?
                It materializes a new offsets buffer and finalizes the child array via child_builder.finish().
                This is the step that converts Python-level nested iterables into an Arrow-like
                (offsets + child + validity) representation.
        """
        num_items = self.length
        if len(self.validity) != num_items:
            raise ValueError(f"validity length {len(self.validity)} does not match number of items {num_items}")
        if len(self.offsets) != num_items + 1:
            raise ValueError(
                f"offsets length must be num_items + 1, got offsets={len(self.offsets)}, num_items={num_items}"
            )

        offsets_buffer = pack_int32(self.offsets)
        child_array = self.child_builder.finish()
        validity_bitmap = Bitmap.from_list(self.validity)

        return ListArray(
            offsets=offsets_buffer,
            length=num_items,
            child=child_array,
            validity=validity_bitmap,
            indices=None,
        )