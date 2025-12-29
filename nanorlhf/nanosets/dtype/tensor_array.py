from typing import List, Optional, Sequence

import torch

from nanorlhf.nanosets.core.bitmap import Bitmap
from nanorlhf.nanosets.core.buffer import Buffer
from nanorlhf.nanosets.dtype.array import Array, ArrayBuilder
from nanorlhf.nanosets.dtype.dtype import TENSOR
from nanorlhf.nanosets.utils import normalize_index, unpack_int32, pack_int32


class TensorArray(Array):
    """
    An Arrow-style tensor array that stores per-row torch.Tensor payloads in a base Python list,
    separates row-level nulls via an optional validity bitmap, and supports zero-copy view semantics
    through an optional int32 indices buffer.

    This array stores:
        - tensors: base storage of Optional[torch.Tensor]
        - validity: Bitmap indicating whether each base row is valid (1) or null (0)
        - indices: optional int32 mapping from logical indices to base indices (view)

    Discussion:
        Q. Why keep tensors as a Python list instead of a contiguous child buffer?
            Tensors are already structured objects with their own storage, dtype/device/shape, and lifecycle.
            TensorArray treats each tensor as an atomic row payload and focuses on Arrow-like semantics for
            null tracking and selection/view operations.

        Q. What is the difference between base_length and logical length?
            base_length is len(tensors) and describes the base storage capacity.
            logical length is the exposed length of the array:
                - if indices is None: logical length == base_length
                - if indices exists: logical length == (#indices entries)

        Q. How does validity interact with indices-based views?
            Validity is stored for the base storage. When the array is a view, logical indices are first mapped
            to base indices; nullness and bounds checks are interpreted in base coordinates.

        Q. Why does this class validate indices buffer alignment (multiple of 4 bytes)?
            Indices are interpreted as int32 entries. Enforcing 4-byte alignment ensures that len(indices) // 4
            is a valid count of indices and that unpacking operations remain well-defined.
    """

    def __init__(
        self,
        tensors: List[Optional[torch.Tensor]],
        validity: Optional[Bitmap] = None,
        indices: Optional[Buffer] = None,
    ):
        """
        Initialize a TensorArray.

        Args:
            tensors (List[Optional[torch.Tensor]]): Base storage for tensor rows. Each element is a torch.Tensor
                or None (placeholder for a null row).
            validity (Optional[Bitmap]): Optional base row-level validity bitmap (1 = valid, 0 = null). If None,
                all base rows are treated as valid.
            indices (Optional[Buffer]): Optional int32 indices buffer mapping logical index -> base index to
                represent a non-contiguous view without copying the base tensors list.

        Discussion:
            Q. How is logical length determined?
                - If indices is None: logical length is base_length = len(tensors).
                - If indices exists: logical length is len(indices) // 4, since indices stores int32 entries.

            Q. Why does the base Array store values=None?
                TensorArray does not store a single flat values buffer. Its payload is the list of tensors, so
                values are stored as None at the base Array level and data is accessed via `self.tensors`.

            Q. Why require validity length to match base_length (not logical length)?
                Validity describes nullness of base storage rows. Views are represented by indices, so logical
                rows are mapped into base rows; therefore validity must align with base storage.
        """
        base_length = len(tensors)
        logical_length = (len(indices) // 4) if indices is not None else base_length
        super().__init__(TENSOR, logical_length, values=None, validity=validity, indices=indices)

        self.tensors: List[Optional[torch.Tensor]] = tensors
        self.base_length = base_length

        if validity is not None and len(validity) != base_length:
            raise ValueError(
                f"Validity bitmap length ({len(validity)}) does not match number of base rows ({base_length})"
            )

        if not self.is_contiguous():
            if len(indices) % 4 != 0:
                raise ValueError("indices buffer size must be a multiple of 4 (int32)")

    def __len__(self) -> int:
        """
        Return the logical length of the array.

        Returns:
            int: The exposed length of this TensorArray (logical length). If indices is None, it equals
            base_length; otherwise it equals the number of indices entries.

        Discussion:
            Q. Why return logical length instead of base_length?
                TensorArray may be a view (indices exists). In that case, the array exposes only the indexed
                rows, so logical length is determined by indices, not by the base tensors list size.
        """
        return self.length

    def __getitem__(self, key):
        """
        Read one element (int index) or return a selection (slice).

        Args:
            key: Either:
                - int: Returns a torch.Tensor for that logical row, or None if the row is null.
                - slice: Returns a new TensorArray corresponding to the slice, implemented via `take`.

        Returns:
            - If key is int: Optional[torch.Tensor]
            - If key is slice: TensorArray

        Discussion:
            Q. What happens when key is an integer?
                - Normalize key into [0, self.length) via normalize_index.
                - If the normalized logical row is null, return None.
                - Map the logical index to a base index via base_index.
                - Validate base index range.
                - Return tensors[base_idx].

            Q. Why normalize before checking nullness?
                Python allows negative indices. Normalizing ensures a single canonical index is used for both
                null checks and base index mapping.

            Q. Why delegate slicing to take(range(...))?
                Using take for slicing keeps selection semantics consistent and centralizes the view/copy logic
                (contiguous optimization vs indices-based view construction).
        """
        if isinstance(key, int):
            normalized_idx = normalize_index(key, self.length)
            if self.is_null(normalized_idx):
                return None
            base_idx = self.base_index(normalized_idx)
            if not (0 <= base_idx < self.base_length):
                raise IndexError(f"base index {base_idx} out of range [0, {self.base_length})")
            return self.tensors[base_idx]

        if isinstance(key, slice):
            start, stop, step = key.indices(self.length)
            return self.take(range(start, stop, step))

        raise TypeError(f"Invalid index type for TensorArray: {type(key).__name__}")

    def take(self, indices: Sequence[int]) -> "TensorArray":
        """
        Select elements by logical indices and return a new TensorArray.

        Args:
            indices (Sequence[int]): Sequence of logical indices to take. Negative indices are allowed and are
                normalized.

        Returns:
            TensorArray: A TensorArray representing the selected rows. Depending on contiguity and whether this
            array is a base array or a view, the result may be:
                - a new contiguous TensorArray with sliced tensors/validity, or
                - an indices-based view sharing the same base tensors/validity.

        Discussion:
            Q. What are "logical indices" here?
                Indices are interpreted in the coordinate system of this TensorArray: 0..self.length-1.
                If this array is a view, logical indices map to base indices through the indices buffer.

            Q. What happens when indices is empty?
                It returns an empty TensorArray with no tensors and no validity/indices:
                    TensorArray([], None, None)

            Q. How does it detect a contiguous logical slice?
                It normalizes the indices and checks whether they are consecutive:
                    normalized[k+1] == normalized[k] + 1 for all k

            Q. What happens for a contiguous selection when the array is contiguous?
                Logical contiguity implies base contiguity, so it:
                    - slices the tensors list to sub_tensors
                    - slices validity (if present) for the same base range
                    - returns a new contiguous TensorArray (indices=None)

            Q. What happens for a contiguous selection when the array is already a view?
                Logical contiguity does not guarantee base contiguity. To avoid rebuilding base storage, it:
                    - slices the indices buffer to create a smaller view
                    - shares tensors and validity with the base

            Q. What happens for a non-contiguous selection?
                It constructs a new indices buffer (int32) describing base indices:
                    - if contiguous base: base_indices = normalized
                    - if view base: base_indices = [unpack_int32(self.indices, i) for i in normalized]
                Then returns a new view sharing tensors and validity.
        """
        num_items = len(indices)
        if num_items == 0:
            return TensorArray([], None, None)

        normalized = [normalize_index(i, self.length) for i in indices]
        is_contiguous_slice = all(normalized[k] + 1 == normalized[k + 1] for k in range(num_items - 1))

        if is_contiguous_slice:
            start = normalized[0]
            length = num_items

            if self.is_contiguous():
                base_start = start
                base_end = start + length

                sub_tensors = self.tensors[base_start:base_end]
                sub_validity = self.validity.slice(base_start, length) if self.validity is not None else None
                return TensorArray(sub_tensors, sub_validity, None)
            else:
                index_offset = start * 4
                index_length = length * 4
                sub_indices = self.indices.slice(index_offset, index_length)
                return TensorArray(self.tensors, self.validity, sub_indices)

        if self.is_contiguous():
            base_indices = normalized
        else:
            base_indices = [unpack_int32(self.indices, i) for i in normalized]

        new_indices = pack_int32(base_indices)
        return TensorArray(self.tensors, self.validity, new_indices)

    def to_list(self) -> List[Optional[torch.Tensor]]:
        """
        Convert the array into a Python list of tensors (and None for nulls).

        Returns:
            List[Optional[torch.Tensor]]: A list where each element is either:
                - None if the corresponding row is null, or
                - a torch.Tensor otherwise.

        Discussion:
            Q. How does to_list behave for views?
                It iterates over logical indices and uses self[i] for each position. Therefore indices-based
                mapping is applied and the returned list reflects the logical order of the view.

            Q. Why is this a materialization step?
                TensorArray may represent a view via indices. Converting to a Python list materializes the logical
                sequence into a concrete list of objects, which is convenient for debugging and interop.
        """
        output: List[Optional[torch.Tensor]] = []
        for i in range(self.length):
            if self.is_null(i):
                output.append(None)
            else:
                output.append(self[i])
        return output

    @classmethod
    def from_list(cls, data: List[Optional[torch.Tensor]]) -> "TensorArray":
        """
        Build a contiguous TensorArray from a Python list of torch.Tensor and/or None.

        Args:
            data (List[Optional[torch.Tensor]]): Input rows. Each element is either a torch.Tensor or None to
                represent a null row.

        Returns:
            TensorArray: A newly constructed contiguous TensorArray (indices=None) created by TensorArrayBuilder.

        Discussion:
            Q. Why use a builder here?
                The builder centralizes:
                    - validity accumulation (0/1 per row)
                    - tensor prototype checks for dtype/device/shape consistency
                    - final bitmap creation
                This keeps TensorArray construction rules consistent and reusable.
        """
        builder = TensorArrayBuilder()
        for x in data:
            builder.append(x)
        return builder.finish()


class TensorArrayBuilder(ArrayBuilder):
    """
    A builder for constructing a contiguous TensorArray.

    The builder maintains:
        - tensors: List[Optional[torch.Tensor]] base storage being accumulated
        - validity: List[int] of 0/1 flags (1 = valid tensor, 0 = null row)
        - prototype: Optional[torch.Tensor] used to enforce dtype/device/shape consistency

    Discussion:
        Q. Why enforce a prototype (dtype/device/shape) constraint?
            TensorArray represents a single column of tensors. Allowing mixed dtype/device/shape would make the
            column ill-defined for downstream operations that assume a consistent tensor schema. The builder
            therefore selects the first non-null tensor as a prototype and rejects inconsistent tensors.

        Q. How are null rows represented?
            Null rows are stored as:
                - validity bit 0
                - tensors element None
            This keeps row alignment simple and lets TensorArray return None on read without requiring sentinel
            tensors.
    """

    def __init__(self):
        """
        Initialize a TensorArrayBuilder.

        Discussion:
            Q. What state does the builder initialize?
                - tensors: empty list for accumulating Optional[torch.Tensor]
                - validity: empty list for accumulating 0/1 flags aligned with tensors
                - prototype: None until the first non-null tensor is appended
        """
        self.tensors: List[Optional[torch.Tensor]] = []
        self.validity: List[int] = []
        self.prototype: Optional[torch.Tensor] = None

    def check_and_set_prototype(self, value: torch.Tensor):
        """
        Check dtype/device/shape consistency against the prototype, initializing the prototype if needed.

        Args:
            value (torch.Tensor): A non-null tensor to validate (and potentially set as prototype).

        Discussion:
            Q. When is the prototype set?
                The first non-null tensor appended becomes the prototype. Subsequent tensors must match it.

            Q. Why check dtype/device/shape?
                These define the tensor schema for a column. Enforcing equality ensures that every row tensor is
                compatible as part of a single column representation.
        """
        if self.prototype is None:
            self.prototype = value
            return

        if value.dtype != self.prototype.dtype:
            raise TypeError(f"Inconsistent tensor dtype: expected {self.prototype.dtype}, got {value.dtype}")
        if value.device != self.prototype.device:
            raise TypeError(f"Inconsistent tensor device: expected {self.prototype.device}, got {value.device}")
        if value.shape != self.prototype.shape:
            raise TypeError(
                f"Inconsistent tensor shape: expected {tuple(self.prototype.shape)}, got {tuple(value.shape)}"
            )

    def append(self, value: Optional[torch.Tensor]) -> "TensorArrayBuilder":
        """
        Append one tensor row (torch.Tensor) or a null row (None) to the builder.

        Args:
            value (Optional[torch.Tensor]): A torch.Tensor to append, or None to append a null row.

        Returns:
            TensorArrayBuilder: Returns self to allow chaining.

        Discussion:
            Q. What happens when value is None?
                - validity appends 0
                - tensors appends None

            Q. What happens when value is a tensor?
                - check_and_set_prototype enforces dtype/device/shape consistency
                - validity appends 1
                - tensors appends the tensor object

            Q. Why store tensors directly instead of copying them?
                TensorArray treats tensors as row payload objects. Storing references preserves identity and avoids
                unnecessary copies; view semantics are handled via indices, not by duplicating tensors.
        """
        if value is None:
            self.validity.append(0)
            self.tensors.append(None)
            return self

        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"TensorArrayBuilder.append expects torch.Tensor or None, got {type(value).__name__}"
            )

        self.check_and_set_prototype(value)
        self.validity.append(1)
        self.tensors.append(value)
        return self

    def finish(self) -> TensorArray:
        """
        Finalize the builder and produce a contiguous TensorArray.

        Returns:
            TensorArray: A contiguous TensorArray (indices=None) constructed from the accumulated tensors and
            validity bitmap.

        Discussion:
            Q. Why validate len(tensors) == len(validity)?
                Validity stores one 0/1 bit per appended row. A mismatch indicates a builder logic error and would
                misalign null tracking with row storage.

            Q. How is validity stored in the final array?
                The builder creates a bitmap via Bitmap.from_list(self.validity). If there are no nulls, the bitmap
                may be returned as None (depending on Bitmap.from_list behavior).

            Q. Why does finish always return indices=None?
                Builders materialize a contiguous base array. Views are created later via take, which constructs
                indices buffers to represent selections without copying base storage.
        """
        if len(self.tensors) != len(self.validity):
            raise ValueError(
                f"TensorArrayBuilder internal length mismatch: "
                f"tensors={len(self.tensors)}, validity={len(self.validity)}"
            )

        validity_bitmap = Bitmap.from_list(self.validity)
        return TensorArray(self.tensors, validity_bitmap, None)