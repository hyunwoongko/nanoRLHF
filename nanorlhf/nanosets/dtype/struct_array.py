from typing import Any, Dict, List, Optional, Sequence, Union

from nanorlhf.nanosets.core.bitmap import Bitmap
from nanorlhf.nanosets.dtype.array import Array, ArrayBuilder
from nanorlhf.nanosets.dtype.dtype import STRUCT
from nanorlhf.nanosets.dtype.dtype_inference import get_struct_array_builder_from_rows
from nanorlhf.nanosets.utils import normalize_index


class StructArray(Array):
    """
    An Arrow-style struct array that stores rows in a columnar layout: one child Array per field,
    plus an optional validity bitmap for row-level nulls.

    Instead of storing each row as a Python dict object, this array keeps:
        - `field_names`: the ordered field schema
        - `children`: per-field column arrays aligned by row index
        - `validity`: a Bitmap indicating whether each row is valid (1) or null (0)

    Notes:
        - All children must have the same length, which defines the struct length.
        - Row nullness is represented at the struct level via `validity`. If a row is null, `__getitem__`
          returns None for that row.

    Discussion:
        Q. Why store struct rows as field-wise children (columnar) instead of dicts per row?
            A columnar layout keeps values of the same field contiguous, which is often better for field-wise
            processing, cache locality, and type-specific storage (e.g., a StringArray for names and a PrimitiveArray
            for ages). Row selection (take) can be implemented by applying the same indices to every child, preserving
            row alignment.

        Q. What does validity mean for a struct row?
            Validity is row-level: validity[i] == 0 means the entire struct row i is null. In that case, reading
            row i returns None regardless of what the children might contain at that index.

        Q. Why require all children to have the same length?
            Each struct row is formed by taking the i-th element from every child. If child lengths differ, row
            alignment breaks and row materialization becomes ill-defined, so the constructor enforces this invariant.
    """

    def __init__(self, field_names: List[str], children: List[Array], validity: Optional[Bitmap] = None):
        """
        Initialize a StructArray.

        Args:
            field_names (List[str]): Ordered list of field names.
            children (List[Array]): Ordered list of child arrays, where children[i] stores the column for
                field_names[i]. All children must have identical length.
            validity (Optional[Bitmap]): Optional row-level validity bitmap (1 = valid, 0 = null). If None, all rows
                are treated as valid.

        Discussion:
            Q. How is the struct length determined?
                If children are present, length is defined as len(children[0]). The constructor then verifies that
                every other child has the same length. If there are no children, length is 0.

            Q. Why does the base Array store values=None?
                A struct does not have a single flat values buffer; its payload is the set of child arrays. Therefore
                values are stored as None at the base Array level, and the payload is accessed via `children`.
        """
        if len(field_names) != len(children):
            raise ValueError(
                f"field_names length ({len(field_names)}) and children length ({len(children)}) " f"must match"
            )

        if children:
            length = len(children[0])
            for i, child in enumerate(children):
                if len(child) != length:
                    raise ValueError(
                        f"All child arrays must have the same length; "
                        f"child[{i}] has length {len(child)}, expected {length}"
                    )
        else:
            length = 0

        if validity is not None and len(validity) != length:
            raise ValueError(f"Validity bitmap length ({len(validity)}) does not match " f"struct length ({length})")

        super().__init__(STRUCT, length, values=None, validity=validity, indices=None)

        self.field_names = field_names
        self.children = children
        self.name_to_index: Dict[str, int] = {name: i for i, name in enumerate(field_names)}

    def check_field_index(self, idx: int):
        """
        Validate a field index against the struct schema.

        Args:
            idx (int): Field index to validate.

        Discussion:
            Q. Why have an explicit index checker?
                Some APIs may accept numeric field indices. Centralizing bounds checks keeps error messages consistent
                and prevents accidental access outside the schema range.
        """
        if not (0 <= idx < len(self.field_names)):
            raise IndexError(f"field index {idx} out of range [0, {len(self.field_names)})")

    def field_index(self, name: str) -> int:
        """
        Resolve a field name to its child index.

        Args:
            name (str): Field name.

        Returns:
            int: The index i such that field_names[i] == name.

        Discussion:
            Q. Why store a name-to-index mapping?
                It provides O(1) lookup for field indices, avoiding repeated linear scans over field_names when
                resolving fields by name.
        """
        return self.name_to_index[name]

    def __getitem__(self, key: Union[int, slice]) -> Union[Optional[Dict[str, Any]], "StructArray"]:
        """
        Read a row (int index) or return a selection (slice).

        Args:
            key (Union[int, slice]):
                - int: returns a single row as a Python dict {field_name: value}, or None if the row is null.
                - slice: returns a new StructArray corresponding to the slice, implemented via `take`.

        Returns:
            Union[Optional[Dict[str, Any]], StructArray]:
                - If key is int: a dict mapping field names to values for that row, or None.
                - If key is slice: a StructArray selection.

        Discussion:
            Q. How does integer indexing materialize a row?
                If the row is not null, it normalizes the index and then reads child[normalized_idx] for each
                (name, child) pair, assembling the results into a Python dict.

            Q. Why return None when the row is null?
                Struct validity is row-level. If validity marks row i as null, the struct row is considered absent,
                so __getitem__ returns None rather than a dict with per-field Nones.

            Q. Why delegate slicing to take(range(...))?
                Using take for slicing keeps selection semantics consistent across arrays and centralizes the logic
                for handling validity and child alignment under selection.
        """
        if isinstance(key, int):
            if self.is_null(key):
                return None

            normalized_idx = normalize_index(key, self.length)

            row: Dict[str, Any] = {}
            for name, child in zip(self.field_names, self.children):
                row[name] = child[normalized_idx]
            return row

        if isinstance(key, slice):
            start, stop, step = key.indices(self.length)
            return self.take(range(start, stop, step))

        raise TypeError(f"Invalid index type for StructArray: {type(key).__name__}")

    def take(self, indices: Sequence[int]) -> "StructArray":
        """
        Select rows by logical indices and return a new StructArray.

        Args:
            indices (Sequence[int]): Sequence of logical row indices to take. Negative indices are allowed and
                are normalized.

        Returns:
            StructArray: A StructArray containing the selected rows, with all children taken using the same
            normalized indices so row alignment is preserved.

        Discussion:
            Q. What are "logical indices" here?
                They are indices in the coordinate system of this StructArray: 0..self.length-1, including support
                for Python negative indices which are normalized to that range.

            Q. Why apply the same take to every child?
                Each struct row is defined by the i-th elements across all children. Applying the same selection to
                each child preserves that alignment for the result.

            Q. How is validity handled under selection?
                - If self.validity is None, new_validity stays None (all rows valid).
                - If the selection is contiguous, it slices the bitmap for efficiency.
                - If the selection is non-contiguous, it reconstructs a new bitmap by checking is_null for each
                  selected index.
        """
        num_items = len(indices)
        if num_items == 0:
            new_children = [child.take([]) for child in self.children]
            new_validity = None
            return StructArray(self.field_names, new_children, new_validity)

        normalized = [normalize_index(i, self.length) for i in indices]
        is_contiguous_slice = all(normalized[k] + 1 == normalized[k + 1] for k in range(num_items - 1))

        if self.validity is None:
            new_validity = None
        else:
            if is_contiguous_slice:
                start = normalized[0]
                new_validity = self.validity.slice(start, num_items)
            else:
                bits: List[int] = []
                for src_i in normalized:
                    bits.append(0 if self.is_null(src_i) else 1)
                new_validity = Bitmap.from_list(bits)

        new_children = [child.take(normalized) for child in self.children]
        return StructArray(self.field_names, new_children, new_validity)

    def to_list(self) -> List[Optional[Dict[str, Any]]]:
        """
        Convert the StructArray into a Python list of dicts (and None for null rows).

        Returns:
            List[Optional[Dict[str, Any]]]: A list where each element is either:
                - None if the corresponding row is null, or
                - a dict mapping field names to row values otherwise.

        Discussion:
            Q. Why is this a materialization step?
                The struct payload is stored column-wise in child arrays. Converting to Python dicts allocates
                per-row objects and loses the compact Arrow-style representation, but is convenient for interop
                and debugging.

            Q. How does it interpret null rows?
                It checks row validity via is_null(i). If null, it appends None; otherwise it constructs a dict
                by reading each child at index i.
        """
        output = []
        for i in range(self.length):
            if self.is_null(i):
                output.append(None)
            else:
                row: Dict[str, Any] = {}
                for name, child in zip(self.field_names, self.children):
                    row[name] = child[i]
                output.append(row)
        return output

    @classmethod
    def from_list(
        cls,
        rows: List[Optional[Dict[str, Any]]],
        strict_keys: bool = False,
    ) -> "StructArray":
        """
        Build a StructArray from Python rows (dicts) and/or None.

        Args:
            rows (List[Optional[Dict[str, Any]]]): Input rows. Each element is either a dict mapping field names
                to values, or None to represent a null struct row.
            strict_keys (bool): Present for API symmetry; schema enforcement is handled by builders. This argument
                is not used in this method body.

        Returns:
            StructArray: A newly constructed StructArray where field schema and per-field builders are inferred
            from the rows via get_struct_array_builder_from_rows.

        Discussion:
            Q. How are field names determined from rows?
                get_struct_array_builder_from_rows scans the rows and collects keys in first-seen order to form
                the struct schema.

            Q. How are per-field types determined?
                Builders are inferred per column. Each field column is assembled and a builder is chosen based on
                a representative sample value, while rejecting mixed types.

            Q. Why use a builder instead of constructing children directly here?
                Building needs to:
                - append per-field values into appropriate child builders,
                - keep all children aligned by row,
                - record row validity consistently.
                The builder encapsulates that logic and returns a consistent StructArray in finish().
        """
        builder = get_struct_array_builder_from_rows(rows)
        for row in rows:
            builder.append(row)
        return builder.finish()


class StructArrayBuilder(ArrayBuilder):
    """
    A builder for constructing a StructArray in a columnar layout.

    The builder maintains:
        - `field_names`: ordered schema
        - `child_builders`: one ArrayBuilder per field, accumulating column values
        - `validity`: row-level 0/1 flags (1 = valid row dict, 0 = null row)
        - `length`: number of appended rows
        - `strict_keys`: whether to reject unexpected keys at append time

    Discussion:
        Q. Why append None to every child builder when a struct row is None?
            Even if the struct row is null, children must remain the same length to preserve row alignment.
            Appending None to each child ensures every child advances one row.

        Q. What does strict_keys do?
            If enabled, append(row) checks that every key in the input row dict exists in the schema, raising a
            KeyError for unexpected fields. Missing fields are still allowed (they become None via row.get).
    """

    def __init__(
        self,
        field_names: List[str],
        child_builders: List[ArrayBuilder],
        strict_keys: bool = False,
    ):
        """
        Initialize a StructArrayBuilder.

        Args:
            field_names (List[str]): Ordered field names (schema).
            child_builders (List[ArrayBuilder]): Per-field builders aligned with field_names.
            strict_keys (bool): If True, reject unexpected keys in appended row dicts.

        Discussion:
            Q. Why must field_names and child_builders have the same length?
                The schema is positional: field_names[i] corresponds to child_builders[i]. A mismatch would make
                append ambiguous and break field alignment.
        """
        if len(field_names) != len(child_builders):
            raise ValueError(
                f"field_names length ({len(field_names)}) and child_builders length "
                f"({len(child_builders)}) must match"
            )

        self.field_names = field_names
        self.child_builders = child_builders
        self.strict_keys = strict_keys
        self.name_to_index: Dict[str, int] = {name: i for i, name in enumerate(field_names)}

        self.validity: List[int] = []
        self.length: int = 0

    def append(self, row: Optional[Dict[str, Any]]) -> "StructArrayBuilder":
        """
        Append one struct row (dict) or a null row (None) to the builder.

        Args:
            row (Optional[Dict[str, Any]]): A dict mapping field names to values, or None to represent a null row.

        Returns:
            StructArrayBuilder: Returns self to allow chaining.

        Discussion:
            Q. What happens when row is None?
                - validity appends 0
                - every child builder receives append(None) to keep child lengths aligned
                - length increments by 1

            Q. What happens when row is a dict?
                - if strict_keys is enabled, unexpected keys are rejected
                - validity appends 1
                - for each field name, value = row.get(name, None) is appended to the corresponding child builder
                - length increments by 1

            Q. Why use row.get(name, None) rather than requiring keys to exist?
                Missing fields are treated as nulls at the field level. This supports sparse dict rows while keeping
                a consistent schema.
        """
        if row is None:
            self.validity.append(0)
            for builder in self.child_builders:
                builder.append(None)
            self.length += 1
            return self

        if not isinstance(row, dict):
            raise TypeError(f"StructArrayBuilder.append expects dict or None, got {type(row).__name__}")

        if self.strict_keys:
            for key in row.keys():
                if key not in self.name_to_index:
                    raise KeyError(f"Unexpected field name in struct row: {key!r}")

        self.validity.append(1)

        for name, builder in zip(self.field_names, self.child_builders):
            value = row.get(name, None)
            builder.append(value)

        self.length += 1
        return self

    def finish(self) -> StructArray:
        """
        Finalize the builder and produce a StructArray.

        Returns:
            StructArray: A StructArray constructed from finished child arrays and the accumulated validity.
                If length is 0, validity is returned as None.

        Discussion:
            Q. Why return validity=None when length is 0?
                An empty array has no rows, so a bitmap is unnecessary. Returning None avoids allocating a bitmap
                while preserving the meaning that all (non-existent) rows are valid.

            Q. How are the final children produced?
                Each child builder is finalized via b.finish() and collected in order corresponding to field_names.
                Validity is built from the accumulated 0/1 list via Bitmap.from_list(self.validity).
        """
        if self.length == 0:
            children: List[Array] = [b.finish() for b in self.child_builders]
            validity_bitmap: Optional[Bitmap] = None
            return StructArray(self.field_names, children, validity_bitmap)

        children: List[Array] = [b.finish() for b in self.child_builders]
        validity_bitmap = Bitmap.from_list(self.validity)

        return StructArray(self.field_names, children, validity_bitmap)