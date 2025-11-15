from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

from nanorlhf.nanosets.core.bitmap import Bitmap
from nanorlhf.nanosets.core.buffer import Buffer
from nanorlhf.nanosets.dtype.array import Array, ArrayBuilder
from nanorlhf.nanosets.dtype.dtype import STRUCT
from nanorlhf.nanosets.dtype.dtype_inference import get_struct_array_builder_from_rows
from nanorlhf.nanosets.utils import normalize_index


class StructArray(Array):
    """
    Row-wise struct array.

    - 각 row는 dict(str -> value)로 표현 가능
    - 내부 저장:
        - field_names: List[str]
        - children: List[Array]   (각 필드별 column array)
        - validity: Bitmap or None (row가 통째로 null인지 여부)
    """

    def __init__(
        self,
        field_names: List[str],
        children: List[Array],
        validity: Optional[Bitmap] = None,
    ):
        if len(field_names) != len(children):
            raise ValueError(
                f"field_names length ({len(field_names)}) and children length ({len(children)}) "
                f"must match"
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
            raise ValueError(
                f"Validity bitmap length ({len(validity)}) does not match "
                f"struct length ({length})"
            )

        super().__init__(STRUCT, length, values=None, validity=validity, indices=None)

        self.field_names = field_names
        self.children = children
        self._name_to_index: Dict[str, int] = {name: i for i, name in enumerate(field_names)}

    def _check_field_index(self, idx: int) -> None:
        if not (0 <= idx < len(self.field_names)):
            raise IndexError(
                f"field index {idx} out of range [0, {len(self.field_names)})"
            )

    def field_index(self, name: str) -> int:
        return self._name_to_index[name]

    def __getitem__(self, key: Union[int, slice]) -> Union[Optional[Dict[str, Any]], "StructArray"]:
        if isinstance(key, int):
            if self.is_null(key):
                return None

            i = normalize_index(key, self.length)

            row: Dict[str, Any] = {}
            for name, child in zip(self.field_names, self.children):
                row[name] = child[i]
            return row

        if isinstance(key, slice):
            start, stop, step = key.indices(self.length)
            return self.take(range(start, stop, step))

        raise TypeError(f"Invalid index type for StructArray: {type(key).__name__}")

    def take(self, indices: Sequence[int]) -> "StructArray":
        num_items = len(indices)
        if num_items == 0:
            new_children = [child.take([]) for child in self.children]
            new_validity = None
            return StructArray(self.field_names, new_children, new_validity)

        normalized = [normalize_index(i, self.length) for i in indices]
        is_contiguous_slice = all(
            normalized[k] + 1 == normalized[k + 1] for k in range(num_items - 1)
        )

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
        builder = get_struct_array_builder_from_rows(rows)
        for row in rows:
            builder.append(row)
        return builder.finish()


class StructArrayBuilder(ArrayBuilder):

    def __init__(
        self,
        field_names: List[str],
        child_builders: List[ArrayBuilder],
        strict_keys: bool = False,
    ):
        if len(field_names) != len(child_builders):
            raise ValueError(
                f"field_names length ({len(field_names)}) and child_builders length "
                f"({len(child_builders)}) must match"
            )

        self.field_names = field_names
        self.child_builders = child_builders
        self.strict_keys = strict_keys
        self._name_to_index: Dict[str, int] = {name: i for i, name in enumerate(field_names)}

        self.validity: List[int] = []
        self.length: int = 0

    def append(self, row: Optional[Dict[str, Any]]) -> "StructArrayBuilder":
        if row is None:
            self.validity.append(0)
            for builder in self.child_builders:
                builder.append(None)
            self.length += 1
            return self

        if not isinstance(row, dict):
            raise TypeError(
                f"StructArrayBuilder.append expects dict or None, got {type(row).__name__}"
            )

        if self.strict_keys:
            for key in row.keys():
                if key not in self._name_to_index:
                    raise KeyError(f"Unexpected field name in struct row: {key!r}")

        self.validity.append(1)

        for name, builder in zip(self.field_names, self.child_builders):
            value = row.get(name, None)
            builder.append(value)

        self.length += 1
        return self

    def finish(self) -> StructArray:
        if self.length == 0:
            children: List[Array] = [b.finish() for b in self.child_builders]
            validity_bitmap: Optional[Bitmap] = None
            return StructArray(self.field_names, children, validity_bitmap)

        children: List[Array] = [b.finish() for b in self.child_builders]
        validity_bitmap = Bitmap.from_list(self.validity)

        return StructArray(self.field_names, children, validity_bitmap)
