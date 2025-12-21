from typing import List, Optional, Any, Iterable, Dict

import torch

from nanorlhf.nanosets.dtype.array import ArrayBuilder
from nanorlhf.nanosets.dtype.dtype import PrimitiveType, DataType, FLOAT64, INT64, BOOL


def infer_primitive_dtype(values: List[Optional[PrimitiveType]]) -> DataType:
    saw_float = False
    saw_int = False
    saw_bool = False

    for value in values:
        if value is None:
            continue
        if isinstance(value, bool):
            saw_bool = True
            continue
        if isinstance(value, float):
            saw_float = True
        elif isinstance(value, int):
            saw_int = True
        else:
            raise ValueError(f"Unsupported primitive type: {type(value).__name__}")

    if saw_float:
        return FLOAT64
    if saw_int:
        return INT64
    if saw_bool:
        return BOOL

    raise ValueError("Cannot infer primitive dtype from empty or unsupported values")


def infer_child_builder(rows: List[Optional[Iterable[Any]]]) -> ArrayBuilder:
    from nanorlhf.nanosets.dtype.primitive_array import PrimitiveArrayBuilder
    from nanorlhf.nanosets.dtype.string_array import StringArrayBuilder
    from nanorlhf.nanosets.dtype.list_array import ListArrayBuilder
    from nanorlhf.nanosets.dtype.tensor_array import TensorArrayBuilder

    sample: Any = None
    for row in rows:
        if row is None:
            continue
        for element in row:
            if element is not None:
                sample = element
                break
        if sample is not None:
            break

    if sample is None:
        raise ValueError("Cannot infer element type: all rows are None or empty.")

    if isinstance(sample, (list, tuple)):
        inner_rows: List[Optional[Iterable[Any]]] = []
        for row in rows:
            if row is None:
                continue
            for sub in row:
                if sub is None:
                    inner_rows.append(None)
                elif isinstance(sub, (list, tuple)):
                    inner_rows.append(sub)
                else:
                    raise TypeError(f"Expected nested list elements, found {type(sub).__name__}")
        inner_child_builder = infer_child_builder(inner_rows)
        return ListArrayBuilder(inner_child_builder)

    if isinstance(sample, dict):
        dict_elements: List[Optional[Dict[str, Any]]] = []
        for row in rows:
            if row is None:
                continue
            for element in row:
                if element is None:
                    dict_elements.append(None)
                elif isinstance(element, dict):
                    dict_elements.append(element)
                else:
                    raise TypeError(f"Mixed element types: expected dict, got {type(element).__name__}")

        return get_struct_array_builder_from_rows(dict_elements)

    if isinstance(sample, str):
        for row in rows:
            if row is None:
                continue
            for element in row:
                if element is None:
                    continue
                if not isinstance(element, str):
                    raise TypeError(f"Mixed element types: expected str, got {type(element).__name__}")
        return StringArrayBuilder()

    if isinstance(sample, (bool, int, float)):
        prims: List[Optional[PrimitiveType]] = []
        for row in rows:
            if row is None:
                continue
            for element in row:
                if element is None:
                    prims.append(None)
                    continue
                if isinstance(element, (bool, int, float)):
                    prims.append(element)
                else:
                    raise TypeError(f"Mixed element types: expected primitive, got {type(element).__name__}")

        data_type = infer_primitive_dtype(prims)
        return PrimitiveArrayBuilder(data_type)

    if torch.is_tensor(sample):
        for row in rows:
            if row is None:
                continue
            for element in row:
                if element is None:
                    continue
                if not torch.is_tensor(element):
                    raise TypeError(f"Mixed element types: expected tensor-like, got {type(element).__name__}")
        return TensorArrayBuilder()

    raise TypeError(f"Unsupported element type for list: {type(sample).__name__}")


def get_struct_array_builder_from_rows(rows: List[Optional[Dict[str, Any]]]) -> "StructArrayBuilder":
    from nanorlhf.nanosets.dtype.struct_array import StructArrayBuilder

    inner_names: List[str] = []
    seen = set()
    for row in rows:
        if row is None:
            continue
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                inner_names.append(key)

    if not inner_names:
        return StructArrayBuilder([], [], strict_keys=False)

    num_rows = len(rows)
    inner_columns: Dict[str, List[Optional[Any]]] = {name: [None] * num_rows for name in inner_names}
    for index, row in enumerate(rows):
        if row is None:
            continue
        for name in inner_names:
            inner_columns[name][index] = row.get(name, None)

    inner_child_builders: List[ArrayBuilder] = []
    for name in inner_names:
        inner_builder = inference_builder_for_column(inner_columns[name])
        inner_child_builders.append(inner_builder)

    return StructArrayBuilder(inner_names, inner_child_builders, strict_keys=False)


def inference_builder_for_column(values: List[Optional[Any]]) -> ArrayBuilder:
    from nanorlhf.nanosets.dtype.primitive_array import PrimitiveArrayBuilder
    from nanorlhf.nanosets.dtype.string_array import StringArrayBuilder
    from nanorlhf.nanosets.dtype.list_array import ListArrayBuilder
    from nanorlhf.nanosets.dtype.tensor_array import TensorArrayBuilder

    sample: Any = None
    for v in values:
        if v is not None:
            sample = v
            break

    if sample is None:
        return StringArrayBuilder()

    if isinstance(sample, dict):
        for v in values:
            if v is None:
                continue
            if not isinstance(v, dict):
                raise TypeError("Mixed types in struct field: expected dict or None.")
        return get_struct_array_builder_from_rows(values)

    if isinstance(sample, (list, tuple)):
        for v in values:
            if v is None:
                continue
            if not isinstance(v, (list, tuple)):
                raise TypeError("Mixed types in list field: expected list/tuple or None.")

        child_builder = infer_child_builder(values)
        return ListArrayBuilder(child_builder)

    if isinstance(sample, str):
        for v in values:
            if v is None:
                continue
            if not isinstance(v, str):
                raise TypeError("Mixed types in string field: expected str or None.")
        return StringArrayBuilder()

    if isinstance(sample, (bool, int, float)):
        for v in values:
            if v is None:
                continue
            if not isinstance(v, (bool, int, float)):
                raise TypeError("Mixed types in primitive field: expected bool/int/float or None.")
        dtype = infer_primitive_dtype(values)  # type: ignore[arg-type]
        return PrimitiveArrayBuilder(dtype)

    if torch.is_tensor(sample):
        for v in values:
            if v is None:
                continue
            if not torch.is_tensor(v):
                raise TypeError("Mixed types in tensor field: expected tensor-like or None.")
        return TensorArrayBuilder()

    raise TypeError(f"Unsupported field type in struct: {type(sample).__name__}")
