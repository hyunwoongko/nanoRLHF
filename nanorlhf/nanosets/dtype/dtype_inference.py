from typing import List, Optional, Any, Iterable, Dict

import torch

from nanorlhf.nanosets.dtype.array import ArrayBuilder
from nanorlhf.nanosets.dtype.dtype import PrimitiveType, DataType, FLOAT64, INT64, BOOL


def infer_primitive_dtype(values: List[Optional[PrimitiveType]]) -> DataType:
    """
    Infer a primitive DataType from a list of Python primitive values (and/or None).

    Args:
        values (List[Optional[PrimitiveType]]): Input values consisting of bool/int/float and/or None.

    Returns:
        DataType: The inferred primitive dtype.
            - FLOAT64 if any non-null float is present.
            - INT64 if no float is present and any non-null int is present.
            - BOOL if only bool values (and/or None) are present.

    Discussion:
        Q. Why does float take precedence over int and bool?
            In mixed Python numeric inputs, floats imply fractional values and cannot be represented losslessly
            by integer types. Choosing FLOAT64 is a conservative default that preserves information.

        Q. Why does int take precedence over bool?
            Python bool is a subclass of int, but semantically it often represents a distinct boolean domain.
            If any non-bool int appears, the data is treated as integer-valued rather than strictly boolean.

        Q. What happens if values contains only None?
            There is no observed non-null value to infer from, so the function raises a ValueError.

        Q. Does this function distinguish between FLOAT32 and FLOAT64, or INT32 and INT64?
            No. The inference here selects FLOAT64 for floats and INT64 for ints as general-purpose defaults.
            Narrower types can be selected explicitly elsewhere if desired.
    """
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
    """
    Infer an appropriate child ArrayBuilder for a ListArray from nested iterable rows.

    Args:
        rows (List[Optional[Iterable[Any]]]): Input rows where each row is an iterable of elements or None.
            - None rows are treated as null lists and do not contribute element samples.
            - Empty iterables do not contribute element samples.

    Returns:
        ArrayBuilder: A builder suitable for accumulating all child elements of a ListArray.
            - ListArrayBuilder for nested lists/tuples (recursively inferred).
            - A struct builder (via get_struct_array_builder_from_rows) for dict elements.
            - StringArrayBuilder for str elements.
            - PrimitiveArrayBuilder for bool/int/float elements (dtype inferred by infer_primitive_dtype).
            - TensorArrayBuilder for torch tensor elements.

    Discussion:
        Q. Why does it search for a "sample" element first?
            The child builder must commit to a concrete element representation (primitive, string, list, struct, tensor).
            Scanning for the first non-null element provides a representative type to decide which builder to use.

        Q. What happens if all rows are None, empty, or contain only None elements?
            The function cannot infer a child type, so it raises ValueError. A caller can handle this by providing
            an explicit schema or choosing a default child type.

        Q. How does it handle nested lists?
            If the sample is a list/tuple, the function flattens one level into inner_rows, validating that every
            non-null element is itself a list/tuple (or None). It then recursively calls infer_child_builder to infer
            the builder for the next nesting level, and wraps it in a ListArrayBuilder.

        Q. How does it enforce type consistency for strings and tensors?
            After identifying a sample type (e.g., str or torch tensor), it scans all non-null elements and raises
            TypeError if any element violates the expected type. This prevents silently mixing incompatible element
            representations in a single ListArray child.

        Q. Why does it treat bool/int/float together for primitives?
            These are the Python scalar numeric/boolean primitives. The function collects them and delegates to
            infer_primitive_dtype to decide the final numeric dtype, then returns a PrimitiveArrayBuilder for that dtype.

        Q. Why do dict elements map to a struct builder?
            Dicts typically represent structured records with named fields. Delegating to a struct builder factory
            allows building a columnar representation of those records with appropriate field builders.

        Q. What if elements are of an unsupported type (e.g., custom objects)?
            The function raises TypeError to force an explicit decision, rather than guessing an encoding.
    """
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
