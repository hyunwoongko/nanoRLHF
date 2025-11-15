import pytest

from nanorlhf.nanosets.core.buffer import Buffer
from nanorlhf.nanosets.dtype.dtype import (
    BOOL, INT32, INT64, FLOAT64, INT32_MIN, INT32_MAX,
)
from nanorlhf.nanosets.dtype.primitive_array import PrimitiveArray, PrimitiveArrayBuilder, \
    infer_primitive_dtype


def test_infer_dtype_int_float_bool():
    assert infer_primitive_dtype([1, 2, None]) is INT64
    assert infer_primitive_dtype([1.0, 2, True, None]) is FLOAT64
    assert infer_primitive_dtype([True, False, None]) is BOOL
    with pytest.raises(ValueError):
        infer_primitive_dtype([None, None])


def test_builder_finish_and_to_list_int64():
    data = [1, None, 3, 4]
    b = PrimitiveArrayBuilder(INT64)
    for v in data:
        b.append(v)
    arr = b.finish()
    assert len(arr) == 4
    assert arr.to_list() == data


def test_builder_finish_and_to_list_float64():
    data = [1.5, None, 3.25]
    b = PrimitiveArrayBuilder(FLOAT64)
    for v in data:
        b.append(v)
    arr = b.finish()
    assert arr.to_list() == data


def test_builder_int32_range_check():
    b = PrimitiveArrayBuilder(INT32)
    b.append(INT32_MIN)
    b.append(INT32_MAX)
    arr = b.finish()
    assert arr.to_list() == [INT32_MIN, INT32_MAX]
    # overflow should raise in from_list path with explicit dtype
    with pytest.raises(OverflowError):
        PrimitiveArray.from_list([INT32_MAX + 1], dtype=INT32)


def test_scalar_getitem_and_negative_index():
    arr = PrimitiveArray.from_list([10, 20, None, 40], dtype=INT64)
    assert arr[0] == 10
    assert arr[1] == 20
    assert arr[2] is None
    assert arr[-1] == 40
    with pytest.raises(IndexError):
        _ = arr[999]


def test_contiguous_slice_zero_copy_values_and_validity():
    arr = PrimitiveArray.from_list([0, 1, None, 3, 4, 5, None, 7], dtype=INT64)
    # slice 2..6  => [None,3,4,5]
    sub = arr[2:6]
    assert isinstance(sub, PrimitiveArray)
    assert sub.to_list() == [None, 3, 4, 5]
    # values buffer should be a zero-copy view (same underlying object)
    assert sub.indices is None
    assert sub.values.data.obj is arr.values.data.obj
    # validity zero-copy view (if present)
    assert sub.validity is not None and arr.validity is not None
    assert sub.validity.buffer.data.obj is arr.validity.buffer.data.obj


def test_strided_slice_indirect_indices():
    arr = PrimitiveArray.from_list([0, 1, 2, 3, 4, 5, 6, 7], dtype=INT64)
    sub = arr[1:8:2]  # [1,3,5,7]
    assert isinstance(sub, PrimitiveArray)
    assert sub.to_list() == [1, 3, 5, 7]
    assert sub.indices is not None
    # values buffer is shared
    assert sub.values.data.obj is arr.values.data.obj


# ---------------------------
# take() contiguous run
# ---------------------------

def test_take_contiguous_run_contiguous_array():
    arr = PrimitiveArray.from_list([10, 11, 12, 13, 14, 15], dtype=INT64)
    sub = arr.take(range(2, 5))
    assert sub.indices is None
    assert sub.to_list() == [12, 13, 14]
    assert sub.values.data.obj is arr.values.data.obj


# ---------------------------
# take() general (non-contiguous)
# ---------------------------

def test_take_general_noncontiguous_indirect():
    arr = PrimitiveArray.from_list([10, 11, 12, 13, 14, 15], dtype=INT64)
    sub = arr.take([4, 0, 5])  # [14,10,15]
    assert sub.indices is not None
    assert sub.to_list() == [14, 10, 15]
    assert sub.values.data.obj is arr.values.data.obj


# ---------------------------
# indirect view then contiguous slice on indices buffer
# ---------------------------

def test_indirect_then_contiguous_slice_on_indices():
    arr = PrimitiveArray.from_list([0, 1, 2, 3, 4, 5, 6, 7], dtype=INT64)
    mid = arr.take([4, 6, 7, 5])
    assert mid.indices is not None

    sub = mid[1:3]
    assert sub.to_list() == [6, 7]
    assert sub.indices is not None
    assert sub.indices.data.obj is mid.indices.data.obj
    assert sub.values.data.obj is arr.values.data.obj


# ---------------------------
# null checks
# ---------------------------

def test_is_null_behavior():
    arr = PrimitiveArray.from_list([None, 1, None, 3], dtype=INT64)
    assert arr.is_null(0) is True
    assert arr.is_null(1) is False
    assert arr.is_null(2) is True
    assert arr.is_null(3) is False
    with pytest.raises(IndexError):
        arr.is_null(999)


# ---------------------------
# float and bool round-trip
# ---------------------------

def test_float_roundtrip_and_bool_roundtrip():
    f = PrimitiveArray.from_list([1.0, None, 2, True], dtype=FLOAT64)
    assert f.to_list() == [1.0, None, 2.0, 1.0]  # bool→float cast to 1.0
    b = PrimitiveArray.from_list([True, False, None], dtype=BOOL)
    assert b.to_list() == [True, False, None]


# ---------------------------
# constructor size check (contiguous)
# ---------------------------

def test_constructor_values_size_mismatch_raises():
    # craft a wrong-sized values buffer for length=3 of INT64 (should be 24 bytes, we give 8)
    wrong_raw = bytearray(8)
    values = Buffer.from_bytearray(wrong_raw)
    with pytest.raises(ValueError):
        PrimitiveArray(dtype=INT64, values=values, length=3, validity=None, indices=None)


# ---------------------------
# normalize_index edge cases
# ---------------------------

def test_normalize_index_out_of_range():
    arr = PrimitiveArray.from_list([0, 1, 2], dtype=INT64)
    with pytest.raises(IndexError):
        _ = arr[-4]
    with pytest.raises(IndexError):
        _ = arr[3]


# ---------------------------
# slice empty paths
# ---------------------------

def test_empty_slice_and_empty_take():
    arr = PrimitiveArray.from_list([0, 1, 2, 3], dtype=INT64)
    sub1 = arr[2:2]  # empty
    assert isinstance(sub1, PrimitiveArray)
    assert len(sub1) == 0
    assert sub1.to_list() == []
    sub2 = arr.take([])
    assert len(sub2) == 0
    assert sub2.to_list() == []


# ---------------------------
# performance sanity: avoid copying on contiguous slice
# ---------------------------

def test_values_zero_copy_memoryview_identity_on_slice():
    arr = PrimitiveArray.from_list(list(range(100)), dtype=INT64)
    sub = arr[10:90]
    assert sub.indices is None
    # same underlying object for memoryview.obj (bytearray from builder)
    assert sub.values.data.obj is arr.values.data.obj


def test_double_strided_slice_equivalence_and_zerocopy_values():
    arr = PrimitiveArray.from_list(list(range(30)), dtype=INT64)

    mid = arr[::2]
    fin = mid[::3]

    direct = arr[::6]

    assert fin.to_list() == direct.to_list() == [0, 6, 12, 18, 24]

    assert mid.values.data.obj is arr.values.data.obj
    assert fin.values.data.obj is arr.values.data.obj

    assert mid.indices is not None
    assert fin.indices is not None
    assert fin.indices.data.obj is not mid.indices.data.obj


def test_contiguous_then_slice_of_indices_is_zerocopy_on_indices():
    arr = PrimitiveArray.from_list(list(range(10)), dtype=INT64)

    mid = arr.take([2, 5, 6, 7, 8])
    assert mid.indices is not None

    sub = mid[1:4]
    assert sub.to_list() == [5, 6, 7]
    assert sub.values.data.obj is arr.values.data.obj
    assert sub.indices is not None
    assert sub.indices.data.obj is mid.indices.data.obj  # indices도 제로카피


if __name__ == '__main__':
    print("test_infer_dtype_int_float_bool")
    test_infer_dtype_int_float_bool()
    print("test_builder_finish_and_to_list_int64")
    test_builder_finish_and_to_list_int64()
    print("test_builder_finish_and_to_list_float64")
    test_builder_finish_and_to_list_float64()
    print("test_builder_int32_range_check")
    test_builder_int32_range_check()
    print("test_scalar_getitem_and_negative_index")
    test_scalar_getitem_and_negative_index()
    print("test_contiguous_slice_zero_copy_values_and_validity")
    test_contiguous_slice_zero_copy_values_and_validity()
    print("test_strided_slice_indirect_indices")
    test_strided_slice_indirect_indices()
    print("test_take_contiguous_run_contiguous_array")
    test_take_contiguous_run_contiguous_array()
    print("test_take_general_noncontiguous_indirect")
    test_take_general_noncontiguous_indirect()
    print("test_indirect_then_contiguous_slice_on_indices")
    test_indirect_then_contiguous_slice_on_indices()
    print("test_is_null_behavior")
    test_is_null_behavior()
    print("test_float_roundtrip_and_bool_roundtrip")
    test_float_roundtrip_and_bool_roundtrip()
    print("test_constructor_values_size_mismatch_raises")
    test_constructor_values_size_mismatch_raises()
    print("test_normalize_index_out_of_range")
    test_normalize_index_out_of_range()
    print("test_empty_slice_and_empty_take")
    test_empty_slice_and_empty_take()
    print("test_values_zero_copy_memoryview_identity_on_slice")
    test_values_zero_copy_memoryview_identity_on_slice()
    print("test_double_strided_slice_equivalence_and_zerocopy_values")
    test_double_strided_slice_equivalence_and_zerocopy_values()
    print("test_contiguous_then_slice_of_indices_is_zerocopy_on_indices")
    test_contiguous_then_slice_of_indices_is_zerocopy_on_indices()
    print("All tests passed.")
