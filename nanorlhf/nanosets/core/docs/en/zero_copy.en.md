# Zero-Copy
This document explains Zero-copy.

## 1. What is Zero-Copy?

When we usually handle data in Python, it is common for data to be copied and a new object to be created. 
For example, when slicing lists, strings, bytes objects, etc., Python copies the original data and stores it in a new memory region.

For example, consider the following code.

```python
a = b"hello world"
b = a[0:5]  # A new bytes object b'hello' is created.
```

In this case, `b` is a new object that copies part of `a`, and they occupy different memory regions.
We can verify this with the `id()` function.

```python
print(id(a))  # 4387007312
print(id(b))  # 4387006544
```

However, since `b`'s data is a subset of `a`'s data, the same value `hello` ends up existing twice in memory.
Could we handle this more smartly? 

To solve this, we can use `memoryview`.
`memoryview` lets us reference a portion of the original data without copying it.
Therefore, the reference uses the same memory region as the original.

```python
a = b"hello world"
b = memoryview(a)[0:5]  # A memoryview object for b'hello' is created.
```

In this case, `b` does not copy part of `a` but references the original data, so it uses the same memory region.

```python
print(id(a))  # 4387007312
print(id(b.obj))  # 4387007312
```

We call this approach of referencing original data without copying it Zero-Copy,
and it can be very useful when processing large amounts of data.

## 2. Why is Zero-Copy important in Arrow-like systems?
In columnar data formats like Arrow, it is important to process large-scale data efficiently.
Zero-copy plays an important role in reducing the cost of copying data and improving performance in such systems.
For example, when processing large table data, if we can reference each column's data without copying it, we can greatly reduce memory usage and processing time.

## Zero-copy: 3-line summary
- Zero-copy means referencing data without copying it.
- Using Python's `memoryview`, you can reference a portion of the original data without copying it.
- In systems like Arrow, Zero-copy is important for improving the efficiency of large-scale data processing.
 important for improving the efficiency of large-scale data processing.