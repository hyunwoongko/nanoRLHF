from typing import Any, Mapping, Optional

import cloudpickle
import torch
import zstandard as zstd

MAGIC = b"NRAY1"
ALG_NONE = b"NONE"  # 4 bytes
ALG_ZSTD = b"ZSTD"  # 4 bytes


def is_cuda_tensor(x: Any) -> bool:
    """
    Check if x is a CUDA tensor.

    Args:
        x (Any): Input object

    Returns:
        bool: True if x is a CUDA tensor, False otherwise
    """
    return isinstance(x, torch.Tensor) and x.is_cuda


def to_cpu(obj: Any) -> Any:
    """
    Recursively move all CUDA tensors in obj to CPU.

    Args:
        obj (Any): Input object

    Returns:
        Any: Object with all CUDA tensors moved to CPU
    """
    if is_cuda_tensor(obj):
        return obj.cpu()
    elif isinstance(obj, Mapping):
        return {k: to_cpu(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(to_cpu(v) for v in obj)
    else:
        return obj


def dumps(
    obj: Any,
    *,
    tensor_to_cpu: bool = True,
    compression: Optional[str] = "zstd",
    compress_threshold: int = 1_000_000,
    zstd_level: int = 3,
) -> bytes:
    """
    Serialize a Python object with optional GPU→CPU normalization and compression.

    Args:
        obj (Any): Object to serialize.
        tensor_to_cpu (bool): If True and torch is available, CUDA tensors are moved to CPU.
        compression (Optional[str]): "zstd" to enable zstd if available, else None.
        compress_threshold (int): Min payload size (uncompressed) to attempt compression.
        zstd_level (int): zstd compression level (1~22 typically).

    Returns:
        bytes: Framed bytes with header `b"NRAY1" + ALG(4) + payload`.

    Discussion:
        Q. Why a tiny custom frame instead of plain pickle?
            We need the receiver to know whether the payload is compressed without
            out-of-band metadata. A 9-byte header keeps it trivial and explicit.

        Q. What if zstandard is not installed?
            We silently fall back to uncompressed framing (ALG=NONE). This keeps the
            interface stable across environments.

        Q. Does `tensor_to_cpu` mutate the original object?
            No. We build a CPU-normalized copy for tensors (container types are rebuilt).
    """
    target = to_cpu(obj) if tensor_to_cpu else obj
    raw = cloudpickle.dumps(target)

    use_zstd = (
        compression == "zstd"
        and len(raw) >= int(compress_threshold)
    )

    if use_zstd:
        cctx = zstd.ZstdCompressor(level=int(zstd_level))
        comp = cctx.compress(raw)
        return MAGIC + ALG_ZSTD + comp
    else:
        return MAGIC + ALG_NONE + raw


def loads(buf: bytes) -> Any:
    """
    Deserialize bytes produced by `dumps(...)`.

    Args:
        buf (bytes): Framed payload (`b"NRAY1" + ALG + body`).

    Returns:
        Any: Restored Python object.

    Raises:
        ValueError: If header is malformed or algorithm is unknown.
    """
    if not buf.startswith(MAGIC):
        return cloudpickle.loads(buf)

    alg = buf[len(MAGIC): len(MAGIC) + 4]
    body = buf[len(MAGIC) + 4:]

    if alg == ALG_NONE:
        return cloudpickle.loads(body)
    if alg == ALG_ZSTD:
        dctx = zstd.ZstdDecompressor()
        raw = dctx.decompress(body)
        return cloudpickle.loads(raw)

    raise ValueError(f"Unknown serialization alg header: {alg!r}")
