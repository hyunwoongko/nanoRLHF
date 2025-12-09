import secrets
from typing import Final


DEFAULT_LEN: Final[int] = 8
# total hex chars after the prefix, e.g., "obj-<8 hex>"


def new_id(prefix: str = "id-", length: int = DEFAULT_LEN) -> str:
    """
    Generate a short hex id with a given prefix.

    Args:
        prefix (str): Prefix to prepend (e.g., "obj-", "act-", "tsk-").
        length (int): Number of hex characters after the prefix. Must be even.

    Returns:
        str: An id like "obj-08d41795" or "act-4be01de0".

    Examples:
        >>> new_id("obj-")
        'obj-1a2b3c4d'
        >>> new_id("act-", length=8)
        'act-09efab77'

    Discussion:
        Q. Why not `uuid4()`?
            UUIDs are long and noisy in logs. Short hex ids are easier to read.
            We use a short cryptographic token instead.

        Q. Is collision a risk?
            With 8 hex chars (32 bits), collisions are *possible* but unlikely
            for small-scale demos. If you’ll create millions of ids, bump
            `length` to 16 or more.

        Q. Why `secrets.token_hex`?
            It’s a standard, thread-safe generator for random bytes. We only
            need a simple, dependency-free source of randomness.

        Q. What about reproducibility?
            These ids are for human/logging convenience and routing keys,
            not for numeric experiments. If you require deterministic ids,
            replace this with a counter/seeded RNG in your environment.
    """
    if length % 2 != 0:
        raise ValueError("length must be even to represent whole bytes in hex.")
    return f"{prefix}{secrets.token_hex(length // 2)}"


def new_object_id() -> str:
    """Return an id like 'obj-XXXXXXXX' (8 hex)."""
    return new_id("obj-")


def new_actor_id() -> str:
    """Return an id like 'act-XXXXXXXX' (8 hex)."""
    return new_id("act-")


def new_task_id() -> str:
    """Return an id like 'tsk-XXXXXXXX' (8 hex)."""
    return new_id("tsk-")


def new_placement_group_id() -> str:
    """Return an id like 'pg-XXXXXXXX' (8 hex)."""
    return new_id("pg-")


def task_result_object_id(task_id: str) -> str:
    """
    Derive a stable ``object_id`` for a task result from its ``task_id``.

    This keeps bookkeeping straightforward when callers request blocking
    submission: we can deterministically match the produced ``ObjectRef`` to
    the originating task even if additional tasks are drained in the same
    pass.

    Args:
        task_id (str): The ``task.task_id`` identifier (e.g., ``"tsk-ab12cd34"``).

    Returns:
        str: An ``object_id`` using the ``obj-`` prefix, e.g.,
            ``"obj-ab12cd34"``.
    """

    suffix = task_id.split("tsk-", 1)[-1] if task_id.startswith("tsk-") else task_id
    return f"obj-{suffix}"
