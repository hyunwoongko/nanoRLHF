from dataclasses import dataclass
from typing import Generic, Optional, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class ObjectRef(Generic[T]):
    """
    `ObjectRef` is an immutable handle that identifies a remotely stored object
    produced by a task or an actor method.

    The ref does not contain the value itself. It only stores metadata that allows
    the runtime to locate and fetch the value when `get()` is called.

    Attributes:
        object_id (str): A globally unique identifier for the object.
        owner_node_id (Optional[str]): The node currently owning (serving) the object.
            If None, it’s assumed local (single-node mode).
        size_bytes (Optional[int]): Optional size hint for the object payload.

    Examples:
        >>> ref = ObjectRef(object_id="obj-abc123", owner_node_id="node-A")
        >>> print(ref.object_id)
        'obj-abc123'
        >>> print(ref.owner_node_id)
        'node-A'

    Discussion:
        Q. Why is `ObjectRef` separate from the actual value?
            In a distributed system, the actual value might be stored on another node
            or spilled to disk. By exchanging a lightweight handle instead of raw data,
            tasks pass references (like pointers) without copying bytes. The runtime
            decides *when* and *how* to fetch efficiently.

        Q. Why make it immutable?
            `ObjectRef` flows across threads and nodes. If it were mutable, one worker
            could accidentally alter metadata (e.g., `owner_node_id`), causing cluster-wide
            inconsistencies. Immutability preserves safety and debuggability.

        Q. Is this like a `Future`?
            Similar idea (value not yet available), but `ObjectRef` also encodes
            *location/ownership*. Futures model readiness; `ObjectRef` models both
            readiness and placement.

        Q. What is an "owner" and why do we need it?
            The owner is the node responsible for serving the object’s bytes.
            - Single-node: `owner_node_id=None` (implicitly local).
            - Multi-node: `owner_node_id` tells the runtime where to route `get(ref)`.
            Ownership enables deterministic routing without broadcasting.
    """

    object_id: str
    owner_node_id: Optional[str] = None
    size_bytes: Optional[int] = None

    def is_local(self, current_node_id: Optional[str]) -> bool:
        """
        Check if this ref is local to the given node.

        Args:
            current_node_id (Optional[str]): The id of the node running the current code.

        Returns:
            bool: True if the object is considered local.

        Notes:
            - Single-node: `owner_node_id=None` is treated as local.
            - Multi-node: `owner_node_id` guides whether to fetch remotely from the owner.

        Examples:
            >>> ObjectRef("obj-1").is_local(None)      # single-node convention
            True
            >>> ObjectRef("obj-2", "node-A").is_local("node-A")
            True
            >>> ObjectRef("obj-3", "node-A").is_local("node-B")
            False
        """
        if self.owner_node_id is None:
            return True
        return current_node_id is not None and self.owner_node_id == current_node_id

    def short(self, n: int = 8) -> str:
        """
        Return a short string for logging/debugging.

        Args:
            n (int): Number of leading characters to keep.

        Returns:
            str: A shortened string like `"obj-abc1..."` for readable logs.
        """
        return f"{self.object_id[:n]}..."
