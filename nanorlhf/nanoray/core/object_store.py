import uuid
from concurrent.futures import Future
from typing import Any, Dict, Optional

from nanorlhf.nanoray.core.serialization import dumps, loads
from nanorlhf.nanoray.core.object_ref import ObjectRef


class ObjectStore:
    """
    `ObjectStore` is a key-value store that maps `ObjectRef` identifiers
    to their actual in-memory values.

    Each node maintains its own store instance. In a multi-node setup,
    other nodes must contact the *owner* node to fetch remote objects.

    Args:
        node_id (str): Identifier of the node owning this store.

    Examples:
        >>> # Single-node usage
        >>> store = ObjectStore(node_id="node-A")
        >>> ref = store.put(42)
        >>> ref.owner_node_id
        'node-A'
        >>> store.get(ref)
        42
        >>> store.has(ref)
        True
        >>> len(store)
        1

    Discussion:
        Q. Why is `ObjectStore` needed when we already have `ObjectRef`?
            `ObjectRef` is just a handle (metadata) telling *where* an object lives.
            `ObjectStore` actually *holds* the bytes in memory. Tasks return `ObjectRef`s
            that point to entries inside this store.

        Q. Why keep it node-local?
            Local stores allow nodes to operate independently and avoid global locks.
            Cross-node fetches (RPC, transfer, caching) are layered on top later.
    """

    def __init__(self, node_id: str):
        """
        Initialize the store for a given node.

        Args:
            node_id (str): Identifier of the node owning this store.
        """
        self.node_id = node_id
        self.store: Dict[str, Any] = {}
        self.sizes: Dict[str, int] = {}

    def __len__(self) -> int:
        """
        Return the number of stored objects.

        Returns:
            int: Count of locally stored objects.

        Examples:
            >>> store = ObjectStore("node-A")
            >>> _ = store.put("a")
            >>> _ = store.put("b")
            >>> len(store)
            2
        """
        return len(self.store)

    def has(self, ref_or_id: Any) -> bool:
        """
        Check whether the given object exists in this store.

        Args:
            ref_or_id (Any): Either `ObjectRef` or raw `object_id` string.

        Returns:
            bool: True if the object is present locally.

        Examples:
            >>> store = ObjectStore("node-A")
            >>> r = store.put(1)
            >>> store.has(r), store.has(r.object_id)
            (True, True)
        """
        oid = ref_or_id.object_id if isinstance(ref_or_id, ObjectRef) else str(ref_or_id)
        return oid in self.store

    def put(self, value: Any) -> ObjectRef:
        """
        Insert a value into the store and return an `ObjectRef`.

        Args:
            value (Any): The Python object to store.

        Returns:
            ObjectRef: A new reference pointing to the stored value (owned by this node).

        Examples:
            >>> store = ObjectStore("node-A")
            >>> r = store.put({"x": 1})
            >>> isinstance(r, ObjectRef), r.owner_node_id == "node-A"
            (True, True)
            >>> store.get(r)
            {'x': 1}

        Discussion:
            Q. What happens internally when we `put()`?
                - Generate a unique id (e.g., `"obj-<uuid>"`).
                - Store the object in the local dictionary under that id.
                - Return an `ObjectRef(object_id=id, owner_node_id=this_node)`.
        """
        object_id = f"obj-{uuid.uuid4().hex[:8]}"
        self.store[object_id] = value
        # NOTE: we DO NOT compute size here to avoid double-serialization cost.
        return ObjectRef(object_id=object_id, owner_node_id=self.node_id, size_bytes=None)

    def put_future(self, future: Future, object_id: Optional[str] = None) -> ObjectRef:
        """
        Insert a ``Future`` into the store and return an ``ObjectRef`` immediately.

        The future result will be resolved lazily on the first ``get``/``get_bytes``
        call. Once resolved, the stored value is replaced with the concrete result
        so subsequent reads are fast and do not re-wait the future.

        Args:
            future (Future): Future whose ``result()`` yields the value to store.
            object_id (Optional[str]): Optional stable object id. If omitted, a
                new id is generated.

        Returns:
            ObjectRef: Handle to the future-backed value.
        """

        oid = object_id or f"obj-{uuid.uuid4().hex[:8]}"
        self.store[oid] = future

        def materialize(f: Future):
            if f.cancelled():
                return
            try:
                value = f.result()
                self.store[oid] = value
            except Exception as exc:  # pragma: no cover - stored as-is
                self.store[oid] = exc

        future.add_done_callback(materialize)
        return ObjectRef(object_id=oid, owner_node_id=self.node_id, size_bytes=None)

    def get(self, ref: ObjectRef) -> Any:
        """
        Retrieve a value by its reference.

        Args:
            ref (ObjectRef): The handle whose value should be fetched.

        Returns:
            Any: The stored Python object.

        Raises:
            RuntimeError: If the object is owned by another node and this store
            has no networking to fetch it (teaching placeholder).

        Examples:
            >>> store = ObjectStore("node-A")
            >>> r = store.put(99)
            >>> store.get(r)
            99
        """
        if ref.object_id not in self.store:
            raise KeyError(f"Object not found locally: {ref.object_id}")

        value = self.store[ref.object_id]
        if isinstance(value, Future):
            value = value.result()
            self.store[ref.object_id] = value
        if isinstance(value, Exception):
            raise value
        return value

    def get_bytes(self, object_id: str) -> bytes:
        """
        Serialize and return the object payload as bytes

        Args:
            object_id (str): Local object id.

        Returns:
            bytes: The serialized payload

        Discussion:
            Q. when to use this method?
                - This is a low-level API for networking layers to fetch raw bytes.
                - Most users should use `get(ref)` which handles deserialization.
        """
        if object_id not in self.store:
            raise KeyError(f"Object not found locally: {object_id}")

        value = self.store[object_id]
        if isinstance(value, Future):
            value = value.result()
            self.store[object_id] = value
        if isinstance(value, Exception):
            raise value

        payload = dumps(value)
        self.sizes[object_id] = len(payload)
        return payload

    def put_bytes(self, payload: bytes) -> ObjectRef:
        """
        Deserialize bytes and store the object locally.

        Args:
            payload (bytes): Serialized object bytes.

        Returns:
            ObjectRef: Reference to the newly stored object.
                owner is this node.

        Examples:
            >>> store = ObjectStore("node-A")
            >>> data = dumps([1, 2, 3])
            >>> ref = store.put_bytes(data)
            >>> store.get(ref)
            [1, 2, 3]
        """
        value = loads(payload)
        ref = self.put(value)
        self.sizes[ref.object_id] = len(payload)
        return ObjectRef(object_id=ref.object_id, owner_node_id=self.node_id, size_bytes=len(payload))

    def get_size(self, object_id: str) -> Optional[int]:
        """
        Return the cached serialized size of an object if known.

        Args:
            object_id (str): Local object id.

        Returns:
            Optional[int]: Size in bytes if known, else None.
        """
        return self.sizes.get(object_id)

    def delete(self, object_id: str) -> None:
        """
        Delete an object by id (no-op if absent).

        Args:
            object_id (str): Id to remove.
        """
        self.store.pop(object_id, None)
