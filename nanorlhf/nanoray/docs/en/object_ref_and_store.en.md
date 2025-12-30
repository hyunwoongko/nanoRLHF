# ObjectRef & ObjectStore

When you first try Ray, you usually run into a situation like this. You run a remote function, but instead of getting a number or a dictionary, something like an ObjectRef comes back. Even if you print it, you can’t see the value right away, and at first it’s not obvious why you need to call ray.get.

In a distributed execution framework, computation may run in another process or on another machine, and the result value may also live in memory over there. So rather than immediately copying the value back, Ray is designed to return a handle that points to the value first, and fetch it only when needed.

In Ray, this handle is called an ObjectRef, and the in-memory space that stores the actual value is called an ObjectStore. nanoray follows the same core idea, and this document focuses on understanding that principle.

## 1. Why does a remote call return a ref instead of a value?

### Intuition in local Python
In local Python, the following feels natural.
When a function returns a value, the value itself comes back.

```python
x = f()
# x is the value itself.
```

### Intuition in a distributed environment
In Ray, a remote call typically returns an ObjectRef instead of the value. Since the value may live elsewhere, it returns a reference first, and the runtime fetches the value when needed.

```python
ref = f.remote()
# ref is an ObjectRef. It is not the value.
x = ray.get(ref)
# You must call ray.get to fetch the actual value.
```

## 2. ObjectRef: a handle that points to a value, not the value itself

ObjectRef is a reference that points to some object. The key point is that ObjectRef does not contain the value. It only carries minimal metadata that helps the runtime locate and fetch the value.

```python
from dataclasses import dataclass
from typing import Generic, Optional, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class ObjectRef(Generic[T]):
    object_id: str
    owner_node_id: Optional[str] = None
    size_bytes: Optional[int] = None

    def is_local(self, current_node_id: Optional[str]) -> bool:
        if self.owner_node_id is None:
            return True
        return current_node_id is not None and self.owner_node_id == current_node_id

    def short(self, n: int = 8) -> str:
        return f"{self.object_id[:n]}..."
```

### Why is ObjectRef needed?
Returning the value immediately might look simplest, but in a distributed environment it can be very expensive at that moment. For example, if you copy a large tensor or a large list back to the driver every time, network bandwidth and memory can quickly become bottlenecks. So Ray keeps results in the ObjectStore by default, and lets users fetch them only when needed via ObjectRef.

### What is owner?
In a multi-node environment, each node has its own ObjectStore. The owner_node_id in ObjectRef is a hint for which node’s ObjectStore holds the object, and the runtime uses it to decide where to route get(ref).

### What does is_local mean?
When you call get(ref), if it’s local you can read it immediately; otherwise you must request it from the owner. is_local represents that branching condition.

```python
ref1 = ObjectRef("obj-1")
print(ref1.is_local(None))  # True (single-node convention)

ref2 = ObjectRef("obj-2", "node-A")
print(ref2.is_local("node-A"))  # True
print(ref2.is_local("node-B"))  # False
```

## 3. ObjectStore: a storage that holds the actual values

ObjectStore stores actual Python objects using object_id as the key. If ObjectRef is a label, ObjectStore is the warehouse where the labeled items are stored.

```python
import uuid
from concurrent.futures import Future
from typing import Any, Dict, Optional

from nanorlhf.nanoray.core.serialization import dumps, loads
from nanorlhf.nanoray.core.object_ref import ObjectRef


class ObjectStore:
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.store: Dict[str, Any] = {}
        self.sizes: Dict[str, int] = {}

    def __len__(self) -> int:
        return len(self.store)

    def has(self, ref_or_id: Any) -> bool:
        oid = ref_or_id.object_id if isinstance(ref_or_id, ObjectRef) else str(ref_or_id)
        return oid in self.store

    def put(self, value: Any) -> ObjectRef:
        object_id = f"obj-{uuid.uuid4().hex[:8]}"
        self.store[object_id] = value
        return ObjectRef(object_id=object_id, owner_node_id=self.node_id, size_bytes=None)

    def put_future(self, future: Future, object_id: Optional[str] = None) -> ObjectRef:
        oid = object_id or f"obj-{uuid.uuid4().hex[:8]}"
        self.store[oid] = future

        def materialize(f: Future):
            if f.cancelled():
                return
            try:
                value = f.result()
                self.store[oid] = value
            except Exception as exc:
                self.store[oid] = exc

        future.add_done_callback(materialize)
        return ObjectRef(object_id=oid, owner_node_id=self.node_id, size_bytes=None)

    def get(self, ref: ObjectRef) -> Any:
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
        value = loads(payload)
        ref = self.put(value)
        self.sizes[ref.object_id] = len(payload)
        return ObjectRef(object_id=ref.object_id, owner_node_id=self.node_id, size_bytes=len(payload))

    def get_size(self, object_id: str) -> Optional[int]:
        return self.sizes.get(object_id)

    def delete(self, object_id: str) -> None:
        self.store.pop(object_id, None)
```

## 4. Behavior example

```python
store = ObjectStore("node-A")

ref = store.put({"x": 1})
print(ref.object_id)      # e.g., "obj-a1b2c3d4"
print(ref.owner_node_id)  # "node-A"

value = store.get(ref)
print(value)  # {"x": 1}
```

The key point is that ObjectStore.put stores the value and returns an ObjectRef that points to it. The flow in Ray where a remote task returns an ObjectRef instead of a value follows the same principle.

## 5. Why is the bytes API needed?
In a multi-node setup, you may need to fetch a value from another node’s ObjectStore. Since you can’t send a Python object over the network as-is, you need APIs that send and receive a serialized bytes form.

```python
store_1 = ObjectStore("node-A")
ref_1 = storeA.put([1, 2, 3])

payload = store_1.get_bytes(ref_1.object_id)

store_b = ObjectStore("node-B")
ref_b = store_b.put_bytes(payload)

print(storeB.get(ref_b))  # [1, 2, 3]
```

These get_bytes / put_bytes APIs are not really a high-level user API; they are closer to low-level building blocks needed to implement remote fetch in Ray.

## 6. Why store Futures?
In distributed systems, the result may not be ready yet. Even then, it is useful to return a ref immediately, and then reflect the real value into the store once it becomes available.

```python
from concurrent.futures import Future

store = ObjectStore("node-A")

f = Future()
ref = store.put_future(f, object_id="obj-fixed")
print(ref.object_id)  # "obj-fixed"

# Assume some other thread/task fills in the result later.
f.set_result(123)

print(store.get(ref))  # 123
```

## ObjectRef & ObjectStore: 3-line summary
- ObjectRef is not the value itself, but a handle that points to a result in a distributed environment.
- ObjectStore is the storage that holds the actual values referenced by those handles.
- In Ray, results are typically kept in the ObjectStore and returned as ObjectRefs, then fetched later with get(ref) when needed.