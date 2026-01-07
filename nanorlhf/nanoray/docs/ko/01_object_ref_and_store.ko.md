# ObjectRef & ObjectStore

Ray를 처음 써보면 보통 이런 상황부터 만나게 됩니다. remote로 함수를 실행했는데 결과가 숫자나 딕셔너리가 아니라 ObjectRef 같은 것이 돌아옵니다. 출력해도 값이 바로 보이지 않고, 왜 ray.get을 해야 하는지도 처음엔 감이 잘 오지 않습니다.

분산 실행 프레임워크에서는 계산이 다른 프로세스나 다른 머신에서 수행될 수 있고, 결과 값도 그곳의 메모리에 존재할 수 있습니다. 그래서 Ray는 값 자체를 즉시 복사해서 돌려주기보다, 값을 가리키는 핸들을 먼저 돌려주고 필요할 때 가져오게 설계합니다.

Ray에서는 이 핸들을 ObjectRef라고 부르고, 실제 값을 저장하는 메모리 공간을 ObjectStore라고 부릅니다. nanoray도 같은 핵심 아이디어를 따라가며, 이 문서에서는 그 원리를 이해하는 데 집중합니다.

## 1. remote 호출의 반환값은 왜 값이 아니라 ref인가요?

### 로컬 파이썬의 직관
로컬 파이썬에서는 다음이 자연스럽습니다.
어떤 함수가 값을 반환하면, 그 값 자체가 돌아옵니다.

```python
x = f()
# x는 값 자체입니다.
```

### 분산 환경의 직관
Ray에서 remote 호출은 보통 값 대신 ObjectRef를 반환합니다. 값이 다른 곳에 있을 수 있기 때문에, 우선 참조를 돌려주고 런타임이 필요할 때 값을 가져오도록 합니다.

```python
ref = f.remote()
# ref는 ObjectRef입니다. 값이 아닙니다.
x = ray.get(ref)
# ray.get을 호출해야 실제 값을 가져올 수 있습니다.
```

## 2. ObjectRef: 값이 아니라 값을 가리키는 핸들

ObjectRef는 어떤 객체를 가리키는 참조입니다. 중요한 점은 ObjectRef에는 값이 들어있지 않다는 점입니다. ObjectRef는 런타임이 값을 찾고 가져오도록 도와주는 최소한의 메타데이터만 들고 있습니다.

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

### ObjectRef는 왜 필요한가요?
값을 바로 돌려주면 제일 간단해 보이지만, 분산 환경에서는 그 순간에 큰 비용이 발생할 수 있습니다. 예를 들어 큰 텐서나 큰 리스트를 매번 복사해서 드라이버로 가져오면, 네트워크 대역폭과 메모리가 빠르게 병목이 됩니다. 그래서 Ray는 기본적으로 결과를 ObjectStore에 두고, 사용자는 ObjectRef를 통해 필요할 때만 가져오게 합니다.

### owner는 무엇인가요?
멀티 노드 환경에서는 ObjectStore가 노드마다 따로 존재합니다. ObjectRef의 owner_node_id는 어느 노드의 ObjectStore에 있는지를 가리키는 힌트이고, 런타임이 get(ref)를 어디로 라우팅할지 결정할 때 사용됩니다.

### is_local은 어떤 의미인가요?
get(ref)를 할 때, 로컬이면 바로 읽고, 아니면 owner에게 요청해야 합니다. is_local은 그 분기 조건을 표현합니다.

```python
ref1 = ObjectRef("obj-1")
print(ref1.is_local(None))  # True (single-node convention)

ref2 = ObjectRef("obj-2", "node-A")
print(ref2.is_local("node-A"))  # True
print(ref2.is_local("node-B"))  # False
```

## 3. ObjectStore: 실제 값을 들고 있는 저장소

ObjectStore는 object_id를 키로 해서 실제 Python 객체를 저장합니다. ObjectRef가 라벨이라면, ObjectStore는 그 라벨이 붙은 실제 물건들이 쌓여 있는 저장소입니다.

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

## 4. 동작 예시

```python
store = ObjectStore("node-A")

ref = store.put({"x": 1})
print(ref.object_id)      # 예: "obj-a1b2c3d4"
print(ref.owner_node_id)  # "node-A"

value = store.get(ref)
print(value)  # {"x": 1}
```

ObjectStore의 put은 값을 저장하고, 그 값을 가리키는 ObjectRef를 돌려준다는 점입니다. Ray에서 remote 태스크가 값 대신 ObjectRef를 반환하는 흐름도 이 원리와 같습니다.

## 5. bytes API는 왜 필요한가요?
멀티 노드가 되면 다른 노드의 ObjectStore에서 값을 가져와야 합니다. 네트워크는 Python 객체를 그대로 옮길 수 없기 때문에, 직렬화된 bytes 형태로 보내고 받는 API가 필요합니다.

```python
store_1 = ObjectStore("node-A")
ref_1 = store_1.put([1, 2, 3])

payload = store_1.get_bytes(ref_1.object_id)

store_2 = ObjectStore("node-B")
ref_2 = store_2.put_bytes(payload)

print(store_2.get(ref_2))  # [1, 2, 3]
```

이 `get_bytes` / `put_bytes`는 사용자용 고수준 API라기보다, Ray에서 원격 fetch를 구현할 때 필요한 저수준 빌딩블록에 가깝습니다.

## 6. Future를 저장하는 이유는 무엇인가요?
분산 시스템에서는 결과가 아직 준비되지 않았을 수도 있습니다. 그럴 때도 ref는 즉시 돌려주고, 실제 값은 준비되는 대로 store에 반영하는 흐름이 유용합니다.

```python
from concurrent.futures import Future

store = ObjectStore("node-A")

f = Future()
ref = store.put_future(f, object_id="obj-fixed")
print(ref.object_id)  # "obj-fixed"

# 나중에 다른 스레드/작업에서 결과를 채운다고 가정합니다.
f.set_result(123)

print(store.get(ref))  # 123
```

## ObjectRef & ObjectStore: 3줄 요약
- ObjectRef는 값 자체가 아니라, 분산 환경에서 결과를 가리키는 핸들입니다.
- ObjectStore는 그 핸들이 가리키는 실제 값을 저장하는 저장소입니다.
- Ray에서는 보통 결과를 ObjectStore에 두고 ObjectRef를 돌려준 뒤, 필요할 때 get(ref)로 값을 가져오는 흐름을 씁니다.