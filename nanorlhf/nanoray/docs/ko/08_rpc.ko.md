# RPC

## 개요

분산 런타임에서 중요한 사실 하나는 실행 주체(Worker)와 호출자(Session)가 같은 프로세스에 있지 않을 수 있다는 점입니다. 노드가 여러 개면, Task는 어떤 노드의 Worker에서 실행될지에 따라 네트워크를 타고 이동해야 하고, 실행 결과(ObjectRef가 가리키는 바이트)도 다시 네트워크를 통해 읽혀야 합니다.

이 프로젝트의 RPC는 딱 두 가지 기능만 최소한으로 제공합니다.

- 원격 노드에 Task 실행을 요청한다
- 원격 노드에서 object_id에 해당하는 바이트를 가져온다

즉 RPC는 스케줄링 정책, 실행 로직의 일부가 아니라 노드 사이에서 Task와 결과를 운반하는 전송 계층입니다.

## 구성요소

- RemoteWorkerProxy
  - Scheduler가 Worker처럼 사용할 수 있는 프록시입니다
  - execute_task 호출을 원격 노드 RPC 호출로 바꿉니다

- NodeRegistry
  - node_id -> (주소, 토큰) 매핑을 보관합니다

- Router
  - ObjectRef를 보고 어느 노드에 가야 하는지 결정합니다
  - 현재는 ref.owner_node_id를 그대로 사용합니다

- RpcClient
  - HTTP JSON 기반 클라이언트입니다
  - execute_task, get_object 요청을 보냅니다

- RpcServer
  - 각 노드에서 떠 있는 HTTP 서버입니다
  - Worker의 rpc_execute_task, rpc_read_object_bytes를 외부로 노출합니다

## 큰 그림

RPC는 실행 요청 경로와 결과 읽기 경로를 각각 제공합니다.

```text
실행 요청 경로

Scheduler
  -> (노드 선택)
  -> Worker.execute_task(Task) 호출

로컬 노드면
  Scheduler -> Worker.execute_task

원격 노드면
  Scheduler -> RemoteWorkerProxy.execute_task
           -> RpcClient.execute_task(node_id, task)
           -> (HTTP) RpcServer /rpc/execute_task
           -> Worker.rpc_execute_task(task)
           -> ObjectRef 반환
```

```text
결과 읽기 경로

ObjectRef(object_id, owner_node_id)
  -> Router.route_object(ref) 로 대상 노드 결정
  -> RpcClient.get_object(owner_node_id, object_id)
  -> (HTTP) RpcServer /rpc/get_object
  -> Worker.rpc_read_object_bytes(object_id)
  -> bytes 반환
```

## RemoteWorkerProxy

Scheduler는 실행이 로컬 Worker인지 원격 노드인지 구분하지 않으려 합니다. 그래서 execute_task만 있으면 되는 형태를 기대하고, RemoteWorkerProxy가 그 자리를 채웁니다.

```python
class RemoteWorkerProxy:
    def __init__(self, node_id: str, rpc: RpcClient):
        self.node_id = node_id
        self.rpc = rpc

    def execute_task(self, task: Task) -> ObjectRef:
        return self.rpc.execute_task(self.node_id, task)
```

핵심은 하나입니다. Scheduler 입장에서는 로컬 Worker 호출이든 원격 RPC 호출이든 동일한 execute_task 인터페이스로 보이게 만드는 것입니다.

## NodeRegistry와 Router

NodeRegistry는 노드에 접속하는 방법을 저장합니다. Router는 어떤 ref를 어디로 보내야 하는지 결정합니다. 역할이 분리되어 있어 라우팅 정책을 바꾸더라도 레지스트리 저장 구조를 건드릴 필요가 없습니다.

```python
@dataclass
class NodeRegistry:
    _table: Dict[str, Tuple[str, Optional[str]]] = None

    def register(self, node_id: str, address: str, token: Optional[str] = None):
        self._table[node_id] = (address.rstrip("/"), token)

    def get(self, node_id: str) -> Optional[Tuple[str, Optional[str]]]:
        return self._table.get(node_id, None)
```

```python
class Router:
    def __init__(self, registry: NodeRegistry):
        self._registry = registry

    def route_object(self, ref: ObjectRef) -> Optional[str]:
        return ref.owner_node_id
```

현재 라우팅은 단순합니다. ObjectRef가 누가 소유자인지 이미 들고 있으니 그 노드로 간다고 보면 됩니다.

## RpcClient

RpcClient는 표준 라이브러리로 HTTP POST JSON 요청을 보내는 최소 구현입니다. 중요한 엔드포인트는 두 개입니다.

- /rpc/execute_task
- /rpc/get_object

Task는 그대로 JSON에 넣지 않고, 직렬화한 바이트를 base64로 인코딩해 전송합니다. 이유는 HTTP JSON 바디는 bytes를 직접 담을 수 없기 때문입니다.

```python
class RpcClient:
    def execute_task(self, node_id: str, task: Task) -> ObjectRef:
        blob = dumps(task)
        response = self.async_request(
            node_id=node_id,
            path="/rpc/execute_task",
            body={"task_b64": base64.b64encode(blob).decode("ascii")},
        ).result()

        ref = response["ref"]
        return ObjectRef(
            object_id=ref["object_id"],
            owner_node_id=ref["owner_node_id"],
            size_bytes=ref.get("size_bytes"),
        )

    def get_object(self, node_id: str, object_id: str) -> bytes:
        response = self.async_request(
            node_id=node_id,
            path="/rpc/get_object",
            body={"object_id": object_id},
        ).result()

        return base64.b64decode(response["payload_b64"])
```

여기서 알아야 할 개념은 두 가지입니다.

- execute_task는 결과 값 자체가 아니라 ObjectRef를 돌려줍니다
- get_object는 object_id에 해당하는 직렬화 바이트를 돌려줍니다

즉 원격 실행의 반환은 언제나 참조이며, 값은 별도 경로로 읽습니다.

## RpcServer

RpcServer는 노드마다 떠 있는 HTTP 서버입니다. 외부에서 들어온 요청을 Worker 호출로 변환하는 얇은 어댑터입니다.

```text
POST /rpc/execute_task
  입력: task_b64
  처리: task 역직렬화 -> worker.rpc_execute_task(task)
  출력: ObjectRef를 JSON으로 반환

POST /rpc/get_object
  입력: object_id
  처리: worker.rpc_read_object_bytes(object_id)
  출력: bytes를 base64로 인코딩해 반환
```

ThreadingMixIn을 사용해서 요청별로 스레드를 할당합니다. 이 설계는 RPC 핸들링이 I/O 중심인 상황에서 간단히 동시성을 확보하기 위함입니다. 서버 자체는 실행 스케줄링을 하지 않고, Worker가 제공하는 두 RPC 메서드만 호출합니다.

## Scheduler, Worker, RPC의 연결

이 프로젝트의 연결 고리는 이렇게 정리하면 끝납니다.

- Scheduler는 노드를 고릅니다
- 실행이 로컬이면 Worker.execute_task를 직접 호출합니다
- 실행이 원격이면 RemoteWorkerProxy.execute_task를 호출합니다
- RemoteWorkerProxy는 RpcClient로 원격 RpcServer를 호출합니다
- RpcServer는 Worker.rpc_execute_task를 호출해 실행을 위임합니다
- 결과는 ObjectRef로 돌아옵니다
- 값이 필요하면 Router가 owner_node_id를 결정하고 RpcClient.get_object로 바이트를 가져옵니다

## RPC: 3줄 요약

- RPC는 Task 실행 요청과 결과 바이트 읽기만 제공하는 최소 전송 계층입니다
- NodeRegistry는 접속 정보, Router는 ref 기준 목적지 결정을 담당합니다
- RpcClient는 요청 전송, RpcServer는 Worker 호출로 변환하는 역할입니다