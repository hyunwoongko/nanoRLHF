# Worker

## 개요

Worker는 분산 런타임에서 "실제로 실행하는 쪽"입니다. 호출자가 RemoteFunction.remote나 ActorClass.remote를 부르면, 그 호출은 곧바로 실행되지 않고 Task로 바뀐 뒤 어딘가에 제출됩니다. 그 Task가 특정 노드에 도착했을 때, 그 노드에서 Task를 받아 실행하고 결과를 ObjectStore에 기록하는 컴포넌트가 Worker입니다.

정리하면 Worker는 Task를 받아서 실행 형태로 바꾸고, 실행 결과를 ObjectRef로 연결해 주는 실행 엔진입니다.

## 전체 관계

여기까지 등장한 구성요소를 한 그림으로 묶으면 아래 구조입니다.

```text
사용자 코드
  |
  | remote로 감싼 뒤 .remote 호출
  v
RemoteFunction / ActorClass / ActorMethod
  |
  | 호출을 Task로 변환
  v
Session.submit(Task)
  |
  | (네트워크/스케줄링을 거쳐) 특정 노드에 전달
  v
Worker.rpc_execute_task(Task)
  |
  | Task의 종류를 보고 실행 경로 선택
  v
Worker.execute_task
  |
  +--> 일반 함수 실행: execute_task_call -> ThreadPoolExecutor
  |
  +--> 생성 요청: execute_actor_create -> ActorRuntime.create
  |
  +--> 메서드 호출: execute_actor_call -> ActorRuntime.call
  |
  v
ObjectStore.put_future -> ObjectRef 반환
```

핵심은 이 흐름입니다.

- RemoteFunction / ActorClass / ActorMethod는 Task를 만든다
- Worker는 Task를 실행한다
- ActorRuntime과 Handle은 Worker 내부에서 Actor 관련 실행을 뒷받침한다

## Worker의 역할

Worker의 역할은 세 가지로 요약됩니다.

- Task를 실행 경로로 분기한다
- 실행을 비동기로 진행하고 결과를 ObjectStore에 등록한다
- RPC 형태로 외부에서 실행과 읽기 요청을 처리한다

여기서 중요한 점은 Worker가 결과 값을 직접 반환하지 않는다는 점입니다. Worker는 Future를 ObjectStore에 등록하고, 호출자는 ObjectRef를 통해 나중에 값을 읽는 구조입니다.

## Task 분기

Worker.execute_task는 task.fn의 형태를 보고 실행 경로를 고릅니다.

- fn이 일반 호출 가능 객체이면 일반 Task 실행입니다
- fn이 dict이고 kind가 actor_create이면 생성 요청입니다
- fn이 dict이고 kind가 actor_call이면 메서드 호출 요청입니다

```python
def execute_task(self, task: Task) -> ObjectRef:
    ctx = getattr(task, "runtime_env", None)
    ctx_mgr = ctx.apply() if ctx is not None else nullcontext()

    with ctx_mgr:
        fn = task.fn

        if isinstance(fn, dict) and fn.get("kind") == "actor_create":
            return self.execute_actor_create(task, fn)

        if isinstance(fn, dict) and fn.get("kind") == "actor_call":
            return self.execute_actor_call(task, fn)

        return self.execute_task_call(task)
```

여기서 메시지는 이것입니다: "Worker는 Task의 내용만 보고 실행 경로를 선택하는 단일 진입점이다."

## 일반 함수 실행

일반 함수 실행은 ThreadPoolExecutor로 처리합니다. Task에 담긴 fn, args, kwargs를 직렬화해 _invoke로 넘기고, executor가 그 payload를 실행합니다.

동시성은 task.max_concurrency로 결정되며, 같은 동시성 값에 대해서는 executor를 재사용합니다.

```python
def execute_task_call(self, task: Task) -> ObjectRef:
    payload = dumps((task.fn, task.args, task.kwargs))
    max_concurrency = max(int(task.max_concurrency or 1), 1)

    executor = self.task_executors.get(max_concurrency)
    if executor is None:
        executor = ThreadPoolExecutor(max_workers=max_concurrency)
        self.task_executors[max_concurrency] = executor

    future = executor.submit(_invoke, payload)
    return self.store.put_future(future, object_id=task_result_object_id(task.task_id))
```

여기서 중요한 개념은 실행이 아니라 연결입니다.

- 실행은 executor에서 비동기로 진행됩니다
- Worker는 그 Future를 ObjectStore에 등록합니다
- 반환은 값이 아니라 ObjectRef입니다

## ObjectStore와 ObjectRef

Worker는 실행 결과를 ObjectStore에 넣고, 그 결과를 가리키는 ObjectRef를 반환합니다. 이때 object_id는 task_id 기반으로 만들어져 호출 단위의 결과를 안정적으로 식별합니다.

```python
return self.store.put_future(future, object_id=task_result_object_id(task.task_id))
```

이 구조는 분산 실행에서 다음을 가능하게 합니다.

- 호출자는 즉시 다음 단계로 진행할 수 있습니다
- 결과는 나중에 필요할 때 읽을 수 있습니다
- 결과 전달은 값 복사가 아니라 참조 전달처럼 동작합니다

## 원격 실행 RPC

Worker는 외부에서 Task 실행을 요청받는 진입점이 있습니다. rpc_execute_task는 내부 execute_task를 호출한 뒤, 외부로 넘길 ObjectRef를 구성합니다.

```python
def rpc_execute_task(self, task: Task) -> ObjectRef:
    ref = self.execute_task(task)
    return ObjectRef(
        object_id=ref.object_id,
        owner_node_id=self.store.node_id,
        size_bytes=self.store.get_size(ref.object_id),
    )
```

그리고 결과 읽기는 object_id로 ObjectStore에서 바이트를 꺼내 반환합니다.

```python
def rpc_read_object_bytes(self, object_id: str) -> bytes:
    return self.store.get_bytes(object_id)
```

즉 Worker는 실행 서버이면서 결과 저장소에 대한 읽기 창구 역할도 함께 합니다.

## RemoteFunction, ActorClass와 Worker의 연결

RemoteFunction과 ActorClass는 실행 주체가 아닙니다. 이들은 호출을 Task로 바꿔 세션에 제출하는 쪽입니다.

- RemoteFunction.remote는 Task.from_call로 일반 함수 Task를 만든다
- ActorClass.remote는 kind가 actor_create인 Task를 만든다
- ActorMethod.remote는 kind가 actor_call인 Task를 만든다

Worker는 그 Task를 받아 kind와 형태를 보고 실행 경로를 선택합니다. 그래서 Worker는 "Task 실행기"이고, RemoteFunction이나 ActorClass는 "Task 생성기"입니다.

## ActorRuntime과 Handle의 위치

Worker는 ActorRuntime을 내부에 가지고 있습니다.

```python
self.actors = ActorRuntime(node_id=self.node_id)
```

Actor 관련 Task가 들어오면 Worker는 ActorRuntime으로 넘깁니다.

- actor_create: ActorRuntime.create를 호출하고, 준비가 끝나면 ActorRef를 결과로 만든다
- actor_call: ActorRuntime.call을 호출하고, 그 Future를 ObjectStore에 등록한다

```python
def execute_actor_create(self, task: Task, fn: Dict[str, Any]) -> ObjectRef:
    actor_id, ready_future = self.actors.create(...)
    future = Future()

    def finish(done: Future):
        done.result()
        future.set_result(ActorRef(actor_id=actor_id, owner_node_id=self.node_id))

    ready_future.add_done_callback(finish)
    return self.store.put_future(future, object_id=task_result_object_id(task.task_id))
```

```python
def execute_actor_call(self, task: Task, fn: Dict[str, Any]) -> ObjectRef:
    future = self.actors.call(...)
    return self.store.put_future(future, object_id=task_result_object_id(task.task_id))
```

여기서 ActorHandle은 ActorRuntime 내부에 존재하는 호출 관리 구조이고, Worker는 ActorHandle을 직접 다루지 않습니다. Worker 입장에서는 ActorRuntime이 Actor 관련 실행을 캡슐화해 주는 모듈입니다.

## Worker: 3줄 요약

- RemoteFunction, ActorClass, ActorMethod는 호출을 Task로 바꿔 세션에 제출합니다.
- Worker는 노드에서 Task를 받아 실행 경로를 선택하고 실행을 시작하고 실행 결과는 ObjectStore에 Future로 등록되고 ObjectRef로 연결됩니다.
- Actor 관련 Task는 Worker 내부의 ActorRuntime이 처리하며, Worker는 그 Future를 ObjectStore에 연결합니다.