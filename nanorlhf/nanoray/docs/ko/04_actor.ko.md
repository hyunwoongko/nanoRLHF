# Actor

## 개요

Actor는 Ray에서 상태를 가진 실행 단위입니다. 한 번 만들어진 인스턴스가 계속 살아 있으면서, 이후 들어오는 메서드 호출을 반복해서 처리합니다. 이 덕분에 분산 환경에서도 캐시, 누적 통계, 모델 핸들처럼 상태를 유지하는 컴포넌트를 자연스럽게 구성할 수 있습니다.

Task는 호출 하나가 끝나면 수명이 종료되는 단발성 실행인 반면, Actor는 생성 이후 동일 인스턴스로 호출이 라우팅되는 장수 실행 모델입니다.

## Task와 Actor의 차이

Actor를 Task와 구분하는 핵심은 실행 모델입니다.

- 수명
  - Task: 호출 단위로 생성되고 실행 후 종료됩니다.
  - Actor: 한 번 생성되면 계속 살아 있으며 여러 호출을 처리합니다.

- 상태
  - Task: 상태 유지가 기본 목표가 아닙니다.
  - Actor: 인스턴스 상태를 유지하는 것이 핵심입니다.

- 라우팅
  - Task: 호출마다 실행 위치가 달라질 수 있습니다.
  - Actor: 생성된 인스턴스가 actor_id를 가지며 이후 호출이 그 인스턴스로 전달됩니다.

이 차이 때문에 Actor는 서비스처럼 계속 요청을 처리하는 구조에 적합합니다.

## 구현

### 구성 관계

이 프로젝트는 Actor를 세 조각으로 나눠 구현합니다.

- ActorRuntime: 노드 내부에서 actor_id 기준으로 핸들을 찾고, 생성과 라우팅을 맡습니다.
- ActorHandle: 호출자 쪽에서 요청을 보내고, 응답을 받아 Future를 완료하는 클라이언트 핸들입니다.
- actor main process: 실제 인스턴스에서 동작하는 함수이며, 요청을 실행하고 응답을 반환합니다.

### Actor 생성

```text
Caller
  |
  | 1) create(cls, init_args, init_kwargs)
  v
ActorRuntime
  |
  | 2) 새 actor_id 발급
  | 3) request_q, response_q 생성
  | 4) actor main process spawn
  | 5) ActorHandle 만들고 start
  v
ActorHandle.start
  |
  | 6) 프로세스 start
  | 7) listen_loop 스레드 start
  v
actor main process
  |
  | 8) 인스턴스 생성
  | 9) CreatedResponse를 response_q에 put
  v
response_q
  |
  | 10) listen_loop가 CreatedResponse를 받고 created_future 완료
  v
Caller는 created_future 완료로 준비됨을 확인
```

### Actor 호출

```text

Caller
  |
  | 1) call(actor_id, method, args, kwargs)
  v
ActorRuntime
  | 
  | 2) actor_id로 handle 찾기
  v 
ActorHandle.submit
  |
  | 3) pending[call_id] = Future 만들고 등록
  | 4) CallRequest(call_id, method, payload)를 request_q에 put
  v 
request_q  ----------------------------------------------->  actor main process
                                                              |
                                                              | 5) request_q.get()
                                                              | 6) instance.method(*args, **kwargs) 실행
                                                              | 7) ResultResponse(call_id, ok, value/error)
                                                              |    를 response_q에 put
                                                              v
response_q <----------------------------------------------  actor main process
  |
  | 8) ActorHandle의 listen_loop가 response_q.get()
  | 9) call_id로 pending에서 Future를 찾고 완료
  |    - ok면 set_result(value)
  |    - 아니면 set_exception(error)
  v
Caller는 Future가 완료된 것을 관찰
```

이제 실제 구현된 코드를 살펴봅니다.

### 1) ActorRuntime

ActorRuntime은 ActorHandle과 actor_main_process를 생성하며, 호출시 적합한 핸들을 찾아주는 역할을 합니다.
actor_id별로 하나의 ActorHandle과 actor_main_process를 가집니다.

```python
class ActorRuntime:
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.actors: Dict[str, ActorHandle] = {}

    def create(self, cls, init_args, init_kwargs, max_concurrency):
        actor_id = new_actor_id()
        mp_ctx = mp.get_context("spawn")
        request_q = mp_ctx.Queue()
        response_q = mp_ctx.Queue()

        create_payload = dumps((cls, init_args, init_kwargs))
        proc = mp_ctx.Process(
            target=actor_main_process,
            args=(actor_id, create_payload, request_q, response_q, max_concurrency),
            daemon=True,
        )

        handle = ActorHandle(
            actor_id=actor_id,
            node_id=self.node_id,
            request_q=request_q,
            response_q=response_q,
            process=proc,
            max_concurrency=max_concurrency,
        )
        self.actors[actor_id] = handle
        handle.start()
        return actor_id, handle.created_future

    def call(self, actor_id, call_id, method_name, args, kwargs, max_concurrency):
        handle = self.actors.get(actor_id)
        if handle is None:
            raise RuntimeError(f"Actor {actor_id} not found on node {self.node_id}.")
        return handle.submit(call_id, method_name, args, kwargs, max_concurrency=max_concurrency)
```

Actor에서 중요한 점은 생성 이후 actor_id 찾아 라우팅 된다는 점입니다. 
Task는 호출이 곧 실행 단위지만, Actor는 id가 있는 인스턴스가 생기고 이후 호출이 그 인스턴스로 전달됩니다.

### 2) ActorHandle

ActorHandle은 호출자 쪽에서 실제 Actor 인스턴스와 통신하는 클라이언트 핸들입니다.
Actor 객체와 request_q와 response_q를 통해 메시지를 주고받습니다.

```python
@dataclass
class ActorHandle:
    pending: Dict[str, Future] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def submit(self, call_id, method_name, args, kwargs, max_concurrency):
        future = Future()
        with self.lock:
            self.pending[call_id] = future

        payload = dumps((args, kwargs))
        self.request_q.put(CallRequest(call_id=call_id, method_name=method_name, payload=payload))
        return future

    def listen_loop(self):
        while True:
            response = self.response_q.get()

            if isinstance(response, ResultResponse):
                with self.lock:
                    future = self.pending.pop(response.call_id, None)
                if future is None:
                    continue

                if response.ok:
                    future.set_result(loads(response.value_payload))
                else:
                    exc_name, exc_msg, tb = loads(response.error_payload)
                    future.set_exception(RuntimeError(f"Actor call failed actor_id={response.actor_id} call_id={response.call_id} exc={exc_name}: {exc_msg}\n{tb}"))

            elif isinstance(response, ShutdownDoneResponse):
                break
```

이 구조 덕분에 호출자는 비동기 Future 인터페이스로 결과를 관찰할 수 있고, 내부에서는 프로세스 간 메시지로 통신합니다. Ray에서도 Actor 호출이 즉시 값 대신 참조를 반환하고, 나중에 값을 가져옵니다.

### 3) actor main process

Actor 인스턴스 (프로세스)에서 동작하는 함수입니다. 
이 프로세스는 request_q에서 요청을 읽고, 메서드를 실행한 뒤 response_q로 응답을 보냅니다.

```python
def actor_main_process(actor_id, create_payload, request_queue, response_queue, initial_max_concurrency):
    cls, init_args, init_kwargs = loads(create_payload)
    instance = cls(*init_args, **(init_kwargs or {}))

    max_concurrency = max(int(initial_max_concurrency or 1), 1)
    parallel_thread_pool = ThreadPoolExecutor(max_workers=max_concurrency)
    serial_thread_pool = ThreadPoolExecutor(max_workers=1)
    lock = threading.Lock()

    response_queue.put(CreatedResponse(actor_id=actor_id, max_concurrency=max_concurrency))

    while True:
        request = request_queue.get()

        if isinstance(request, ShutdownRequest):
            break
        elif isinstance(request, ResizeRequest):
            ...
            response_queue.put(ResizedResponse(actor_id=actor_id, max_concurrency=max_concurrency))
        elif isinstance(request, CallRequest):
            ...
        else:
            raise RuntimeError(f"Unknown request type: {type(request)}")

    response_queue.put(ShutdownDoneResponse(actor_id=actor_id))
```
response_queue에 리스폰스를 보내면 그것을 listening 하고 있는 ActorHandle이 받아서 Future를 완료하며,
그것을 ActorRuntime에게 돌려주어 호출자에게 전달됩니다.

## Actor: 3줄 요약

- Actor는 상태를 가진 장수 실행 단위이며, 메서드 호출을 메시지로 받아 처리합니다.
- Task는 단발성 호출 실행에 가깝고, Actor는 생성된 인스턴스로 호출이 라우팅되는 모델입니다.
- 이 프로젝트는 ActorRuntime이 라우팅, ActorHandle이 Future 매핑, actor main process가 실제 실행을 맡는 구조로 Ray 스타일 Actor를 구현합니다.