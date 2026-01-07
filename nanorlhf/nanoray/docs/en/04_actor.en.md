# Actor

## Overview

An Actor is a stateful unit of execution in Ray. Once an instance is created, it stays alive, repeatedly handling incoming method calls over time. This makes it natural to build components that preserve state even in a distributed environment, such as caches, accumulated statistics, or model handles.

While a Task is a one-shot execution whose lifetime ends after a single call finishes, an Actor is a long-lived execution model where calls are routed to the same instance after creation.

## Differences Between Task and Actor

The key point that distinguishes an Actor from a Task is the execution model.

- Lifetime
  - Task: created per call and ends after execution.
  - Actor: created once, stays alive, and handles many calls.

- State
  - Task: preserving state is not the primary goal.
  - Actor: preserving instance state is the core.

- Routing
  - Task: the execution location can vary per call.
  - Actor: the created instance has an actor_id, and subsequent calls are delivered to that instance.

Because of this difference, Actors are suitable for structures that keep serving requests like a service.

## Implementation

### Component Relationships

This project implements Actor by splitting it into three pieces.

- ActorRuntime: inside a node, finds handles by actor_id and is responsible for creation and routing.
- ActorHandle: a client-side handle that sends requests, receives responses, and completes Futures.
- actor main process: a function that runs on the real instance, executes requests, and returns responses.

### Actor Creation

```text
Caller
  |
  | 1) create(cls, init_args, init_kwargs)
  v
ActorRuntime
  |
  | 2) issue a new actor_id
  | 3) create request_q, response_q
  | 4) spawn actor main process
  | 5) create ActorHandle and start
  v
ActorHandle.start
  |
  | 6) start process
  | 7) start listen_loop thread
  v
actor main process
  |
  | 8) create instance
  | 9) put CreatedResponse into response_q
  v
response_q
  |
  | 10) listen_loop receives CreatedResponse and completes created_future
  v
Caller confirms readiness via created_future completion
```

### Actor Call

```text

Caller
  |
  | 1) call(actor_id, method, args, kwargs)
  v
ActorRuntime
  | 
  | 2) find handle by actor_id
  v 
ActorHandle.submit
  |
  | 3) create Future and register pending[call_id] = Future
  | 4) put CallRequest(call_id, method, payload) into request_q
  v 
request_q  ----------------------------------------------->  actor main process
                                                              |
                                                              | 5) request_q.get()
                                                              | 6) execute instance.method(*args, **kwargs)
                                                              | 7) put ResultResponse(call_id, ok, value/error)
                                                              |    into response_q
                                                              v
response_q <----------------------------------------------  actor main process
  |
  | 8) ActorHandle's listen_loop does response_q.get()
  | 9) find Future in pending by call_id and complete it
  |    - if ok: set_result(value)
  |    - else: set_exception(error)
  v
Caller observes the Future completion
```

Now we look at the actual implemented code.

### 1) ActorRuntime

ActorRuntime creates ActorHandle and actor_main_process, and finds the appropriate handle when calling.
It has one ActorHandle and actor_main_process per actor_id.

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

The important point in Actor is that after creation, it is routed by actor_id.
In Task, the call itself is the execution unit, but in Actor, an instance with an id is created and later calls are delivered to that instance.

### 2) ActorHandle

ActorHandle is a client-side handle that communicates with the real Actor instance.
It exchanges messages through request_q and response_q.

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

With this structure, the caller can observe results through an asynchronous Future interface, while internally communication happens via inter-process messages. In Ray as well, an Actor call returns a reference instead of an immediate value, and the value is fetched later.

### 3) actor main process

This is the function that runs inside the Actor instance (process).
It reads requests from request_q, executes the method, then sends responses to response_q.

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
When it sends a response to response_queue, ActorHandle that is listening receives it and completes the Future,
then returns it through ActorRuntime to the caller.

## Actor: 3-line Summary

- An Actor is a long-lived, stateful execution unit that handles method calls as messages.
- A Task is closer to one-shot call execution, while an Actor is a model where calls are routed to a created instance.
- This project implements a Ray-style Actor where ActorRuntime does routing, ActorHandle maps Futures, and actor main process performs the actual execution.