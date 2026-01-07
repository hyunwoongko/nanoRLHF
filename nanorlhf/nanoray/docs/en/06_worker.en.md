# Worker

## Overview

Worker is the component that actually executes in a distributed runtime. When a caller invokes RemoteFunction.remote or ActorClass.remote, that call is not executed immediately. Instead, it is converted into a Task and submitted somewhere. When that Task arrives at a specific node, the component on that node that receives the Task, executes it, then records the result into the ObjectStore is the Worker.

In short, Worker is an execution engine that takes a Task, turns it into an executable form, then connects the execution result back as an ObjectRef.

## Overall Relationship

If we put the components introduced so far into one picture, it looks like this.

```text
User code
  |
  | wrap with remote, then call .remote
  v
RemoteFunction / ActorClass / ActorMethod
  |
  | convert the call into a Task
  v
Session.submit(Task)
  |
  v
inside Scheduler.try_place or Scheduler.drain
  |
  | after choosing a node
  v
WorkerLike.execute_task(Task)
  |
  +--> local: Worker.execute_task(Task)
  |
  +--> remote: RemoteWorkerProxy.execute_task(Task)
          -> RpcClient.execute_task
          -> RpcServer rpc_execute_task
          -> Worker.rpc_execute_task(Task)
  |
  v
ObjectStore.put_future -> returns an ObjectRef
```

The core flow is:

- RemoteFunction / ActorClass / ActorMethod create Tasks
- Worker executes Tasks
- ActorRuntime and Handle support actor-related execution inside Worker

## Worker Responsibilities

Worker responsibilities can be summarized into three points.

- route each Task into the correct execution path
- run execution asynchronously, then register the result into the ObjectStore
- handle external requests for execution and reads via RPC-style entrypoints

One important point is that Worker does not return the result value directly. Worker registers a Future into the ObjectStore, then the caller later reads the value through an ObjectRef.

## Task Routing

Worker.execute_task chooses an execution path by looking at the shape of task.fn.

- if fn is a normal callable, it is a normal Task execution
- if fn is a dict and kind is actor_create, it is a create request
- if fn is a dict and kind is actor_call, it is a method call request

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

The message here is: "Worker is a single entrypoint that chooses an execution path only from the Task content."

## Normal Function Execution

Normal function execution is handled by a ThreadPoolExecutor. It serializes fn, args, kwargs contained in the Task, passes them into _invoke, then the executor runs that payload.

Concurrency is determined by task.max_concurrency, and for the same concurrency value the executor is reused.

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

The key idea here is not execution, but linkage.

- execution runs asynchronously in the executor
- Worker registers that Future into the ObjectStore
- the return value is not a value, but an ObjectRef

## ObjectStore and ObjectRef

Worker puts the execution result into the ObjectStore, then returns an ObjectRef that points to that result. Here, object_id is derived from task_id so each call result can be identified stably.

```python
return self.store.put_future(future, object_id=task_result_object_id(task.task_id))
```

This structure enables the following in distributed execution.

- the caller can proceed immediately to the next step
- the result can be read later when needed
- result passing behaves like passing references, not copying values

## Remote Execution RPC

Worker has an entrypoint that accepts external requests to execute Tasks. rpc_execute_task calls the internal execute_task, then constructs an ObjectRef for external use.
Difference is that size_bytes information is included in the ObjectRef. The caller can use this value to know the result size in advance.

```python
def rpc_execute_task(self, task: Task) -> ObjectRef:
    ref = self.execute_task(task)
    return ObjectRef(
        object_id=ref.object_id,
        owner_node_id=self.store.node_id,
        size_bytes=self.store.get_size(ref.object_id),
    )
```

Result reads fetch bytes from the ObjectStore by object_id and return them.

```python
def rpc_read_object_bytes(self, object_id: str) -> bytes:
    return self.store.get_bytes(object_id)
```

So Worker is both an execution server and a read gateway for the result store.

## Connecting RemoteFunction, ActorClass with Worker

RemoteFunction and ActorClass are not the executors. They convert calls into Tasks and submit them to the session.

- RemoteFunction.remote creates a normal function Task via Task.from_call
- ActorClass.remote creates a Task whose kind is actor_create
- ActorMethod.remote creates a Task whose kind is actor_call

Worker receives that Task, then selects an execution path based on kind and shape. So Worker is a Task executor, while RemoteFunction and ActorClass are Task producers.

## Where ActorRuntime and Handle Live

Worker holds an ActorRuntime internally.

```python
self.actors = ActorRuntime(node_id=self.node_id)
```

When an actor-related Task arrives, Worker forwards it to ActorRuntime.

- actor_create: calls ActorRuntime.create, then once ready, produces an ActorRef as the result
- actor_call: calls ActorRuntime.call, then registers that Future into the ObjectStore

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

Here, ActorHandle exists inside ActorRuntime as a call management structure, and Worker does not manipulate ActorHandle directly. From the Worker perspective, ActorRuntime is a module that encapsulates actor-related execution.

## Worker: 3-line Summary

- RemoteFunction, ActorClass, ActorMethod convert calls into Tasks and submit them to the session.
- Worker receives Tasks on a node, chooses an execution path, starts execution, then registers the result as a Future in the ObjectStore and links it via an ObjectRef.
- Actor-related Tasks are handled by the ActorRuntime inside Worker, and Worker links that Future into the ObjectStore.