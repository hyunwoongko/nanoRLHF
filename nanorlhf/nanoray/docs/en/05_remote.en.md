# remote, RemoteFunction, ActorClass

## Overview

In Ray, remote execution usually starts in two forms.

- Function: remote function execution that runs once and finishes
- Class: Actor execution where you create once, then call methods many times

This project provides the remote decorator to handle both forms under the same surface API. Users attach remote to a function or a class, then start execution via a remote call. Internally, that call is converted into a Task and submitted to the session.

## Surface API

The usage a user sees looks like this.

```python
@remote(num_cpus=2.0)
def f(x):
    return x + 1

@remote(num_gpus=1.0, max_concurrency=4)
class A:
    def inc(self, x):
        return x + 1

r = f.remote(10)
a = A.remote()
o = a.inc.remote(10)
```

The important point here is that in both cases, the remote call does not immediately return the result value, but submits work to the session.

## Usage

In this project, you can think of the remote decorator as creating one of two wrappers depending on the input target.

- If it is a function, it wraps it with RemoteFunction
- If it is a class, it wraps it with ActorClass

```python
def remote(obj: Optional[Union[type, Callable[..., Any]]] = None, **opts: Any):
    def _wrap(x: Union[type, Callable[..., Any]]):
        if inspect.isclass(x):
            return ActorClass(cls=x, **opts)
        else:
            return RemoteFunction(fn=x, **opts)

    return _wrap if obj is None else _wrap(obj)
```

In other words, remote does not execute anything directly. Instead, it creates an executable handle object so the user can later call remote on it.

## RemoteFunction

RemoteFunction is the handle for remote function execution. It has one important role.

- Convert a function call into a Task and submit it

When a user calls f.remote(...), internally the flow is:

- f.remote(args, kwargs)
- create Task
- session submit

The key point of RemoteFunction is that the task method creates a Task.

```python
@dataclass(frozen=True)
class RemoteFunction:
    fn: Callable[..., Any]
    num_cpus: float = 1.0
    num_gpus: float = 0.0
    resources: Dict[str, float] = None
    runtime_env: Optional[RuntimeEnv] = None
    placement_group: Optional[PlacementGroup] = None
    bundle_index: Optional[int] = None
    pinned_node_id: Optional[str] = None
    max_concurrency: Optional[int] = 1

    def task(self, *args: Any, **kwargs: Any) -> Task:
        return Task.from_call(
            self.fn,
            args=tuple(args),
            kwargs=dict(kwargs),
            num_cpus=self.num_cpus,
            num_gpus=self.num_gpus,
            resources=dict(self.resources),
            runtime_env=self.runtime_env,
            placement_group_id=self.placement_group.pg_id if self.placement_group else None,
            bundle_index=self.bundle_index,
            pinned_node_id=self.pinned_node_id,
            max_concurrency=self.max_concurrency,
        )

    def remote(self, *args: Any, blocking: bool = False, **kwargs: Any) -> Optional[ObjectRef]:
        task = self.task(*args, **kwargs)
        sess = get_session()
        return sess.submit(task, blocking=blocking)
```

The message the reader should take away is this.

RemoteFunction does not execute the function call directly, but converts it into a Task to make it schedulable, then submits it

## ActorClass

ActorClass is the handle for class-based remote execution. When a user attaches remote to a class, they get an ActorClass, and they call remote on that ActorClass to send an instance creation request.

In this project, Actor creation is also represented as a Task. However, instead of putting a normal callable function in fn, it places a dictionary whose kind is actor_create, so the worker interprets it as an Actor creation request.

```python
@dataclass
class ActorClass:
    cls: type
    num_cpus: float = 0.0
    num_gpus: float = 0.0
    resources: Dict[str, float] = field(default_factory=dict)
    pinned_node_id: Optional[str] = None
    placement_group: Optional[PlacementGroup] = None
    bundle_index: Optional[int] = None
    runtime_env: Optional[RuntimeEnv] = None
    max_concurrency: Optional[int] = 1

    def remote(self, *args: Any, blocking: bool = False, **kwargs: Any):
        sess = get_session()
        task = Task(
            fn={
                "kind": "actor_create",
                "cls": self.cls,
                "args": args,
                "kwargs": kwargs,
            },
            args=(),
            kwargs={},
            num_cpus=self.num_cpus,
            num_gpus=self.num_gpus,
            resources=dict(self.resources),
            runtime_env=self.runtime_env,
            placement_group_id=self.placement_group.pg_id if self.placement_group else None,
            bundle_index=self.bundle_index,
            pinned_node_id=self.pinned_node_id,
            max_concurrency=self.max_concurrency,
            task_id=new_task_id(),
        )
        return sess.submit(task, blocking=blocking)
```

In other words, ActorClass.remote creates and submits a Task that requests: create an Actor from this class.

## Meaning of options

Both RemoteFunction and ActorClass provide options. This method returns a new handle with modified execution constraints. The important point is that options does not trigger execution.

- options only changes configuration
- execution happens at the remote call

So options can be understood as: a step to pre-set the default constraints of Tasks that will be submitted later.

## ActorRef and ActorMethod

After an Actor is created, the caller must be able to select methods through ActorRef and send remote calls. This project uses Python __getattr__ to turn attribute access like a.inc into an ActorMethod.

```python
@dataclass(frozen=True)
class ActorRef:
    actor_id: str
    owner_node_id: str

    def __getattr__(self, method_name: str) -> "ActorMethod":
        if method_name.startswith("__") and method_name.endswith("__"):
            raise AttributeError(method_name)
        return ActorMethod(self, method_name)
```

ActorMethod.remote also turns a method call into a Task and submits it. Here too, fn is a dictionary whose kind is actor_call.

```python
@dataclass(frozen=True)
class ActorMethod:
    ref: ActorRef
    method_name: str
    ...

    def remote(self, *args: Any, blocking: bool = False, **kwargs: Any):
        sess = get_session()
        task = Task(
            fn={
                "kind": "actor_call",
                "actor_id": self.ref.actor_id,
                "method": self.method_name,
            },
            args=args,
            kwargs=kwargs,
            ...
            pinned_node_id=self.ref.owner_node_id,
            ...
            task_id=new_task_id(),
        )
        return sess.submit(task, blocking=blocking)
```

The important concept here is this.

ActorMethod.remote converts the method call into a Task and submits it, and that Task has its execution location fixed by owner_node_id

## Remote: 3-line Summary

- remote is a factory that wraps functions and classes into RemoteFunction and ActorClass, respectively.
- RemoteFunction.remote converts a function call into Task.from_call and submits it.
- ActorClass.remote converts an Actor creation request into a Task whose kind is actor_create and submits it. ActorRef converts method access into ActorMethod, and ActorMethod.remote submits a Task whose kind is actor_call.