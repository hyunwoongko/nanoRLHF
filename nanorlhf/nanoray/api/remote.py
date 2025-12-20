from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Protocol

from nanorlhf.nanoray.api.session import get_session
from nanorlhf.nanoray.core.actor import actor as actor_class
from nanorlhf.nanoray.core.object_ref import ObjectRef
from nanorlhf.nanoray.core.placement import PlacementGroup
from nanorlhf.nanoray.core.runtime_env import RuntimeEnv
from nanorlhf.nanoray.core.task import Task


@dataclass(frozen=True)
class RemoteFunction:
    """
    A thin Ray-like wrapper that turns a Python function into a remote-invocable one.

    Attributes:
        fn (Callable[..., Any]): The original Python function to execute remotely.
        num_cpus (float): CPU requirement for scheduling (teaching-scale).
        num_gpus (float): GPU requirement for scheduling.
        resources (Dict[str, float]): Custom named resource requirements.
        runtime_env (Optional[RuntimeEnv]): Scoped env applied around this task.

    Examples:
        >>> from nanorlhf.nanoray.api.initialization import init, shutdown
        >>> from nanorlhf.nanoray.api.remote import remote
        >>> from nanorlhf.nanoray.api.session import get, drain
        >>>
        >>> @remote(num_cpus=2.0)
        ... def add(x, y): return x + y
        ...
        >>> sess = init()                      # zero-arg init
        >>> r = add.remote(3, 4)               # Optional[ObjectRef]
        >>> refs = ([r] if r is not None else []) + drain()
        >>> get(refs[-1])
        7
        >>> shutdown()

    Discussion:
        Q. Why not always return an ObjectRef from `.remote(...)`?
            Our teaching scheduler may queue tasks if resources are unavailable.
            We keep this visible by returning `Optional[ObjectRef]` and using
            `drain()` to advance queued work. This makes scheduling state explicit.

        Q. What if I want a `Task` without submitting?
            Use `.task(*args, **kwargs)` to build a `Task`. This works even
            without a global session and can be submitted manually later.
    """

    fn: Callable[..., Any]
    num_cpus: float = 1.0
    num_gpus: float = 0.0
    resources: Dict[str, float] = None
    runtime_env: Optional[RuntimeEnv] = None
    placement_group: Optional[PlacementGroup] = None
    bundle_index: Optional[int] = None
    pinned_node_id: Optional[str] = None
    max_concurrency: Optional[int] = 1

    def __post_init__(self) -> None:
        if self.resources is None:
            object.__setattr__(self, "resources", {})

    def options(
        self,
        *,
        num_cpus: Optional[float] = None,
        num_gpus: Optional[float] = None,
        resources: Optional[Dict[str, float]] = None,
        runtime_env: Optional[RuntimeEnv] = None,
        placement_group: Optional[PlacementGroup] = None,
        bundle_index: Optional[int] = None,
        pinned_node_id: Optional[str] = None,
        max_concurrency: Optional[int] = None,
    ) -> "RemoteFunction":
        """
        Return a new RemoteFunction with per-call overrides.

        Examples:
            >>> @remote()
            ... def f(x): return x * 2
            ...
            >>> g = f.options(num_cpus=2.0)
        """
        return RemoteFunction(
            fn=self.fn,
            num_cpus=self.num_cpus if num_cpus is None else float(num_cpus),
            num_gpus=self.num_gpus if num_gpus is None else float(num_gpus),
            resources=self.resources if resources is None else dict(resources),
            runtime_env=self.runtime_env if runtime_env is None else runtime_env,
            placement_group=self.placement_group if placement_group is None else placement_group,
            bundle_index=self.bundle_index if bundle_index is None else bundle_index,
            pinned_node_id=self.pinned_node_id if pinned_node_id is None else pinned_node_id,
            max_concurrency=self.max_concurrency if max_concurrency is None else max_concurrency,
        )

    def task(self, *args: Any, **kwargs: Any) -> Task:
        """
        Build a `Task` for this function call without submitting it.

        Returns:
            Task: Declarative description of the remote call.

        Examples:
            >>> @remote()
            ... def add(x, y): return x + y
            ...
            >>> task = add.task(3, 4)  # create Task
        """
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
        """
        Submit this function call to the global session's scheduler.

        Returns:
            Optional[ObjectRef]: Reference to the result if placed now; otherwise `None`.

        Raises:
            RuntimeError: If a global session is not initialized.

        Examples:
            >>> @remote()
            ... def add(x, y): return x + y
            ...
            >>> # after init()
            >>> r = add.remote(3, 4)   # may be None if queued
        """
        task = self.task(*args, **kwargs)
        sess = get_session()
        return sess.submit(task, blocking=blocking)


def remote(_fn: Optional[Callable[..., Any]] = None, **opts: Any) -> Callable[..., RemoteFunction] | RemoteFunction:
    """
    Decorator/adapter that wraps a Python function into a `RemoteFunction`.

    Usage patterns:
        @remote(num_cpus=1.0)
        def f(x): ...

        # or:
        def g(x): ...
        g = remote(g, num_cpus=2.0)

    Args:
        _fn (Optional[Callable]): The function to wrap when used as `remote(fn, ...)`.
        **opts: Options forwarded to `RemoteFunction` (num_cpus, num_gpus, resources, runtime_env).

    Returns:
        RemoteFunction or a decorator that produces one.

    Examples:
        >>> @remote(num_cpus=2.0)
        ... def mul(x, y): return x * y
        ...
        >>> # submit immediately (requires init())
        >>> r = mul.remote(6, 7)
        >>> # or build a task without submitting:
        >>> task = mul.task(6, 7)
    """

    def _wrap(fn: Callable[..., Any]) -> RemoteFunction:
        return RemoteFunction(fn=fn, **opts)

    if _fn is None:
        return _wrap
    return _wrap(_fn)


def actor(cls: type):
    """
    Public re-export for actor class decorator.
    """
    return actor_class(cls)
