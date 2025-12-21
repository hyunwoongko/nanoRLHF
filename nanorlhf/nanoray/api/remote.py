import inspect
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from typing import Callable, Union

from nanorlhf.nanoray.api.session import get_session
from nanorlhf.nanoray.core.object_ref import ObjectRef
from nanorlhf.nanoray.core.placement import PlacementGroup
from nanorlhf.nanoray.core.runtime_env import RuntimeEnv
from nanorlhf.nanoray.core.task import Task
from nanorlhf.nanoray.utils import new_task_id


@dataclass(frozen=True)
class ActorRef:
    actor_id: str
    owner_node_id: str

    def __getattr__(self, method_name: str) -> "ActorMethod":
        # IMPORTANT: never intercept special/dunder names.
        # Pickle accesses __reduce__/__reduce_ex__/__getstate__/...;
        # if we return an ActorMethod for those, pickling breaks.
        if method_name.startswith("__") and method_name.endswith("__"):
            raise AttributeError(method_name)

        return ActorMethod(self, method_name)


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

    def __post_init__(self) -> None:
        if self.resources is None:
            object.__setattr__(self, "resources", {})

    def options(
        self,
        *,
        num_cpus: float = 0.0,
        num_gpus: float = 0.0,
        resources: Optional[Dict[str, float]] = None,
        pinned_node_id: Optional[str] = None,
        runtime_env: Optional[RuntimeEnv] = None,
        placement_group: Optional[PlacementGroup] = None,
        bundle_index: Optional[int] = None,
        max_concurrency: Optional[int] = 1,
    ) -> "ActorClass":
        return ActorClass(
            cls=self.cls,
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            resources=(resources or {}),
            pinned_node_id=pinned_node_id,
            placement_group=placement_group,
            bundle_index=bundle_index,
            runtime_env=runtime_env,
            max_concurrency=max_concurrency,
        )

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


@dataclass(frozen=True)
class ActorMethod:
    ref: ActorRef
    method_name: str
    num_cpus: float = 0.0
    num_gpus: float = 0.0
    resources: Dict[str, float] = field(default_factory=dict)
    pinned_node_id: Optional[str] = None
    placement_group: Optional[PlacementGroup] = None
    bundle_index: Optional[int] = None
    runtime_env: Optional[RuntimeEnv] = None
    max_concurrency: Optional[int] = 1

    def __post_init__(self) -> None:
        if self.resources is None:
            object.__setattr__(self, "resources", {})

    def options(
        self,
        *,
        num_cpus: float = 0.0,
        num_gpus: float = 0.0,
        resources: Optional[Dict[str, float]] = None,
        runtime_env: Optional[RuntimeEnv] = None,
        placement_group: Optional[PlacementGroup] = None,
        bundle_index: Optional[int] = None,
        max_concurrency: Optional[int] = 1,
    ) -> "ActorMethod":
        return ActorMethod(
            self.ref,
            self.method_name,
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            resources=(resources or {}),
            placement_group=placement_group,
            bundle_index=bundle_index,
            runtime_env=runtime_env,
            max_concurrency=max_concurrency,
        )

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
            num_cpus=self.num_cpus,
            num_gpus=self.num_gpus,
            resources=dict(self.resources),
            runtime_env=self.runtime_env,
            placement_group_id=self.placement_group.pg_id if self.placement_group else None,
            bundle_index=self.bundle_index,
            pinned_node_id=self.ref.owner_node_id,
            max_concurrency=self.max_concurrency,
            task_id=new_task_id(),
        )
        return sess.submit(task, blocking=blocking)


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


def remote(obj: Optional[Union[type, Callable[..., Any]]] = None, **opts: Any):
    def _wrap(x: Union[type, Callable[..., Any]]):
        if inspect.isclass(x):
            return ActorClass(cls=x, **opts)
        else:
            return RemoteFunction(fn=x, **opts)

    return _wrap if obj is None else _wrap(obj)
