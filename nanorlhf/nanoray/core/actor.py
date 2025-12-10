from dataclasses import dataclass, field, replace
from typing import Any, Dict, Optional

from nanorlhf.nanoray.api.session import get_session
from nanorlhf.nanoray.core.placement import PlacementGroup
from nanorlhf.nanoray.core.runtime_env import RuntimeEnv
from nanorlhf.nanoray.core.task import Task
from nanorlhf.nanoray.utils import new_task_id


@dataclass(frozen=True)
class ActorRef:
    """
    A lightweight, immutable reference to a stateful remote object (actor).

    Attributes:
        actor_id (str): Stable id for this actor within its owning worker
        owner_node_id (str): The node id where the actor instance lives.

    Discussion:
        Q. What is an "Actor" in this runtime?
            An Actor is a *stateful* remote object that is created on a node and
            remains *sticky* to that node for its entire lifetime. All method calls
            are routed back to its owner node.

            (De)serialization and loading states like models, environments,
            or caches is expensive. If we used only stateless tasks,
            each call would need to (de)serialize and reload the state,
            leading to very high overhead and latency.

            Actor resolves problems like these by allowing users to create
            long-lived stateful objects on a node and invoke methods on them
            without repeated (de)serialization/loading.

        Q. How is an Actor different from a regular `Task`?
            - `Task`: stateless function execution; the scheduler may place it
                      on any eligible node. The lifetime is a single call.

            - `Actor`: stateful object; placed once (on creation) and subsequent
                       calls always run where the object lives (sticky placement).
                       Lifetime spans many calls until explicitly terminated or the worker exits.

            - The trade-off: Actors avoid repeated (de)serialization/loading of
                             large state (models, envs, caches), but introduce
                             lifecycle and routing semantics.

        Q. When to use an Actor vs. a regular Task?
            Use an Actor when:
                - You have a large state (model, env, cache) that is expensive to
                    (de)serialize and load repeatedly.
                - You need to maintain state across multiple method calls.

            Use a regular Task when:
                - The function is stateless and can run anywhere.
                - The function is lightweight and does not involve large state.
                - You want maximum flexibility in scheduling and placement.
                - You want to avoid the overhead of managing actor lifecycles.

        Q. Why separate and `ActorRef` from actual instance?
            To enable location transparency and routing. The ref carries the
            minimum metadata (id + owner) to route method calls without exposing
            the in-process object.

        Q. Is this like an `ObjectRef`?
            Similar spirit (an immutable object), but it points to a *living*
            object rather than a materialized value in the object store.
    """

    actor_id: str
    owner_node_id: str

    def __getattr__(self, method_name: str) -> "ActorMethod":
        """
        Return a lightweight proxy that exposes `.remote(...)` to invoke
        `method_name` on this actor.

        Examples:
            >>> h = CounterClass.remote()
            >>> r = h.inc.remote(3)  # enqueue a call to `Counter.inc(3)`

        Args:
            method_name (str): The name of the method to invoke on the actor.

        Returns:
            ActorMethod: A proxy to invoke the method remotely.

        Discussion:
            Q. Does this eagerly resolve the actor?
                No. It only builds a proxy carrying (ref, method_name).
                actual scheduling happens when you call `.remote` on it.
        """

        # IMPORTANT: never intercept special/dunder names.
        # Pickle accesses __reduce__/__reduce_ex__/__getstate__/...;
        # if we return an ActorMethod for those, pickling breaks.
        if method_name.startswith("__") and method_name.endswith("__"):
            raise AttributeError(method_name)

        return ActorMethod(self, method_name)


@dataclass(frozen=True)
class ActorOptions:
    """
    Options for actor creation and method calls.

    Attributes:
        num_cpus (float): CPU requirement (default `0.0`).
        num_gpus (float): GPU requirement (default `0.0`).
        resources (Dict[str, float]): Custom resources (default `{}`).
        pinned_node_id (Optional[str]): If set, the actor will be created on this node.
        placement_group (Optional[PlacementGroup]): Placement group.
        bundle_index (Optional[int]): Bundle index for SPREAD strategy.
        runtime_env (Optional[RuntimeEnv]): Per-call runtime environment.
    """
    num_cpus: float = 0.0
    num_gpus: float = 0.0
    resources: Dict[str, float] = field(default_factory=dict)

    # placement / pinning
    pinned_node_id: Optional[str] = None                  # ActorClass.create 전용
    placement_group: Optional[PlacementGroup] = None      # object in → .id passed to Task
    bundle_index: Optional[int] = None

    # env
    runtime_env: Optional[RuntimeEnv] = None


@dataclass(frozen=True)
class ActorMethod:
    """
    A tiny proxy for a single actor method.
    Use `.remote(*args, **kwargs)` to submit the call via the global session.

    Discussion:
        Q. How does calling an actor method differ from a calling remote function?
            Both return an `ObjectRef`, but actor method calls carry a *pinning constraints*;
            they must execute on the actor's owner node. We encode this by setting `pinned_node_id`
            in the `Task` submitted to the scheduler.
    """

    ref: ActorRef
    method_name: str
    _opts: ActorOptions = ActorOptions()

    def options(
        self,
        *,
        num_cpus: float = 0.0,
        num_gpus: float = 0.0,
        resources: Optional[Dict[str, float]] = None,
        runtime_env: Optional[RuntimeEnv] = None,
        placement_group: Optional[PlacementGroup] = None,
        bundle_index: Optional[int] = None,
    ) -> "ActorMethod":
        """
        Configure scheduling options for this method call.

        Args:
            num_cpus (float): CPU requirement (default `0.0`).
            num_gpus (float): GPU requirement (default `0.0`).
            resources (Optional[Dict[str, float]]): Custom resources (default `{}`).
            runtime_env (Optional[RuntimeEnv]): Per-call runtime environment.
            placement_group (Optional[PlacementGroup]): Placement group.
            bundle_index (Optional[int]): Bundle index for SPREAD strategy.
        """
        return ActorMethod(
            self.ref,
            self.method_name,
            _opts=replace(
                self._opts,
                num_cpus=num_cpus,
                num_gpus=num_gpus,
                resources=(resources or {}),
                placement_group=placement_group,
                bundle_index=bundle_index,
                runtime_env=runtime_env,
            ),
        )

    def remote(self, *args: Any, blocking: bool = False, **kwargs: Any):
        """
        Submit an actor method call to the scheduler (returns an ObjectRef)

        Returns:
            ObjectRef: A reference to the result of the method call.

        Discussion:
            Q. How is 'sticky placement' ensured?
                We carry `pinned_node_id=ref.owner_node_id` inside the `Task`,
                so the scheduler only considers that node as a valid candidate.
        """
        sess = get_session()
        o = self._opts
        task = Task(
            fn={
                "kind": "actor_call",
                "actor_id": self.ref.actor_id,
                "method": self.method_name,
            },
            args=args,
            kwargs=kwargs,
            num_cpus=o.num_cpus,
            num_gpus=o.num_gpus,
            resources=o.resources,
            pinned_node_id=self.ref.owner_node_id,  # hard pin to owner
            placement_group_id=(o.placement_group.pg_id if o.placement_group else None),
            bundle_index=o.bundle_index,
            runtime_env=o.runtime_env,
            task_id=new_task_id(),
        )
        return sess.submit(task, blocking=blocking)


class ActorClass:
    """
    Thin wrapper returned by `@actor` that exposes `.remote(...)` to construct
    an Actor and returns an `ActorRef` (via `ObjectRef`) through the session.
    """

    def __init__(self, cls: type):
        self._cls = cls
        self._opts = ActorOptions()

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
    ) -> "ActorClass":
        """
        Configure scheduling options for the actor constructor task.

        Args:
            num_cpus (float): CPU requirement (default `0.0`).
            num_gpus (float): GPU requirement (default `0.0`).
            resources (Optional[Dict[str, float]]): Custom resources (default `{}`).
            pinned_node_id (Optional[str]): If set, the actor will be created on this node.
            runtime_env (Optional[RuntimeEnv]): Runtime environment to apply during construction.
            placement_group (Optional[PlacementGroup]): Placement group.
            bundle_index (Optional[int]): Bundle index for SPREAD strategy.

        Returns:
            ActorClass: A new proxy with updated options.
        """
        out = ActorClass(self._cls)
        out._opts = replace(
            self._opts,
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            resources=(resources or {}),
            pinned_node_id=pinned_node_id,
            placement_group=placement_group,
            bundle_index=bundle_index,
            runtime_env=runtime_env,
        )
        return out

    def remote(self, *args: Any, blocking: bool = False, **kwargs: Any):
        """
        Enqueue an actor creation task. The returned `ObjectRef` resolves
        to and `ActorRef(actor_id, owner_node_id)`.

        Returns:
            ObjectRef: Reference to the created actor ref.
        """
        sess = get_session()
        o = self._opts

        task = Task(
            fn={
                "kind": "actor_create",
                "cls": self._cls,
                "args": args,
                "kwargs": kwargs,
            },
            args=(),
            kwargs={},
            num_cpus=o.num_cpus,
            num_gpus=o.num_gpus,
            resources=o.resources,
            task_id=new_task_id(),
            pinned_node_id=o.pinned_node_id,
            placement_group_id=(o.placement_group.pg_id if o.placement_group else None),
            bundle_index=o.bundle_index,
            runtime_env=o.runtime_env,
        )
        return sess.submit(task, blocking=blocking)


def actor(cls: type) -> ActorClass:
    """
    Class decorator to declare a stateful remote actor.

    Args:
        cls (type): User-defined class to be used as an actor.

    Returns:
        ActorClass: A thin wrapper exposing `.remote(...)` to create an actor.

    Examples:
        >>> from nanorlhf.nanoray.core.actor import actor
        >>> from nanorlhf.nanoray.api.session import get
        >>> @actor
        >>> class Counter:
        ...     def __init__(self):  self.x = 0
        ...     def inc(self, v):   self.x += v; return self.x
        >>> h_ref = Counter.remote()  # ObjectRef -> ActorRef
        >>> h = get(h_ref)  # ActorRef
        >>> r = h.inc.remote(3)  # ObjectRef -> 3
        >>> get(r)
        3

    Discussion:
        Q. Why not return `ActorRef` directly from `.remote()`?
            Our scheduler contracts return `ObjectRef` for tasks.
            Keeping one return form simplifies the mental model.
            Users call `get()` to obtain the handle, just like any other result.
    """
    return ActorClass(cls)
