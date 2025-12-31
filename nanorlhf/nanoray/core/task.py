from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generic, Optional, Tuple, TypeVar, Union

from nanorlhf.nanoray.core.runtime_env import RuntimeEnv
from nanorlhf.nanoray.utils import new_task_id

T = TypeVar("T")


@dataclass(frozen=True)
class Task(Generic[T]):
    """
    `Task` is an immutable data record that describes a single remote function call.
    Instead of invoking a Python function immediately, the runtime the call as a declarative
    task and submits it to the scheduler for placement and execution.

    Attributes:
        task_id (str): A globally unique id (e.g., `"tsk-9ff8c3a2"`).
        fn (Union[Callable[..., T], dict]): The Python function to execute remotely.
        args (Tuple[Any, ...]): Positional arguments to pass to `fn`.
        kwargs (Dict[str, Any]): Keyword arguments to pass to `fn`.
        num_cpus (float): CPU resource requirement (default 1.0).
        num_gpus (float) : GPU resource requirement (default 0.0).
        resources (Optional[Dict[str, float]]): Custom resources (e.g., {"ram_gb": 4.0}).
        runtime_env (Optional[RuntimeEnv]): Optional environment description.
        placement_group_id (Optional[str]): If set, this task belongs to a PG.
        bundle_index (Optional[int]): Which bundle of that PG this task consumes.

    Examples:
        >>> def add(x, y): return x + y
        >>> task = Task(fn=add, args=(1, 2), kwargs={}, num_cpus=0.5)
        >>> print(task.fn is add, task.args, task.num_cpus)
        True (1, 2) 0.5

    Discussion:
        Q. Why keep a `Task` instead of calling the function right away?
            In a distributed runtime, *when* and *where* to execute matters.
            By turning a call into a task, the scheduler can choose the best node
            (considering resources and placement groups) before any execution happens.

        Q. Why immutable?
            Once submitted, the task is shared across components (driver, scheduler, worker).
            Making it immutable prevents accidental mutation after scheduling decisions.

        Q. What happens if both `pinned_node_id` and PG are set?
            `pinned_node_id` takes precedence (most specific constraint).
    """

    # identify & call
    task_id: str
    fn: Union[Callable[..., T], dict]
    args: Tuple[Any, ...] = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)

    # resources & context
    num_cpus: float = 1.0
    num_gpus: float = 0.0
    resources: Optional[Dict[str, float]] = None
    runtime_env: Optional[RuntimeEnv] = None
    pinned_node_id: Optional[str] = None
    max_concurrency: int = 1

    # placement group (if any)
    placement_group_id: Optional[str] = None
    bundle_index: Optional[int] = None

    @classmethod
    def from_call(
        cls,
        fn: Union[Callable[..., T], dict],
        args: Tuple[Any, ...] = (),
        kwargs: Optional[Dict[str, Any]] = None,
        *,
        num_cpus: float = 1.0,
        num_gpus: float = 0.0,
        resources: Optional[Dict[str, float]] = None,
        runtime_env: Optional[RuntimeEnv] = None,
        pinned_node_id: Optional[str] = None,
        max_concurrency: int = 1,
        placement_group_id: Optional[str] = None,
        bundle_index: Optional[int] = None,
    ) -> "Task[T]":
        """
        Build a `Task` from a Python callable and its arguments.

        Args:
            fn (Union[Callable[..., T], dict]): The function to run remotely.
            args (Tuple[Any, ...]): Positional arguments.
            kwargs (Optional[Dict[str, Any]]): Keyword arguments (default `{}`).
            num_cpus (float): CPU requirement (default `1.0`).
            num_gpus (float): GPU requirement (default `0.0`).
            resources (Optional[Dict[str, float]]): Custom resources (e.g., `{"ram_gb": 4}`).
            runtime_env (Optional[RuntimeEnv]): Optional runtime environment task.
            pinned_node_id (Optional[str]): If set, the task will only run on this node.
            max_concurrency (int): Max concurrent calls if this is an actor creation task.
            placement_group_id (Optional[str]): If set, this task belongs to a PG.
            bundle_index (Optional[int]): Which bundle of that PG this task consumes.

        Returns:
            Task[T]: A new immutable task specification.

        Examples:
            >>> def mul(a, b): return a * b
            >>> Task.from_call(mul, (3, 4)).args
            (3, 4)
        """
        return cls(
            task_id=new_task_id(),
            fn=fn,
            args=args,
            kwargs={} if kwargs is None else kwargs,
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            resources=resources,
            runtime_env=runtime_env,
            pinned_node_id=pinned_node_id,
            max_concurrency=max_concurrency,
            placement_group_id=placement_group_id,
            bundle_index=bundle_index,
        )

