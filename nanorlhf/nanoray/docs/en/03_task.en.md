# Task

## Overview

A Task is the basic unit that represents remote execution in a distributed runtime. Instead of executing a local function call immediately, it turns the call into an execution specification and submits it to the scheduler. The scheduler decides the execution node and execution timing based on this specification.

## Role of Task

A Task has one role: to turn a function call into a schedulable form. In other words, it converts the call into an input the scheduler can reason about, rather than executing it directly.

## Components

A Task contains only two kinds of information.

- Call information: what to execute
- Constraint information: what conditions must be satisfied to execute

It is not important to memorize the exact field names. What matters is that the call and its constraints move together as one bundle and become the scheduler’s decision input.

```python
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generic, Optional, Tuple, TypeVar, Union

from nanorlhf.nanoray.core.runtime_env import RuntimeEnv
from nanorlhf.nanoray.utils import new_task_id

T = TypeVar("T")


@dataclass(frozen=True)
class Task(Generic[T]):
    # identify & call
    task_id: str
    fn: Union[Callable[..., T], dict]
    args: Tuple[Any, ...] = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)

    # resources & context
    num_cpus: float = 1.0
    num_gpus: float = 0.0
    resources: Optional[Dict[str, Any]] = None
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
        resources: Optional[Dict[str, Any]] = None,
        runtime_env: Optional[RuntimeEnv] = None,
        pinned_node_id: Optional[str] = None,
        max_concurrency: int = 1,
        placement_group_id: Optional[str] = None,
        bundle_index: Optional[int] = None,
    ) -> "Task[T]":
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
```

## Scheduling Model

In a distributed runtime, a call is typically processed in the flow below.

- Task creation: bundle the call and constraints into a specification
- Scheduling: the scheduler decides placement and execution timing based on the cluster state
- Execution: a worker actually runs the function

This separation is essential. In distributed systems, execution location and timing depend on runtime conditions, so the system does not commit to execution first; it builds the specification first, then decides.

## Resource Constraints

num_cpus, num_gpus, and resources are feasibility conditions for execution. The scheduler considers only nodes that satisfy these conditions as candidates. If the conditions cannot be met, the call cannot become an execution candidate.

## Placement Constraints

A Task can express intent about where it should run through placement constraints.

- pinned_node_id: force a specific node
- placement_group_id and bundle_index: consume a specific slot of a PlacementGroup

## Immutability

A Task is passed to the scheduler and workers, and multiple components share the same specification. If the specification changes after a scheduling decision, the system behavior becomes unstable. Therefore, a Task must be an immutable record.

## Task: 3-line Summary

- A Task represents a remote function call as an execution specification.
- The scheduler makes execution decisions based on the Task’s resource constraints and placement constraints.
- pinned_node_id is the strongest placement constraint and takes precedence over a PlacementGroup.