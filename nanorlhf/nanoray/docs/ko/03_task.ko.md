# Task

## 개요

Task는 분산 런타임에서 원격 실행을 표현하는 기본 단위입니다. 로컬 함수 호출을 바로 실행하는 대신, 호출을 실행 명세로 바꿔 스케줄러에 제출합니다. 스케줄러는 이 명세를 바탕으로 실행 노드와 실행 시점을 결정합니다.

## Task의 역할

Task의 역할은 하나입니다. 함수 호출을 스케줄링 가능한 형태로 만드는 것입니다. 즉, 실행 그 자체가 아니라 스케줄러가 판단할 수 있는 입력으로 바꿉니다.

## 구성 요소

Task에는 크게 두 종류의 정보만 들어 있습니다.

- 호출 정보: 무엇을 실행할지
- 제약 정보: 어떤 조건을 만족해야 실행할지

세부 필드 이름을 외우는 것이 중요하지 않습니다. 중요한 것은 호출과 제약이 한 묶음으로 이동하며 스케줄러의 의사결정 입력이 된다는 점입니다.

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

## 스케줄링 모델

분산 런타임에서 호출은 보통 아래 흐름으로 처리됩니다.

- Task 생성: 호출과 제약을 하나로 묶어 명세를 만든다
- 스케줄링: 스케줄러가 클러스터 상태를 보고 배치와 실행 시점을 결정한다
- 실행: 워커가 실제 함수를 실행한다

분산에서는 실행 위치와 타이밍이 런타임 상황에 의해 달라지므로, 실행을 먼저 확정하지 않고 명세를 먼저 만든 뒤 결정합니다.

## 자원 제약

num_cpus, num_gpus, resources는 실행 가능 조건입니다. 스케줄러는 이 조건을 만족하는 노드만 후보로 고려합니다. 조건을 못 맞추면 실행 후보가 될 수 없습니다.

## 배치 제약

Task는 배치 제약을 통해 어디에서 실행될지에 대한 의도를 표현할 수 있습니다.

- pinned_node_id: 특정 노드 강제
- placement_group_id와 bundle_index: PlacementGroup의 특정 슬롯 소비

## 불변성

Task는 스케줄러와 워커로 전달되며 여러 컴포넌트가 같은 명세를 공유합니다. 스케줄링 결정 이후 명세가 바뀌면 시스템 동작이 흔들립니다. 그래서 Task는 불변 레코드여야 합니다.

## Task: 3줄 요약

- Task는 원격 함수 호출을 실행 명세로 표현하는 단위입니다
- 스케줄러는 Task의 자원 제약과 배치 제약을 보고 실행 결정을 내립니다
- pinned_node_id는 가장 강한 배치 제약이며 PlacementGroup보다 우선합니다