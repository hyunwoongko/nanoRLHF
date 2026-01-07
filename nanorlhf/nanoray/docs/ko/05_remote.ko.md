# remote, RemoteFunction, ActorClass

## 개요

Ray에서 원격 실행은 보통 두 형태로 시작됩니다.

- 함수: 한 번 호출하고 끝나는 원격 함수 실행
- 클래스: 한 번 생성한 뒤 메서드를 여러 번 호출하는 Actor 실행

이 프로젝트는 두 형태를 같은 표면 API로 다루기 위해 remote 데코레이터를 제공합니다. 사용자는 remote를 함수에 붙이거나 클래스에 붙이고, 이후 remote 호출로 실행을 시작합니다. 내부에서는 그 호출이 Task로 변환되어 세션에 제출됩니다.

## 표면 API

사용자가 보는 사용법은 아래 형태입니다.

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

여기서 중요한 점은 두 경우 모두 remote 호출이 즉시 결과 값을 주지 않고, 세션에 작업을 제출한다는 점입니다.

## 핵심 아이디어

이 프로젝트는 remote 데코레이터가 입력 대상에 따라 두 래퍼 중 하나를 만든다고 보면 됩니다.

- 함수면 RemoteFunction으로 감싼다
- 클래스면 ActorClass로 감싼다

```python
def remote(obj: Optional[Union[type, Callable[..., Any]]] = None, **opts: Any):
    def _wrap(x: Union[type, Callable[..., Any]]):
        if inspect.isclass(x):
            return ActorClass(cls=x, **opts)
        else:
            return RemoteFunction(fn=x, **opts)

    return _wrap if obj is None else _wrap(obj)
```

즉, remote는 실행을 직접 하지 않습니다. 대신 사용자가 이후에 remote 호출을 할 수 있게 실행 가능한 핸들 객체를 만들어 줍니다.

## RemoteFunction

RemoteFunction은 원격 함수 실행의 핸들입니다. 중요한 역할은 하나입니다.

- 함수 호출을 Task로 바꿔서 제출한다

사용자가 f.remote(...)를 호출하면 내부적으로는 아래 흐름입니다.

- f.remote(args, kwargs)
- Task 생성
- 세션 submit

RemoteFunction의 핵심은 task 메서드가 Task를 만든다는 점입니다.

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

여기서 독자가 가져가야 하는 메시지는 이것입니다.

RemoteFunction은 함수 호출을 직접 실행하지 않고, Task로 바꿔 스케줄링 가능한 형태로 만든 뒤 제출한다

## ActorClass

ActorClass는 클래스 기반 원격 실행의 핸들입니다. 사용자는 클래스에 remote를 붙이면 ActorClass를 얻고, 그 ActorClass의 remote로 인스턴스 생성 요청을 보냅니다.

이 프로젝트에서 Actor 생성 역시 Task로 표현됩니다. 다만 fn 자리에 일반 호출 가능한 함수가 아니라 "kind가 actor_create인 딕셔너리"를 넣어, 워커가 이것을 Actor 생성 요청으로 해석하게 만듭니다.

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

즉 ActorClass.remote는 "이 클래스로 Actor를 만들어 달라"는 요청을 Task로 만들어 제출합니다.

## options의 의미

RemoteFunction과 ActorClass 모두 options를 제공합니다. 이 메서드는 실행 제약을 바꾼 새로운 핸들을 돌려줍니다. 중요한 점은 options가 실행을 트리거하지 않는다는 점입니다.

- options는 구성만 바꾼다
- 실행은 remote 호출에서 일어난다

그래서 options는 "앞으로 제출할 Task의 기본 제약을 미리 세팅하는 단계"로 이해하시면 됩니다.

## ActorRef와 ActorMethod

Actor가 생성되면 호출자는 ActorRef를 통해 메서드를 선택하고 원격 호출을 보낼 수 있어야 합니다. 이 프로젝트는 Python의 __getattr__를 사용해 a.inc 같은 접근을 ActorMethod로 바꿉니다.

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

ActorMethod.remote는 메서드 호출 역시 Task로 만들어 제출합니다. 여기서도 fn은 "kind가 actor_call인 딕셔너리"입니다.

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

여기서 중요한 개념은 이것입니다.

ActorMethod.remote는 메서드 호출을 Task로 바꿔 제출하고, 그 Task는 owner_node_id로 실행 위치가 고정된다

## Remote: 3줄 요약

- remote는 함수와 클래스를 각각 RemoteFunction과 ActorClass로 감싸는 팩토리입니다.
- RemoteFunction.remote는 함수 호출을 Task.from_call로 바꿔 제출합니다.
- ActorClass.remote는 Actor 생성 요청을 kind가 actor_create인 Task로 바꿔 제출합니다. ActorRef는 메서드 접근을 ActorMethod로 바꾸고, ActorMethod.remote는 kind가 actor_call인 Task를 제출합니다.