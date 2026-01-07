# Scheduler

## 개요

Scheduler는 Task를 받아서 실행할 노드(node_id)를 고르고, 그 노드의 Worker에게 실행을 맡기는 컴포넌트입니다. 분산 런타임에서 핵심은 실행 그 자체보다 배치 결정을 일관되게 만드는 일인데, Scheduler가 그 결정을 책임집니다.

이 프로젝트에서 Scheduler는 다음을 수행합니다.

- Task를 입력으로 받습니다
- 실행 가능한 후보 노드를 계산합니다
- SchedulingPolicy로 후보 중 하나를 선택합니다
- 선택된 노드의 Worker.execute_task를 호출합니다
- 생성된 ObjectRef를 반환합니다

## 전체 관계

```text
RemoteFunction / ActorClass / ActorMethod
  -> Task 생성
  -> Session.submit(Task)
  -> Scheduler.submit(Task)
     -> eligible_nodes(Task)
     -> policy.select(candidates)
     -> Worker.execute_task(Task)
     -> ObjectRef 반환
```

Scheduler는 실행 엔진이 아닙니다. 실행은 Worker가 합니다. Scheduler는 배치 결정을 내리고, 실행을 호출하는 역할입니다.

## 실행 흐름

Scheduler.submit은 즉시 실행을 시도하고, 안 되면 큐에 넣습니다.

```text
1) submit(Task)
   - try_place(Task)로 지금 바로 실행 가능한지 확인
   - 가능하면 즉시 Worker.execute_task 호출 -> ObjectRef 반환
   - 불가능하면 (seq, Task)를 큐에 넣고 None 반환

2) drain()
   - 큐가 빌 때까지 또는 더는 진전이 없을 때까지 반복
   - 매 라운드마다 큐의 Task를 순서대로 한 번씩만 배치 시도
```

이 구조 덕분에 Scheduler는 다음 성질을 가집니다.

- 즉시 실행 가능한 Task는 지연 없이 실행됩니다
- 당장 실행 불가능한 Task는 Policy에서 지정한 순서(FIFO or RoundRobin)로 대기합니다
- 리소스가 풀리면 다음 라운드에서 다시 시도합니다

## 후보 노드 계산

Scheduler의 핵심은 eligible_nodes입니다. 여기서 Task가 어느 노드에서 실행 가능한지를 계산합니다. 계산에 쓰는 제약은 크게 세 가지입니다.

- 리소스 제약: CPU, GPU, 커스텀 리소스
- pinned_node_id: 특정 노드 강제
- PlacementGroup: PACK, SPREAD 의도 반영

pinned_node_id가 있으면 후보는 사실상 0개 또는 1개가 됩니다. 존재하지 않는 노드이거나 리소스가 안 맞으면 후보가 비어버립니다.

PlacementGroup이 있으면 bundle_index 기준으로 해당 번들이 어느 노드에서 돌아야 하는지가 결정되거나, PACK이면 그룹 전체가 한 노드로 잠깁니다.

## SchedulingPolicy

SchedulingPolicy는 후보 노드 목록이 주어졌을 때 그중 하나를 고르는 전략입니다. Scheduler는 정책에게 후보를 주고, 정책이 선택한 노드를 그대로 사용합니다.

- FIFO: 전역 노드 순서에서 후보에 포함되는 첫 노드를 선택합니다
- RoundRobin: 전역 노드 순서를 원형으로 돌며 후보를 번갈아 선택합니다

정책이 하는 일은 단순합니다. 배치 가능 여부 판단은 Scheduler가 하고, 후보 중 선택만 정책이 합니다.

## 큐잉 모델

Scheduler는 내부 큐를 힙으로 들고 있습니다. 우선순위는 seq이며, seq가 증가하므로 사실상 FIFO입니다.

- submit에서 즉시 실행이 실패하면 (seq, task)를 push합니다
- drain에서 pop하면서 순서대로 배치 시도합니다

drain은 라운드 기반입니다. 한 라운드에서 단 하나도 배치되지 않으면 다음 라운드도 똑같을 가능성이 높으므로 종료합니다. 그래서 무한 루프 없이 결정적으로 끝납니다.

## NodeState

NodeState는 각 노드의 현재 리소스 상태를 추적합니다.

- total: 노드가 가진 총량
- used: 실행 중 Task가 실제로 쓰고 있는 양
- reserved: PlacementGroup 같은 이유로 미리 잡아둔 양

일반 Task는 allocate 후 실행하고, 실행이 끝나면 release합니다. 이 프로젝트는 Worker.execute_task 호출 전후로 allocate, release를 감쌉니다.

PlacementGroup이 있는 경우는 조금 다릅니다. PACK, SPREAD는 번들 단위로 reserve_bundle을 수행하고, 그 예약이 풀리기 전까지 같은 정책을 유지합니다. 즉, 예약은 Task 하나의 수명보다 길 수 있습니다.

## PlacementGroup 처리

PlacementGroup이 있으면 Scheduler는 placement_group_assignment에 배치 결정을 기록합니다.

- PACK
  - `__pack__` 키에 잠긴 노드 id를 기록합니다
  - 최초 배치 시 그룹의 모든 번들을 한 노드에 reserve_bundle로 잡고 잠급니다
- SPREAD
  - bundle_index별로 어떤 노드에 갔는지 기록합니다
  - 아직 배정되지 않은 번들은 가능한 노드 중 사용되지 않은 노드를 선호합니다

중요한 점은 PlacementGroup이 "의도"라는 사실입니다. Scheduler는 이를 후보 계산과 예약으로 구체화해서, 이후 Task 배치가 흔들리지 않게 만듭니다.

## WorkerLike

Scheduler는 WorkerLike라는 최소 인터페이스만 알면 됩니다. 즉, Scheduler는 실행이 로컬 Worker인지 원격 프록시인지 구분할 필요가 없습니다. execute_task만 가능하면 됩니다.

이 설계는 배치 로직을 실행 전송 방식에서 분리해줍니다. Scheduler가 실행 방식에 독립적이어야 한다는 메시지가 핵심입니다.

## Scheduler: 3줄 요약

- Scheduler는 Task를 받아 실행 노드를 결정하고 Worker.execute_task를 호출합니다
- 배치 가능 여부는 eligible_nodes가 판단하고, 후보 선택은 SchedulingPolicy가 담당합니다
- NodeState가 리소스 사용량과 예약량을 추적하며, PlacementGroup은 예약과 배치 기록으로 의도를 고정합니다