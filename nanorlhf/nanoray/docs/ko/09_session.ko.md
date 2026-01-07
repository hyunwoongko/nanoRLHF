# Session

## 개요

Session은 드라이버(driver) 쪽에서 런타임을 대표하는 객체입니다. 사용자가 remote 호출을 만들고 submit 하고 get 하는 흐름은 전부 Session을 통해 들어옵니다. Session의 핵심 역할은 실행을 직접 수행하는 것이 아니라, Scheduler를 통해 실행을 전개하고 결과를 회수하는 드라이버 조정자가 되는 것입니다.

Session이 맡는 책임은 세 가지입니다.

- Task를 Scheduler에 제출합니다
- 필요하면 drain으로 Scheduler 큐를 전개합니다
- ObjectRef로부터 값을 가져오는 get 경로를 제공합니다

중요한 점은 Worker.execute_task 호출은 Session이 하지 않는다는 사실입니다. Worker.execute_task는 Scheduler 내부에서 배치 결정이 끝난 뒤 실행 단계에서 호출됩니다.

## 드라이버

드라이버는 분산 런타임에서 사용자 프로그램이 돌아가는 중심 프로세스를 뜻합니다. 보통 다음 역할을 맡습니다.

- 사용자 코드를 실행하며 Task를 생성합니다
- Task를 런타임에 제출하고 실행을 전개합니다
- ObjectRef를 통해 결과를 조회하고 다음 작업을 구성합니다

Ray를 떠올리면, 여러분이 python 스크립트를 실행하는 그 프로세스가 드라이버입니다. Worker는 실제 실행을 수행하는 실행자이고, 드라이버는 실행자를 직접 관리하는 대신 스케줄러와 런타임 API를 통해 실행을 조정합니다. 이 프로젝트에서 Session은 드라이버의 런타임 핸들 역할을 합니다.

## 전체 관계

```text
User code (driver process)
  -> remote(...) 가 Task를 만든다
  -> Session.submit(Task)
       -> Scheduler.submit(Task)
            -> (가능하면) Scheduler.try_place(Task)
                 -> eligible_nodes(Task)
                 -> policy.select(candidates)
                 -> Worker.execute_task(Task) 또는 RemoteWorkerProxy.execute_task(Task)
                 -> ObjectRef 반환
            -> (불가능하면) 큐에 넣는다

  -> Session.drain()
       -> Scheduler.drain()
            -> (큐를 돌며)
            -> Scheduler.try_place(Task)
                 -> Worker.execute_task(...) 호출
                 -> ObjectRef 리스트 생성

  -> Session.get(ObjectRef)
       -> 로컬 조회
       -> (필요하면) Scheduler.drain()로 실행 전개
       -> (필요하면) router+rpc로 원격 fetch 후 로컬 캐시
```

Session은 드라이버 내부에서 Scheduler를 구동하는 창구입니다. 실행 호출은 Scheduler 내부에서 발생합니다.

## submit과 drain

Session.submit은 Task를 Scheduler에 전달합니다. Scheduler는 즉시 실행을 시도할 수 있고, 안 되면 큐에 넣습니다.

- submit(task)
  - scheduler.submit(task)을 호출합니다
  - 지금 배치가 가능하면 ObjectRef가 바로 반환될 수 있습니다
  - 배치가 불가능하면 Task는 Scheduler 큐에 들어가고 None이 반환됩니다

Session.drain은 큐에 남아 있는 Task들을 실행 가능한 만큼 실행하도록 Scheduler를 전개합니다.

- drain()
  - scheduler.drain()을 호출합니다
  - 큐의 Task들을 순서대로 꺼내 배치 시도합니다
  - 배치되는 Task마다 Scheduler 내부에서 Worker.execute_task가 호출됩니다
  - 실행된 결과로 ObjectRef 리스트가 반환됩니다

blocking=True 옵션은 드라이버 편의 기능입니다. 즉시 배치가 실패하면 drain을 돌려서 최소 한 번은 실행을 전개하고, 그중 요청 Task의 결과 ObjectRef를 찾아 get까지 수행합니다. 이 동작은 실행을 직접 하는 것이 아니라 Scheduler를 한 번 더 구동해 결과를 확보하는 방식입니다.

## put

put은 값을 ObjectStore에 넣고 ObjectRef를 돌려줍니다. 로컬 worker가 있으면 해당 worker의 store를 씁니다. 로컬 worker가 없으면 driver_store(ObjectStore("__driver__"))에 넣습니다.

put은 Task 실행과 무관하게, 드라이버가 값을 준비해 두는 기능입니다. 결과적으로 put은 ObjectRef 기반 모델에서 입력 데이터 또는 설정 값을 관리하는 단순한 도구로 보면 됩니다.

## get

Session.get은 ObjectRef를 실제 Python 값으로 바꾸는 경로를 제공합니다. 여기서 중요한 개념은 get이 단순 조회가 아니라 필요하면 스케줄링을 전개하고, 원격이면 네트워크로 가져오고, 가져온 것은 로컬에 캐시한다는 점입니다.

이 프로젝트의 get은 다음 순서로 동작합니다.

```text
get(ref) 조회 순서

0) ref가 None이면
   - Scheduler.drain()을 한 번 호출해서 가장 최근 결과 ref를 선택합니다

1) aliases 캐시 확인
   - remote object_id -> local object_id로 변환된 기록이 있으면
     로컬 캐시 store에서 즉시 조회합니다

2) owner-first (owner가 로컬일 때)
   - ref.owner_node_id가 로컬 worker id면 그 store에서 먼저 찾습니다

3) 로컬 worker 전체 scan
   - 어떤 로컬 store에라도 있으면 그 값을 반환합니다

4) 아직 없으면 스케줄링 전개
   - Scheduler.drain()을 반복 호출하면서 실행을 진행시킵니다
   - 실행이 진행되면 결과가 로컬 store에 생길 수 있습니다

5) 그래도 없으면 원격 fetch
   - router+rpc가 설정되어 있으면 owner 노드에서 bytes를 가져옵니다
   - bytes를 로컬 cache_store에 put_bytes로 저장해 재물질화합니다
   - aliases[remote_id] = local_id로 기록해 이후 get을 빠르게 만듭니다
```

여기서도 Worker.execute_task를 직접 호출하지 않습니다. get은 Scheduler.drain을 호출해 실행을 전개할 뿐이고, 실행 호출은 Scheduler 내부 try_place에서 발생합니다.

## aliases와 로컬 캐시

원격에서 가져온 객체를 로컬에 저장하면 로컬 store가 새 object_id를 발급합니다. 그래서 원격 object_id를 다시 요청했을 때 로컬 hit을 만들려면 remote id에서 local id로의 매핑이 필요합니다. 그게 aliases입니다.

- remote object_id는 원격 노드 기준 식별자입니다
- local store는 local object_id를 새로 발급합니다
- aliases는 remote id를 local id로 연결해 다음 get을 빠르게 만듭니다

로컬 worker가 없으면 driver_store가 캐시 역할을 수행합니다.

## PlacementGroup 드라이버 API

Session은 PlacementGroup을 만드는 드라이버 API를 제공합니다.

- create_placement_group
  - Bundle 리스트와 strategy로 PlacementGroup을 만듭니다
  - scheduler.register_placement_group로 등록합니다

- remove_placement_group
  - scheduler.unregister_placement_group로 해제합니다

중요한 점은 Session이 배치를 직접 관리하는 것이 아니라, Scheduler의 배치 판단이 가능하도록 입력을 등록하는 창구라는 점입니다.

## 글로벌 세션

GLOBAL_SESSION, init_session, get_session은 드라이버 UX를 단순화하기 위한 편의 계층입니다. 사용자 입장에서는 put, get, submit 같은 함수를 바로 쓸 수 있고, 내부적으로는 전역 Session 객체에 위임됩니다.

## session: 3줄 요약

- 드라이버는 사용자 코드가 돌아가는 중심 프로세스이며, Session은 드라이버의 런타임 핸들입니다
- Session은 Scheduler를 소유하며 submit, drain, get을 통해 실행 전개와 결과 회수를 제공합니다
- get은 로컬 조회, 스케줄링 전개, 원격 fetch, 로컬 캐시를 한 흐름으로 묶습니다