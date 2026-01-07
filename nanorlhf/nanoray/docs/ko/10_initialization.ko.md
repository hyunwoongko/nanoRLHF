# initialzation

## 개요

init은 드라이버에서 nanoray 런타임 한 세트를 구성하는 진입점입니다. 구체적으로 Session, Scheduler, 로컬 Worker, 네트워크 구성(RPC), 라우팅 정보를 한 번에 묶어서 준비합니다. 
사용자는 init을 호출한 뒤 remote 호출을 만들고 submit 및 get만 사용해도 실행 흐름이 이어지게 됩니다.

핵심은 한 가지입니다. 스케줄러가 WorkerLike 인터페이스만 보고도 로컬 실행이든 원격 실행이든 동일하게 다루도록 "실행 경로"를 정렬합니다.

## 드라이버 관점

드라이버는 사용자 코드가 돌아가는 프로세스입니다. 드라이버는 실행을 직접 수행하지 않고, 런타임 구성요소를 세팅한 뒤 Task를 제출하고 값을 회수합니다. init은 이 드라이버가 사용할 런타임 핸들을 만들어 Session으로 돌려드립니다.

```text
driver (user code)
  |
  | init(...) 호출
  v
Session (driver-side runtime handle)
  |
  | submit / drain / get
  v
Scheduler  ->  WorkerLike.execute_task(...)  ->  ObjectRef
```

## init이 만드는 구성요소

init이 준비하는 주요 구성요소는 다음입니다.

- NodeRegistry: node_id -> (address, token) 저장소입니다. RPC 클라이언트가 목적지 주소를 찾는 데 사용합니다.
- RpcClient: node_id 대상으로 HTTP JSON RPC 요청을 보내는 클라이언트입니다.
- Router: ObjectRef가 가리키는 owner_node_id를 기준으로 목적 노드를 결정합니다.
- Worker 또는 RemoteWorkerProxy: Scheduler가 호출할 실행 엔드포인트입니다. 둘 다 WorkerLike처럼 동작합니다.
- Session 및 Scheduler: 드라이버가 Task를 제출하고 큐를 전개하고 ObjectRef를 값으로 바꾸는 중심입니다.

## 노드 구성 모델

NodeConfig는 노드 한 개의 "광고된 자원(capacity)" 및 "통신 방식(RPC)"을 담습니다.

- cpus, gpus, resources: 스케줄러가 배치 판단에 쓰는 용량 정보입니다.
- rpc, host, port, token: 로컬 노드인 경우 RPC 서버를 띄울지, 어느 주소로 띄울지, 인증 토큰을 쓸지에 대한 설정입니다.

이 프로젝트의 단순 규칙에서 로컬 여부는 host가 "127.0.0.1" 또는 "localhost"인지로 판단합니다. 교육용 런타임이므로 주소 기반으로 단순화한 것입니다.

## 로컬 노드 초기화

로컬 노드로 판단되면 init은 해당 노드에 다음을 준비합니다.

1) ObjectStore 생성
2) Worker 생성(해당 ObjectStore를 소유)
3) cfg.rpc가 True이면 RpcServer를 별도 스레드에서 시작
4) RpcServer가 실제로 바인딩한 포트를 확인한 뒤 NodeRegistry에 주소 등록

중요한 점은 "로컬 노드에서도 RPC 경로를 쓸 수 있다"는 설계입니다. cfg.rpc가 True이면 스케줄러는 그 로컬 노드를 직접 Worker로 실행하지 않고, RemoteWorkerProxy를 사용해 RPC로 호출합니다. 이렇게 하면 로컬 실행과 원격 실행이 동일한 프로토콜로 통일되며, 교육 측면에서 실행 경로가 더 일관됩니다.

```text
local node (cfg.rpc = False)
  Scheduler -> Worker.execute_task(...)  (direct call)

local node (cfg.rpc = True)
  Scheduler -> RemoteWorkerProxy.execute_task(...)
           -> RpcClient -> RpcServer -> Worker.rpc_execute_task(...)
```

## 원격 노드 초기화

원격 노드는 로컬 Worker를 만들지 않습니다. 스케줄러 관점에서 그 노드는 RemoteWorkerProxy 하나로 표현됩니다.

- RemoteWorkerProxy.execute_task(Task)
  - RpcClient.execute_task(node_id, task) 호출
  - 원격 RpcServer가 받아서 Worker.rpc_execute_task를 실행
  - ObjectRef를 응답으로 돌려줍니다

원격 노드 주소는 NodeRegistry에 있어야 합니다. 이 코드에서는 로컬 노드만 직접 서버를 띄우고 registry.register를 수행합니다. 원격 노드 주소가 필요하다면 외부에서 NodeRegistry에 등록되는 흐름이 추가되어야 합니다. 현재 init 구현은 교육용 단순화를 위해 "host가 로컬인지" 기준으로만 서버를 기동하고 등록합니다.

## 스케줄러 입력(nodes) 구성

init은 Scheduler에 넘길 nodes 맵을 만듭니다.

- key: node_id
- value: (WorkerLike, capacity_dict)

capacity_dict는 스케줄러가 배치 판단에 쓰는 데이터입니다.

```text
capacity_dict 예시
  cpus: 4.0
  gpus: 1.0
  resources: {"ram_gb": 64.0}
```

WorkerLike 자리에 들어가는 객체는 두 가지 중 하나입니다.

- 로컬 direct 실행이면 Worker
- RPC 경로 실행이면 RemoteWorkerProxy

이 덕분에 Scheduler는 "어떻게 실행되는지"를 몰라도 됩니다. Scheduler는 후보 노드를 고르고 난 뒤 worker_like.execute_task(task)만 호출하면 됩니다.

## Session 연결

init은 init_session을 호출해서 글로벌 세션(Session)을 만들고, 다음을 주입합니다.

- local_workers: 로컬 Worker 목록입니다. Session.get의 로컬 조회, put의 기본 저장소 선택에 쓰입니다.
- default_node_id: put 및 로컬 캐시 기본 노드를 정할 때 사용합니다.
- router, rpc: 원격 get 경로에서 사용합니다. ObjectRef의 owner_node_id로 대상 노드를 결정하고, RPC로 bytes를 가져온 뒤 로컬 캐시에 저장합니다.

즉, init은 드라이버가 submit, drain, get을 사용할 수 있도록 "스케줄링" 및 "원격 접근" 훅을 한 번에 연결합니다.

## 기본 단일 노드 모드

nodes를 주지 않으면 init은 다음을 수행합니다.

- node_id로 platform.node() 값을 사용합니다
- NodeConfig(rpc=True)로 로컬 한 개 노드를 만듭니다
- port가 None이면 0을 써서 OS가 임의 포트를 선택하도록 합니다
- 바인딩된 실제 포트를 읽어 registry에 등록합니다

이 모드는 ray.init()의 "바로 시작" UX를 흉내 냅니다. 사용자는 별 설정 없이도 단일 노드에서 동일한 실행 모델을 체험할 수 있습니다.

## shutdown 개요

shutdown은 init이 만든 전역 리소스를 정리합니다.

- 세션 전역 변수 해제
- 등록된 placement group이 있으면 해제 시도
- 로컬 Worker가 있다면 shutdown 호출 시도
- 기동한 RpcServer를 stop 호출해서 종료

이 정리는 best-effort로 구현되어 있습니다. 교육용 런타임에서 중요한 것은 "구성"과 "흐름"이므로, 실패해도 최대한 계속 정리하도록 작성되어 있습니다.

## initialization: 3줄 요약

- init은 드라이버가 사용할 런타임 구성(Session, Scheduler, 실행 엔드포인트, 네트워크 훅)을 한 번에 준비합니다
- 스케줄러는 WorkerLike.execute_task만 호출하며, 로컬 direct 실행과 RPC 실행이 동일한 형태로 추상화됩니다
- cfg.rpc가 True이면 로컬 노드도 RemoteWorkerProxy를 통해 RPC 경로로 실행되어 실행 모델이 더 일관됩니다
