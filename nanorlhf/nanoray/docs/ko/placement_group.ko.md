# PlacementGroup

Ray로 분산 작업을 돌리다 보면 단순히 CPU/GPU를 몇 개 쓰느냐보다 더 중요한 문제가 생깁니다. 같은 자원을 쓰더라도 작업들이 어떤 노드에 배치되느냐에 따라 속도와 안정성이 크게 달라질 수 있기 때문입니다.

PlacementGroup은 이 작업들은 서로 연관이 있으니 이런 방식으로 배치해 주세요라는 "배치 의도"를 스케줄러에 전달하는 단위입니다. 여러 개의 리소스 묶음(Bundle)을 한 그룹으로 만들고, 그 묶음을 한 노드에 모으거나(PACK) 여러 노드로 퍼뜨리는(SPREAD) 식으로 방향을 잡습니다.

## 1. PlacementGroup은 왜 필요한가요?

로컬 실행에서는 실행 위치를 신경 쓸 일이 거의 없습니다. 하지만 분산 환경에서는 다음이 자주 문제로 이어집니다.

- 작업들이 매번 다른 노드에 흩어져 배치되면서 성능이 들쑥날쑥해집니다.
- 서로 강하게 연관된 작업인데도, 스케줄러는 그 관계를 모른 채 "자원만 맞으면 아무 노드나" 배치할 수 있습니다.
- 노드 간 데이터 이동(특히 큰 텐서/캐시)이 늘어나면 네트워크가 병목이 됩니다.

PlacementGroup은 이런 상황에서 작업들이 어디에 놓이면 좋은지에 대한 힌트를 한 번에 전달합니다.

### PlacementGroup이 특히 유용한 경우는 무엇인가요?
- 같은 큰 상태(모델 가중치, 캐시 등)를 공유하는 작업들이 있을 때
- 파이프라인/샤딩처럼 역할이 나뉜 작업들을 계획된 형태로 배치하고 싶을 때
- 노드 id를 직접 고정하는 대신, 더 높은 수준에서 의도를 표현하고 싶을 때

## 2. Bundle은 무엇인가요?

Bundle은 PlacementGroup 안에서 "한 자리(slot)"를 의미합니다. 즉, 이 정도 자원을 갖춘 실행 자리를 하나 확보해 달라는 요청 단위입니다.

```python
from dataclasses import dataclass, field
from typing import Dict

@dataclass(frozen=True)
class Bundle:
    cpus: float = 0.0
    gpus: float = 0.0
    resources: Dict[str, float] = field(default_factory=dict)
```

- cpus, gpus는 기본 자원 요구량입니다.
- resources는 ram 같은 커스텀 자원을 표현하기 위한 확장 포인트입니다.
- Bundle이 불변인 이유는, 한 번 정한 "요구 스펙"이 실행 중에 바뀌면 스케줄링 판단이 흔들리기 때문입니다.

### Bundle이 여러 개일 수 있다는 건 무엇을 의미하나요?
PlacementGroup은 한 개의 큰 요청이 아니라, 서로 다른 스펙의 자리들을 한 세트로 묶을 수 있습니다. 예를 들어 GPU가 필요한 자리와 CPU만 필요한 자리를 같이 묶고, 그 전체를 PACK이나 SPREAD로 배치하도록 요청할 수 있습니다.

## 3. PlacementStrategy는 무엇인가요?

PlacementStrategy는 번들들을 어떤 방향으로 배치할지에 대한 규칙입니다.

```python
class PlacementStrategy:
    PACK = "PACK"
    SPREAD = "SPREAD"
```

### PACK은 어떤 의미인가요?
가능하면 한 노드에 모으는 방향을 선호합니다.

- 기대 효과: 같은 노드에 있으면 데이터 이동이 줄어들어 빠를 수 있습니다.
- 주의점: 한 노드에 자원이 충분하지 않으면 전체 배치가 어려워질 수 있습니다.

### SPREAD는 어떤 의미인가요?
가능하면 서로 다른 노드로 퍼뜨리는 방향을 선호합니다.

- 기대 효과: 병렬성이 좋아지고 특정 노드 병목을 줄이기 쉽습니다.
- 주의점: 노드 간 데이터 이동이 늘어나면 오히려 느려질 수 있습니다.

### 어떤 전략을 선택해야 할까요?
#### PACK을 선택하면 좋은 경우
- 큰 상태를 공유해서 "가까이" 있어야 이득이 큰 경우
- 노드 간 전송이 부담스럽거나, 전송 횟수를 줄이는 것이 중요한 경우
- 한 노드에 충분한 자원이 있어 한 번에 모을 수 있는 경우

#### SPREAD를 선택하면 좋은 경우
- 작업들이 독립적이어서 흩어질수록 병렬성이 좋아지는 경우
- 한 노드에 몰리면 병목이 자주 생기는 경우
- 데이터 이동 비용보다 분산으로 얻는 처리량 증가가 더 큰 경우

## 4. PlacementGroup 구조는 어떻게 생겼나요?

PlacementGroup은 번들 목록과 전략, 그리고 그룹을 식별하는 id를 가집니다.

```python
from dataclasses import dataclass
from typing import List

from nanorlhf.nanoray.utils import new_placement_group_id

@dataclass(frozen=True)
class PlacementGroup:
    bundles: List[Bundle]
    strategy: str = PlacementStrategy.PACK
    pg_id: str = new_placement_group_id()

    def bundle(self, index: int) -> Bundle:
        return self.bundles[index]

    def __len__(self):
        return len(self.bundles)
```

- bundles는 bundle_index로 접근되는 자리들의 목록입니다.
- strategy는 PACK/SPREAD 중 하나입니다.
- pg_id는 이 그룹을 안정적으로 식별하기 위한 값입니다. 스케줄러/런타임이 같은 그룹을 같은 그룹으로 취급하려면 이런 식별자가 필요합니다.

## 5. 사용 예시

머신러닝 워크로드에서는 데이터셋 처리 파이프라인에서 PlacementGroup이 특히 직관적으로 쓰입니다. 예를 들어 다음처럼 생각할 수 있습니다.

- 데이터 전처리/토크나이징(상대적으로 CPU 중심 작업) 워커 여러 개
- 그 결과를 받아서 배치 구성/캐싱을 하는 워커(역시 CPU 중심이지만 메모리 사용이 큼)
- 경우에 따라 GPU 프리프로세스(augmentation, embedding 등)를 하는 워커

이 구성요소들이 서로 어떤 방식으로 묶여 배치되느냐에 따라 전체 처리량이 크게 달라질 수 있습니다.

### 예시 1: PACK으로 한 노드에 묶어서 캐시/공유 상태를 최대한 활용하기

같은 노드에서 캐시를 강하게 공유한다면, 한 노드로 모으는 것이 유리할 수 있습니다.

```python
from nanorlhf import nanoray
from nanorlhf.nanoray.core.placement import Bundle, PlacementStrategy

@nanoray.remote
def function(shard_id: int):
    return f"output-{shard_id}"

nodes = {
    "node-A": nanoray.NodeConfig(rpc=True, host="127.0.0.1", port=9201, cpus=4.0, gpus=1.0),
    "node-B": nanoray.NodeConfig(rpc=True, host="127.0.0.1", port=9202, cpus=4.0, gpus=1.0),
}
nanoray.init(nodes, default_node_id="node-A")

bundles = [
    Bundle(cpus=1.0, gpus=0.0),
    Bundle(cpus=1.0, gpus=0.0),
    Bundle(cpus=1.0, gpus=0.0),
    Bundle(cpus=1.0, gpus=0.0),
]

pg = nanoray.create_placement_group(
    bundles=bundles,
    strategy=PlacementStrategy.PACK,
)

refs = []
for i in range(len(bundles)):
    ref = function.options(placement_group=pg, bundle_index=i, num_cpus=1.0).remote(i, blocking=False)
    refs.append(ref)

out = nanoray.get(refs)
print(out)

nanoray.shutdown()
```

- bundle을 4개 만들면, 스케줄러는 이 작업들을 위한 4개의 자리(각각 cpus=1.0)를 한 세트로 봅니다.
- PACK을 쓰면, 가능하면 그 4자리를 한 노드에 모으는 방향으로 배치합니다.
- 결과적으로 파일 캐시 같은 로컬 공유 이점이 커질 수 있습니다.

### 예시 2: SPREAD로 여러 노드에 퍼뜨려서 처리량을 확보하기

샤드 단위 작업들이 서로 독립적이라면, 노드 전체로 퍼뜨려 병렬성을 극대화하는 것이 유리할 수 있습니다.

```python
from nanorlhf import nanoray
from nanorlhf.nanoray.core.placement import Bundle, PlacementStrategy

@nanoray.remote
def function(shard_id: int):
    return f"output-{shard_id}"

nodes = {
    "node-A": nanoray.NodeConfig(rpc=True, host="127.0.0.1", port=9201, cpus=2.0, gpus=0.0),
    "node-B": nanoray.NodeConfig(rpc=True, host="127.0.0.1", port=9202, cpus=2.0, gpus=0.0),
}
nanoray.init(nodes, default_node_id="node-A")

pg = nanoray.create_placement_group(
    bundles=[
        Bundle(cpus=1.0, gpus=0.0),
        Bundle(cpus=1.0, gpus=0.0),
    ],
    strategy=PlacementStrategy.SPREAD,
)

ref0 = function.options(placement_group=pg, bundle_index=0, num_cpus=1.0).remote(0, blocking=True)
ref1 = function.options(placement_group=pg, bundle_index=1, num_cpus=1.0).remote(1, blocking=True)

v0, v1 = nanoray.get([ref0, ref1])
print(v0, v1)

nanoray.shutdown()
```

- bundle_index=0과 bundle_index=1이 서로 다른 노드에 배치되도록 스케줄러가 퍼뜨리는 쪽을 선호합니다.
- 같은 노드에 몰려서 생기는 CPU 병목을 줄이고, 전체 처리량을 올리는 데 유리합니다.

### 예시 3: CPU/GPU 요구량이 다른 번들을 섞어서 한 그룹으로 다루기

실전에서는 모든 작업이 같은 자원을 쓰지 않습니다. 예를 들어 다음처럼 구성할 수 있습니다.

- CPU 전처리 워커: CPU를 많이 쓰고 GPU는 필요 없음
- GPU 인코더 워커: GPU 1개와 적당한 CPU가 필요
- CPU 집계/머지 워커: CPU 위주로 결과를 모으고 저장

이때 각 역할을 번들로 분리해 두면, 스케줄러에게 이 작업들은 한 세트이고, 각 작업이 어느 정도 자원을 써야 하는지가 더 명확하게 전달됩니다.

```python
from nanorlhf import nanoray
from nanorlhf.nanoray.core.placement import Bundle, PlacementStrategy

@nanoray.remote
def cpu_preprocess(shard_id: int):
    return f"preprocessed-{shard_id}"

@nanoray.remote
def gpu_encode(x: str):
    return f"encoded({x})"

@nanoray.remote
def cpu_merge(xs):
    return " | ".join(xs)

nodes = {
    # 예: A는 GPU가 더 많은 노드, B는 CPU만 있는 노드
    "node-A": nanoray.NodeConfig(rpc=True, host="127.0.0.1", port=9201, cpus=8.0, gpus=2.0),
    "node-B": nanoray.NodeConfig(rpc=True, host="127.0.0.1", port=9202, cpus=8.0, gpus=0.0),
}
nanoray.init(nodes, default_node_id="node-A")

pg = nanoray.create_placement_group(
    bundles=[
        Bundle(cpus=4.0, gpus=0.0),  # bundle_index=0: CPU 전처리 자리
        Bundle(cpus=2.0, gpus=1.0),  # bundle_index=1: GPU 인코더 자리
        Bundle(cpus=2.0, gpus=0.0),  # bundle_index=2: CPU 머지/집계 자리
    ],
    strategy=PlacementStrategy.PACK,
)

# 1) CPU 전처리 (bundle_index=0)
r0 = cpu_preprocess.options(placement_group=pg, bundle_index=0, num_cpus=4.0).remote(0, blocking=True)
r1 = cpu_preprocess.options(placement_group=pg, bundle_index=0, num_cpus=4.0).remote(1, blocking=True)
p0, p1 = nanoray.get([r0, r1])

# 2) GPU 인코딩 (bundle_index=1)
e0 = gpu_encode.options(placement_group=pg, bundle_index=1, num_cpus=2.0, num_gpus=1.0).remote(p0, blocking=True)
e1 = gpu_encode.options(placement_group=pg, bundle_index=1, num_cpus=2.0, num_gpus=1.0).remote(p1, blocking=True)
enc0, enc1 = nanoray.get([e0, e1])

# 3) CPU 머지 (bundle_index=2)
m = cpu_merge.options(placement_group=pg, bundle_index=2, num_cpus=2.0).remote([enc0, enc1], blocking=True)
merged = nanoray.get(m)

print(merged)

nanoray.shutdown()
```

- 번들마다 CPU/GPU 요구량이 다르기 때문에, 스케줄러가 노드의 자원 형태를 더 정확히 고려할 수 있습니다.
- PACK이면 가능하면 한 노드 안에서 CPU 전처리, GPU 인코딩, CPU 머지가 "한 세트"로 같이 움직이도록 유도합니다.
- SPREAD로 바꾸면 bundle_index 단위로 노드가 갈라질 수 있고, 그때는 노드 간 데이터 이동이 늘어날 수도 있습니다.

## PlacementGroup: 3줄 요약
- PlacementGroup은 여러 리소스 자리(Bundle)를 한 세트로 묶고, 스케줄러에게 배치 방향(PACK/SPREAD)을 전달하는 단위입니다.
- Bundle은 CPU/GPU/커스텀 자원 요구량을 가진 "실행 슬롯"이며, 여러 개를 묶어 서로 다른 역할의 자리를 함께 요청할 수 있습니다.
- PACK은 한 노드 선호, SPREAD는 여러 노드 선호이며, 핵심은 성능이 잘 나오도록 "배치 의도"를 간단히 표현하는 것입니다.