# PlacementGroup

When running distributed jobs with Ray, an issue more important than simply how many CPU/GPU you use often appears. Even with the same resources, performance and stability can vary greatly depending on which nodes the tasks are placed on.

A PlacementGroup is a unit that communicates placement intent to the scheduler: these tasks are related, so please place them in this way. You create a group from multiple resource bundles (Bundles) and set a direction such as packing them onto one node (PACK) or spreading them across multiple nodes (SPREAD).

## 1. Why do we need PlacementGroup?

In local execution, you rarely need to care where something runs. But in distributed environments, the following often turn into real problems.

- Tasks get placed across different nodes every time, making performance fluctuate.
- Even if tasks are strongly related, the scheduler may place them on any node as long as resources fit, because it does not know their relationship.
- Inter-node data movement (especially large tensors/caches) increases and the network becomes a bottleneck.

PlacementGroup provides a way to deliver a one-shot hint about where tasks should live.

### When is PlacementGroup especially useful?
- When tasks share large state (model weights, caches, etc.)
- When you want planned placement for pipelining/sharding-like roles
- When you want to express intent at a higher level instead of pinning node ids directly

## 2. What is a Bundle?

A Bundle means one slot inside a PlacementGroup. In other words, it is a request unit that says please reserve one execution slot with about this much resource.

```python
from dataclasses import dataclass, field
from typing import Dict

@dataclass(frozen=True)
class Bundle:
    cpus: float = 0.0
    gpus: float = 0.0
    resources: Dict[str, float] = field(default_factory=dict)
```

- cpus and gpus are the basic resource requirements.
- resources is an extension point for custom resources like ram.
- A Bundle is immutable because if a fixed requirement spec changes during execution, scheduling decisions become unstable.

### What does it mean that there can be multiple Bundles?
A PlacementGroup is not a single large request; it can bundle together a set of slots with different specs. For example, you can group a GPU-required slot and a CPU-only slot together, then ask the scheduler to place the whole set according to PACK or SPREAD.

## 3. What is PlacementStrategy?

PlacementStrategy is the rule that defines how to place the bundles.

```python
class PlacementStrategy:
    PACK = "PACK"
    SPREAD = "SPREAD"
```

### What does PACK mean?
It prefers to pack the bundles onto a single node if possible.

- Expected benefit: fewer data transfers on the same node can make things faster.
- Caveat: if one node does not have enough resources, placing the whole group can become difficult.

### What does SPREAD mean?
It prefers to spread the bundles across different nodes if possible.

- Expected benefit: better parallelism and easier reduction of single-node bottlenecks.
- Caveat: if inter-node movement increases, it can become slower.

### Which strategy should you choose?
#### When PACK is a good choice
- When sharing large state makes being close strongly beneficial
- When inter-node transfer is expensive or reducing transfers is important
- When one node has enough capacity to fit the whole set

#### When SPREAD is a good choice
- When tasks are independent and spreading improves parallelism
- When packing onto one node frequently creates bottlenecks
- When throughput gains from distribution outweigh data movement costs

## 4. What does the PlacementGroup structure look like?

A PlacementGroup has a list of bundles, a strategy, and an id that identifies the group.

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

- bundles is the list of slots accessed by bundle_index.
- strategy is either PACK or SPREAD.
- pg_id is used to identify the group stably. The scheduler/runtime needs such an identifier to treat the same group as the same group.

## 5. Usage examples

In ML workloads, PlacementGroup is especially intuitive in dataset processing pipelines. You can think of it like this.

- Multiple workers for preprocessing/tokenizing (relatively CPU-heavy)
- Workers that build batches/cache from those results (also CPU-heavy but memory-intensive)
- Optionally, workers that run GPU preprocess (augmentation, embedding, etc.)

How these components are grouped and placed can significantly affect overall throughput.

### Example 1: PACK onto one node to maximize cache/shared-state benefits

If you strongly share caches on the same node, packing onto a single node can be beneficial.

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

- If you create 4 bundles, the scheduler treats them as a set of 4 slots (each cpus=1.0) for these tasks.
- With PACK, it prefers to place those 4 slots onto one node if possible.
- As a result, local sharing benefits such as file cache can become larger.

### Example 2: SPREAD across nodes to secure throughput

If shard-level jobs are independent, spreading across nodes to maximize parallelism can be beneficial.

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

- It prefers to place bundle_index=0 and bundle_index=1 onto different nodes.
- This reduces CPU bottlenecks that happen when everything is concentrated on one node, and can improve overall throughput.

### Example 3: Mix bundles with different CPU/GPU requirements in one group

In practice, not all tasks use the same resources. For example:

- CPU preprocessing worker: uses a lot of CPU, no GPU needed
- GPU encoder worker: needs 1 GPU and some CPU
- CPU aggregator/merge worker: collects results and stores them, CPU-heavy

If you split each role into its own bundle, the scheduler can see more clearly that these tasks form a set and what each role needs.

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
    # Example: A has more GPUs, B is CPU-only
    "node-A": nanoray.NodeConfig(rpc=True, host="127.0.0.1", port=9201, cpus=8.0, gpus=2.0),
    "node-B": nanoray.NodeConfig(rpc=True, host="127.0.0.1", port=9202, cpus=8.0, gpus=0.0),
}
nanoray.init(nodes, default_node_id="node-A")

pg = nanoray.create_placement_group(
    bundles=[
        Bundle(cpus=4.0, gpus=0.0),  # bundle_index=0: CPU preprocessing slot
        Bundle(cpus=2.0, gpus=1.0),  # bundle_index=1: GPU encoder slot
        Bundle(cpus=2.0, gpus=0.0),  # bundle_index=2: CPU merge/aggregation slot
    ],
    strategy=PlacementStrategy.PACK,
)

# 1) CPU preprocessing (bundle_index=0)
r0 = cpu_preprocess.options(placement_group=pg, bundle_index=0, num_cpus=4.0).remote(0, blocking=True)
r1 = cpu_preprocess.options(placement_group=pg, bundle_index=0, num_cpus=4.0).remote(1, blocking=True)
p0, p1 = nanoray.get([r0, r1])

# 2) GPU encoding (bundle_index=1)
e0 = gpu_encode.options(placement_group=pg, bundle_index=1, num_cpus=2.0, num_gpus=1.0).remote(p0, blocking=True)
e1 = gpu_encode.options(placement_group=pg, bundle_index=1, num_cpus=2.0, num_gpus=1.0).remote(p1, blocking=True)
enc0, enc1 = nanoray.get([e0, e1])

# 3) CPU merge (bundle_index=2)
m = cpu_merge.options(placement_group=pg, bundle_index=2, num_cpus=2.0).remote([enc0, enc1], blocking=True)
merged = nanoray.get(m)

print(merged)

nanoray.shutdown()
```

- Because each bundle has different CPU/GPU requirements, the scheduler can consider node resource shapes more accurately.
- With PACK, it prefers to keep CPU preprocessing, GPU encoding, and CPU merge together as a set within one node if possible.
- If you switch to SPREAD, the group can split by bundle_index, and then inter-node data movement can increase.

## PlacementGroup: 3-line summary
- PlacementGroup bundles multiple resource slots (Bundles) into one set and communicates placement direction (PACK/SPREAD) to the scheduler.
- Bundle is an execution slot with CPU/GPU/custom resource requirements, and it is especially useful when you want to group different roles together.
- PACK prefers one node, SPREAD prefers multiple nodes, and the key idea is to express placement intent simply for better performance.