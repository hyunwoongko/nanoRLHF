import os
import shutil
import tempfile
import time
from typing import List

import numpy as np

from nanorlhf import nanoray
from nanorlhf.nanoray.core.object_ref import ObjectRef


def file_barrier(name: str, rank: int, world_size: int, barrier_dir: str, timeout: float = 30.0):
    """Very small file-based barrier usable from multiple worker processes.

    Args:
        name: Logical barrier name (e.g., "init", "forward-0").
        rank: Rank of the participant.
        world_size: Total number of participants expected.
        barrier_dir: Directory where barrier tokens are written.
        timeout: Optional timeout in seconds.
    """

    os.makedirs(barrier_dir, exist_ok=True)
    token_path = os.path.join(barrier_dir, f"{name}-rank{rank}")
    with open(token_path, "w", encoding="utf-8") as f:
        f.write(f"rank={rank}\n")

    start = time.time()
    while True:
        files = [fn for fn in os.listdir(barrier_dir) if fn.startswith(name)]
        if len(files) >= world_size:
            break
        if time.time() - start > timeout:
            raise TimeoutError(f"Barrier '{name}' timed out waiting for {world_size} ranks")
        time.sleep(0.05)


@nanoray.remote
class TPShard:
    def __init__(self, rank: int, world_size: int, barrier_dir: str, hidden_size: int = 8, init_delay: float = 0.2):
        import numpy as _np
        import time as _time

        self.rank = rank
        self.world_size = world_size
        self.barrier_dir = barrier_dir
        self.hidden_size = hidden_size
        _time.sleep(init_delay)
        self.weights = _np.full((hidden_size, hidden_size), rank + 1, dtype=_np.float32)
        file_barrier("init", rank, world_size, barrier_dir)

    def forward(self, inputs: List[float], step: int, work_delay: float = 0.3) -> List[float]:
        import numpy as _np
        import time as _time

        file_barrier(f"forward-{step}", self.rank, self.world_size, self.barrier_dir)
        _time.sleep(work_delay)
        x = _np.asarray(inputs, dtype=_np.float32)
        partial = x + self.weights.sum(axis=0)
        return partial.tolist()


def describe(label: str, refs: List[object]):
    print(f"{label}: {[getattr(r, 'object_id', None) for r in refs]}")


def unwrap(ref_or_val):
    val = ref_or_val
    while isinstance(val, ObjectRef):
        val = nanoray.get(val)
    return val


def main():
    config = {
        "rpc-node-1": nanoray.NodeConfig(cpus=2.0, rpc=True, port=8100),
        "rpc-node-2": nanoray.NodeConfig(cpus=2.0, rpc=True, port=8101),
        "rpc-node-3": nanoray.NodeConfig(cpus=2.0, rpc=True, port=8102),
        "rpc-node-4": nanoray.NodeConfig(cpus=2.0, rpc=True, port=8103),
        "rpc-node-5": nanoray.NodeConfig(cpus=2.0, rpc=True, port=8104),
        "rpc-node-6": nanoray.NodeConfig(cpus=2.0, rpc=True, port=8105),
        "rpc-node-7": nanoray.NodeConfig(cpus=2.0, rpc=True, port=8106),
        "rpc-node-8": nanoray.NodeConfig(cpus=2.0, rpc=True, port=8107),
    }

    nanoray.init(config, default_node_id="rpc-node-1")

    barrier_dir = os.path.join(tempfile.gettempdir(), f"nanoray_tp_{os.getpid()}")
    if os.path.exists(barrier_dir):
        shutil.rmtree(barrier_dir)

    world_size = 8
    node_ids = list(config.keys())

    print("\n=== Create tensor-parallel actors (8 nodes) ===")
    actor_refs = []
    start_submit = time.perf_counter()
    for rank, node_id in enumerate(node_ids):
        actor_refs.append(
            TPShard.options(pinned_node_id=node_id).remote(
                rank=rank,
                world_size=world_size,
                barrier_dir=barrier_dir,
                hidden_size=8,
                init_delay=0.4,
            )
        )
    submit_elapsed = time.perf_counter() - start_submit
    describe("Actor create refs", actor_refs)
    print(f"Actor create submit overhead: {submit_elapsed*1e3:.2f} ms")

    start_get = time.perf_counter()
    actors = [unwrap(ref) for ref in actor_refs]
    create_elapsed = time.perf_counter() - start_get
    print(f"Actor create wait time: {create_elapsed:.2f}s (includes simulated init + barrier)")

    # Prepare a fake batch and run a few synchronized forward passes
    batch = np.arange(8, dtype=np.float32) * 0.1
    steps = 3

    for step in range(steps):
        print(f"\n=== Forward step {step} ===")
        start_calls = time.perf_counter()
        refs = [actor.forward.remote(batch.tolist(), step) for actor in actors]
        call_submit = time.perf_counter() - start_calls
        describe("Forward refs", refs)
        print(f"Forward submit overhead: {call_submit*1e3:.2f} ms")

        refs = [r for r in refs if r is not None] + nanoray.drain()
        partials = [nanoray.get(r) for r in refs]
        combined = np.sum([np.array(p, dtype=np.float32) for p in partials], axis=0)
        print("Combined output (sum of partials):", combined.tolist())

    nanoray.shutdown()
    print("\n=== Shutdown ===")


if __name__ == "__main__":
    main()
