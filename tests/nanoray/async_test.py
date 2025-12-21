import time
from typing import List, Dict

from nanorlhf import nanoray
from nanorlhf.nanoray.core.object_ref import ObjectRef


@nanoray.remote
def heavy_sleep(duration: float, value: int) -> str:
    import time as _time

    _time.sleep(duration)
    return f"task {value} finished after {duration:.1f}s"


@nanoray.remote
class SlowActor:
    def __init__(self, work_delay: float, init_delay: float = 0.0):
        import time as _time

        if init_delay:
            _time.sleep(init_delay)
        self.work_delay = work_delay

    def work(self, value: int) -> str:
        import time as _time

        _time.sleep(self.work_delay)
        return f"actor work({value}) finished after {self.work_delay:.1f}s"


def _unwrap_actor_handle(x):
    # In case ActorRef is wrapped as ObjectRef in store
    while isinstance(x, ObjectRef):
        x = nanoray.get(x)
    return x


def _describe(label: str, refs: List[object]):
    print(f"{label}: {[getattr(r, 'object_id', None) for r in refs]}")


def _expected_parallel_wall_for_tasks(durations: List[float], placements: List[str]) -> float:
    per_node_max: Dict[str, float] = {}
    for d, nid in zip(durations, placements):
        per_node_max[nid] = max(d, per_node_max.get(nid, 0.0))
    return max(per_node_max.values()) if per_node_max else 0.0


def bench_tasks_concurrent(
    label: str,
    node_ids: List[str],
    durations: List[float],
    *,
    max_concurrency: int,
):
    print(f"\n=== [1] {label}: tasks concurrently across {', '.join(node_ids)} ===")

    placements = [node_ids[i % len(node_ids)] for i in range(len(durations))]
    expected = _expected_parallel_wall_for_tasks(durations, placements)

    # submit (enqueue only)
    t0 = time.perf_counter()
    t_submit0 = time.perf_counter()
    refs = [
        heavy_sleep.options(
            pinned_node_id=placements[i],
            max_concurrency=max_concurrency,
        ).remote(durations[i], i, blocking=False)
        for i in range(len(durations))
    ]
    submit_elapsed = time.perf_counter() - t_submit0

    print("Task placement:")
    for i, (d, nid) in enumerate(zip(durations, placements)):
        print(f"  task {i} -> {nid} (sleep={d:.1f}s)")
    _describe("Immediately after submit", refs)
    print(f"Submit overhead: {submit_elapsed*1e3:.2f} ms")

    # drive execution
    t_drain0 = time.perf_counter()
    produced = nanoray.drain()
    drain_elapsed = time.perf_counter() - t_drain0
    # normalize refs list: include any None + produced
    refs2 = [r for r in refs if r is not None] + produced

    # get results
    t_get0 = time.perf_counter()
    values = [nanoray.get(r) for r in refs2]
    get_elapsed = time.perf_counter() - t_get0

    total_elapsed = time.perf_counter() - t0

    print(f"Drain time: {drain_elapsed:.2f}s, Get time: {get_elapsed:.2f}s")
    print("Results:")
    for v in values:
        print(" ", v)
    print(f"Wall time: {total_elapsed:.2f}s (expected ~{expected:.1f}s)")
    return total_elapsed


def bench_actor_creation_concurrent(
    label: str,
    node_ids: List[str],
    *,
    num_actors: int,
    init_delay: float,
    work_delay: float,
    actor_max_concurrency: int,
):
    print(f"\n=== [2] {label}: create {num_actors} actors concurrently across {', '.join(node_ids)} ===")

    placements = [node_ids[i % len(node_ids)] for i in range(num_actors)]

    t0 = time.perf_counter()
    t_submit0 = time.perf_counter()

    actor_refs = []
    for i in range(num_actors):
        nid = placements[i]
        r = SlowActor.options(pinned_node_id=nid, max_concurrency=actor_max_concurrency).remote(
            work_delay, init_delay=init_delay, blocking=False
        )
        actor_refs.append(r)

    submit_elapsed = time.perf_counter() - t_submit0
    print("Actor placement:")
    for i, nid in enumerate(placements):
        print(f"  actor {i} -> {nid} (init_delay={init_delay:.1f}s)")
    _describe("Immediately after actor create submit", actor_refs)
    print(f"Submit overhead: {submit_elapsed*1e3:.2f} ms")

    t_drain0 = time.perf_counter()
    produced = nanoray.drain()
    drain_elapsed = time.perf_counter() - t_drain0

    # actor_refs should not be None in your model, but normalize anyway
    actor_refs2 = [r for r in actor_refs if r is not None] + produced

    t_get0 = time.perf_counter()
    actors = [_unwrap_actor_handle(nanoray.get(r)) for r in actor_refs2[:num_actors]]
    get_elapsed = time.perf_counter() - t_get0

    total_elapsed = time.perf_counter() - t0

    print(f"Drain time: {drain_elapsed:.2f}s, Get time: {get_elapsed:.2f}s")
    print(f"Actor create wall time: {total_elapsed:.2f}s (ideal-ish ~{init_delay:.1f}s + spawn/boot)")
    return actors, total_elapsed


def bench_actor_calls_concurrent(
    label: str,
    actors: List,
    *,
    calls_per_actor: int,
):
    print(f"\n=== [3] {label}: actor calls concurrently (actors={len(actors)}, calls/actor={calls_per_actor}) ===")

    t0 = time.perf_counter()
    t_submit0 = time.perf_counter()

    refs = []
    idx = 0
    for a_i, a in enumerate(actors):
        for j in range(calls_per_actor):
            refs.append(a.work.remote(idx, blocking=False))
            idx += 1

    submit_elapsed = time.perf_counter() - t_submit0
    _describe("Immediately after actor call submit", refs)
    print(f"Submit overhead: {submit_elapsed*1e3:.2f} ms")

    t_drain0 = time.perf_counter()
    produced = nanoray.drain()
    drain_elapsed = time.perf_counter() - t_drain0

    refs2 = [r for r in refs if r is not None] + produced

    t_get0 = time.perf_counter()
    values = [nanoray.get(r) for r in refs2[: len(refs)]]
    get_elapsed = time.perf_counter() - t_get0

    total_elapsed = time.perf_counter() - t0

    print(f"Drain time: {drain_elapsed:.2f}s, Get time: {get_elapsed:.2f}s")
    print("Results (first 10):")
    for v in values[:10]:
        print(" ", v)
    if len(values) > 10:
        print(f"  ... ({len(values)-10} more)")
    print(f"Actor calls wall time: {total_elapsed:.2f}s")
    return total_elapsed


def main():
    durations = [2.0, 3.0, 2.0, 3.0]
    cpus = len(durations)

    config = {
        "rpc-node-1": nanoray.NodeConfig(cpus=cpus, rpc=True, port=8092),
        "rpc-node-2": nanoray.NodeConfig(cpus=cpus, rpc=True, port=8093),
        "local-node-1": nanoray.NodeConfig(cpus=cpus, rpc=False),
        "local-node-2": nanoray.NodeConfig(cpus=cpus, rpc=False),
    }
    nanoray.init(config, default_node_id="local-node-1")

    rpc_nodes = ["rpc-node-1", "rpc-node-2"]
    local_nodes = ["local-node-1", "local-node-2"]

    bench_tasks_concurrent("RPC", rpc_nodes, durations, max_concurrency=len(durations))
    bench_tasks_concurrent("Local", local_nodes, durations, max_concurrency=len(durations))

    num_actors = 4
    init_delay = 0.8
    work_delay = 0.6
    actor_max_concurrency = 8

    rpc_actors, _ = bench_actor_creation_concurrent(
        "RPC",
        rpc_nodes,
        num_actors=num_actors,
        init_delay=init_delay,
        work_delay=work_delay,
        actor_max_concurrency=actor_max_concurrency,
    )
    local_actors, _ = bench_actor_creation_concurrent(
        "Local",
        local_nodes,
        num_actors=num_actors,
        init_delay=init_delay,
        work_delay=work_delay,
        actor_max_concurrency=actor_max_concurrency,
    )

    bench_actor_calls_concurrent("RPC", rpc_actors, calls_per_actor=4)
    bench_actor_calls_concurrent("Local", local_actors, calls_per_actor=4)
    nanoray.shutdown()


if __name__ == "__main__":
    main()
