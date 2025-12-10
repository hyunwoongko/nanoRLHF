import time
from typing import List

from nanorlhf import nanoray
from nanorlhf.nanoray.core.object_ref import ObjectRef


@nanoray.remote()
def heavy_sleep(duration: float, value: int) -> str:
    """Simulate a heavy computation by sleeping before returning."""
    import time as _time

    _time.sleep(duration)
    message = f"task {value} finished after {duration:.1f}s"
    return message


@nanoray.actor
class SlowActor:
    def __init__(self, work_delay: float, init_delay: float = 0.0):
        import time as _time

        if init_delay:
            _time.sleep(init_delay)
        self.work_delay = work_delay

    def work(self, value: int) -> str:
        import time as _time

        _time.sleep(self.work_delay)
        return f"actor {value} finished after {self.work_delay:.1f}s"


def describe(label: str, refs: List[object]):
    print(f"{label}: {[getattr(r, 'object_id', None) for r in refs]}")


def run_task_case(label: str, node_ids: List[str], durations: List[float]):
    print(f"\n=== {label} (tasks across {', '.join(node_ids)}) ===")
    start_total = time.perf_counter()
    start_submit = time.perf_counter()
    refs = []
    placements = []
    for i, d in enumerate(durations):
        node_id = node_ids[i % len(node_ids)]
        placements.append((i, d, node_id))
        refs.append(heavy_sleep.options(pinned_node_id=node_id).remote(d, i, blocking=False))
    submit_elapsed = time.perf_counter() - start_submit

    print("Task placement:")
    for idx, d, node_id in placements:
        print(f"   task {idx} -> {node_id} (sleep={d:.1f}s)")

    describe("Immediately after submit", refs)
    print(f"Submission overhead: {submit_elapsed*1e3:.2f} ms")

    refs = [r for r in refs if r is not None] + nanoray.drain()
    values = [nanoray.get(r) for r in refs]
    total_elapsed = time.perf_counter() - start_total

    print("Results:")
    for v in values:
        print("  ", v)

    per_node_max = {}
    for _, d, node_id in placements:
        per_node_max[node_id] = max(d, per_node_max.get(node_id, 0.0))
    expected = max(per_node_max.values()) if per_node_max else 0.0
    print(f"Wall time: {total_elapsed:.2f}s (expected ~{expected:.1f}s)")


def run_actor_creation_case(label: str, node_id: str, init_delay: float, work_delay: float):
    print(f"\n=== {label} (actor create on {node_id}) ===")
    start_total = time.perf_counter()
    start_submit = time.perf_counter()
    actor_ref = SlowActor.options(pinned_node_id=node_id).remote(work_delay, init_delay=init_delay)
    submit_elapsed = time.perf_counter() - start_submit
    describe("Immediately after actor create submit", [actor_ref])
    print(f"Actor create submit overhead: {submit_elapsed*1e3:.2f} ms")

    start_get = time.perf_counter()
    actor = nanoray.get(actor_ref)
    while isinstance(actor, ObjectRef):  # unwrap nested refs if stored indirectly
        actor = nanoray.get(actor)
    create_elapsed = time.perf_counter() - start_get
    total_elapsed = time.perf_counter() - start_total

    print(f"Actor create get time: {create_elapsed:.2f}s (expected ~{init_delay:.1f}s)")
    print(f"Actor create wall time: {total_elapsed:.2f}s")
    return actor


def run_actor_method_case(label: str, actor, count: int, expected_delay: float):
    print(f"\n=== {label} (actor methods) ===")
    start_total = time.perf_counter()
    start_calls = time.perf_counter()
    refs = [actor.work.remote(i) for i in range(count)]
    call_submit = time.perf_counter() - start_calls
    describe("Immediately after actor method submit", refs)
    print(f"Actor call submission overhead: {call_submit*1e3:.2f} ms")

    refs = [r for r in refs if r is not None] + nanoray.drain()
    values = [nanoray.get(r) for r in refs]
    total_elapsed = time.perf_counter() - start_total

    print("Results:")
    for v in values:
        print("  ", v)

    if values:
        expected = expected_delay
    else:
        expected = 0.0
    print(f"Wall time: {total_elapsed:.2f}s (expected ~{expected:.1f}s)")


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

    run_task_case("RPC tasks", rpc_nodes, durations)
    run_task_case("Local tasks", local_nodes, durations)

    actors = []
    for node_id in rpc_nodes:
        actors.append(
            (f"RPC actor ({node_id})", run_actor_creation_case("RPC actor", node_id, init_delay=0.8, work_delay=0.6))
        )
    for node_id in local_nodes:
        actors.append(
            (
                f"Local actor ({node_id})",
                run_actor_creation_case("Local actor", node_id, init_delay=0.8, work_delay=0.6),
            )
        )

    for label, actor in actors:
        run_actor_method_case(label, actor, count=2, expected_delay=0.6)

    nanoray.shutdown()


if __name__ == "__main__":
    main()
