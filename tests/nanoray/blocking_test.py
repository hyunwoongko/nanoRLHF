import time

import pytest

import nanorlhf.nanoray as nanoray
from nanorlhf.nanoray.api.initialization import NodeConfig


@nanoray.remote
def sleep_and_double(x: int, delay: float = 0.05):
    time.sleep(delay)
    return x * 2


@nanoray.remote
class Accumulator:
    def __init__(self):
        self.total = 0

    def add(self, value: int, delay: float = 0.05):
        time.sleep(delay)
        self.total += value
        return self.total


@pytest.fixture(autouse=True)
def runtime():
    nanoray.shutdown()
    nanoray.init(
        nodes={
            "rpc-node": NodeConfig(cpus=2.0, rpc=True),
            "local-node": NodeConfig(cpus=2.0, rpc=False),
        },
        default_node_id="local-node",
    )
    yield
    nanoray.shutdown()


def test_remote_blocking_variants():
    ref_non_blocking = sleep_and_double.options(pinned_node_id="rpc-node").remote(2, delay=0.1, blocking=False)
    ref_blocking = sleep_and_double.options(pinned_node_id="local-node").remote(3, delay=0.1, blocking=True)

    refs = [r for r in (ref_non_blocking, ref_blocking) if r is not None]
    refs += nanoray.drain()

    values = {nanoray.get(r) for r in refs}
    assert values == {4, 6}

    assert ref_blocking.object_id.startswith("obj-")


def test_actor_blocking_variants():
    actor_ref = Accumulator.options(pinned_node_id="local-node").remote(blocking=True)
    h = nanoray.get(actor_ref)

    first = h.add.remote(1, delay=0.05, blocking=False)
    second = h.add.remote(2, delay=0.05, blocking=True)

    refs = [r for r in (first, second) if r is not None]
    refs += nanoray.drain()

    results = sorted(nanoray.get(r) for r in refs)
    assert results[-1] == 3
    assert results[0] in (1, 2)


def test_blocking_reports_unplaceable_task():
    impossible = sleep_and_double.options(pinned_node_id="missing-node")
    with pytest.raises(RuntimeError):
        impossible.remote(1, blocking=True)


def test_blocking_impacts_wall_time_for_tasks_and_actors():
    nanoray.shutdown()
    nanoray.init(nodes={"solo": NodeConfig(cpus=2.0, rpc=False)}, default_node_id="solo")

    delays = [2.0, 4.0]

    start = time.time()
    task_refs = [
        sleep_and_double.options(max_concurrency=len(delays)).remote(0, delay=d, blocking=False) for d in delays
    ]
    task_refs = [r for r in task_refs if r is not None]
    task_refs += nanoray.drain()
    for ref in task_refs:
        nanoray.get(ref)
    non_blocking_task_time = time.time() - start

    start = time.time()
    blocking_task_refs = [
        sleep_and_double.options(max_concurrency=len(delays)).remote(0, delay=d, blocking=True) for d in delays
    ]
    blocking_task_refs = [r for r in blocking_task_refs if r is not None]
    blocking_task_refs += nanoray.drain()
    for ref in blocking_task_refs:
        nanoray.get(ref)
    blocking_task_time = time.time() - start

    print(f"non_blocking_task_time: {non_blocking_task_time}, blocking_task_time: {blocking_task_time}")
    assert 3.5 <= non_blocking_task_time <= 5.5
    assert 5.5 <= blocking_task_time <= 7.0

    actor_ref = Accumulator.options(max_concurrency=len(delays)).remote(blocking=True)
    actor_handle = nanoray.get(actor_ref)

    start = time.time()
    actor_refs = [actor_handle.add.remote(1, delay=d, blocking=False) for d in delays]
    actor_refs = [r for r in actor_refs if r is not None]
    actor_refs += nanoray.drain()
    for ref in actor_refs:
        nanoray.get(ref)
    non_blocking_actor_time = time.time() - start

    start = time.time()
    blocking_actor_refs = [actor_handle.add.remote(1, delay=d, blocking=True) for d in delays]
    blocking_actor_refs = [r for r in blocking_actor_refs if r is not None]
    blocking_actor_refs += nanoray.drain()
    for ref in blocking_actor_refs:
        nanoray.get(ref)
    blocking_actor_time = time.time() - start

    print(f"non_blocking_actor_time: {non_blocking_actor_time}, blocking_actor_time: {blocking_actor_time}")
    assert 3.5 <= non_blocking_actor_time <= 5.5
    assert 5.5 <= blocking_actor_time <= 7.0


if __name__ == '__main__':
    test_blocking_impacts_wall_time_for_tasks_and_actors()
