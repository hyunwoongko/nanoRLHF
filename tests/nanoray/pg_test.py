import pytest

from nanorlhf import nanoray
from nanorlhf.nanoray.api.initialization import NANORAY_BASE_PORT
from nanorlhf.nanoray.core.placement import Bundle, PlacementStrategy


@nanoray.remote
class ProbeActor:
    def __init__(self, name: str):
        self.name = name

    def ping(self) -> str:
        return self.name


@pytest.fixture(scope="function", autouse=True)
def nanoray_session():
    nodes = {
        "node-A": nanoray.NodeConfig(rpc=True, host="127.0.0.1", port=NANORAY_BASE_PORT + 101, cpus=2.0, gpus=0.0),
        "node-B": nanoray.NodeConfig(rpc=True, host="127.0.0.1", port=NANORAY_BASE_PORT + 102, cpus=2.0, gpus=0.0),
    }
    nanoray.init(nodes, default_node_id="node-A")
    yield
    nanoray.shutdown()


def test_pack():
    print("\n== TEST: PACK ==")

    pg = nanoray.create_placement_group(
        bundles=[
            Bundle(cpus=1.0, gpus=0.0, resources={}),
            Bundle(cpus=1.0, gpus=0.0, resources={}),
        ],
        strategy=PlacementStrategy.PACK,
    )

    ref0 = ProbeActor.options(placement_group=pg, bundle_index=0).remote("a0", blocking=True)
    ref1 = ProbeActor.options(placement_group=pg, bundle_index=1).remote("a1", blocking=True)

    actor0 = nanoray.get(ref0)
    actor1 = nanoray.get(ref1)

    print(f"actor0.owner_node_id = {actor0.owner_node_id}")
    print(f"actor1.owner_node_id = {actor1.owner_node_id}")

    assert actor0.owner_node_id == actor1.owner_node_id, "PACK인데 서로 다른 노드에 배치되었습니다."

    r0 = actor0.ping.remote(blocking=True)
    r1 = actor1.ping.remote(blocking=True)
    v0, v1 = nanoray.get([r0, r1])
    assert v0 == "a0" and v1 == "a1"


def test_spread():
    print("\n== TEST: SPREAD ==")

    pg = nanoray.create_placement_group(
        bundles=[
            Bundle(cpus=1.0, gpus=0.0, resources={}),
            Bundle(cpus=1.0, gpus=0.0, resources={}),
        ],
        strategy=PlacementStrategy.SPREAD,
    )

    ref0a = ProbeActor.options(placement_group=pg, bundle_index=0).remote("b0a", blocking=True)
    ref1a = ProbeActor.options(placement_group=pg, bundle_index=1).remote("b1a", blocking=True)

    actor0a = nanoray.get(ref0a)
    actor1a = nanoray.get(ref1a)

    print(f"actor0a.owner_node_id = {actor0a.owner_node_id}")
    print(f"actor1a.owner_node_id = {actor1a.owner_node_id}")

    assert actor0a.owner_node_id != actor1a.owner_node_id, "SPREAD인데 bundle_index 0/1이 같은 노드에 배치되었습니다."

    ref0b = ProbeActor.options(placement_group=pg, bundle_index=0).remote("b0b", blocking=True)
    ref1b = ProbeActor.options(placement_group=pg, bundle_index=1).remote("b1b", blocking=True)

    actor0b = nanoray.get(ref0b)
    actor1b = nanoray.get(ref1b)

    print(f"actor0b.owner_node_id = {actor0b.owner_node_id}")
    print(f"actor1b.owner_node_id = {actor1b.owner_node_id}")

    assert actor0b.owner_node_id == actor0a.owner_node_id, "SPREAD에서 bundle_index=0의 노드 고정이 깨졌습니다."
    assert actor1b.owner_node_id == actor1a.owner_node_id, "SPREAD에서 bundle_index=1의 노드 고정이 깨졌습니다."

    vals = nanoray.get(
        [
            actor0a.ping.remote(blocking=True),
            actor1a.ping.remote(blocking=True),
            actor0b.ping.remote(blocking=True),
            actor1b.ping.remote(blocking=True),
        ]
    )
    assert vals == ["b0a", "b1a", "b0b", "b1b"]