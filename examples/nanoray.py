from collections import Counter
from typing import List, Tuple

from nanorlhf import nanoray


@nanoray.remote
def add(a: int, b: int) -> int:
    return a + b


@nanoray.remote
def whoami() -> Tuple[str, int]:
    import os, platform
    return platform.node(), os.getpid()


@nanoray.remote
class TokenizerActor:
    def __init__(self):
        self.counts = {}

    def analyze(self, texts: List[str], topk: int = 3):
        from collections import Counter
        import re

        def toks(s: str):
            return [t for t in re.findall(r"[a-zA-Z]+", s.lower()) if t]

        for s in texts:
            for t in toks(s):
                self.counts[t] = self.counts.get(t, 0) + 1

        return Counter(self.counts).most_common(topk)

    def owner(self) -> Tuple[str, int]:
        import os, platform
        return platform.node(), os.getpid()


def banner(title: str):
    print(f"\n=== {title} ===")


def show_rows(title: str, rows, n: int = 5):
    banner(title)
    for i, r in enumerate(list(rows)[:n]):
        print(f"{i}: {r}")
    print()


def show_owners(title: str, refs):
    owners = [r.owner_node_id for r in refs]
    c = Counter(owners)
    banner(title)
    print("owners:", owners)
    print("counts:", dict(c))
    print()
    return owners, c


def main():
    # 0) Init (LOCAL multi-worker)
    cfg = {
        "A": nanoray.NodeConfig(cpus=1.0, rpc=True, port=8080),
        "B": nanoray.NodeConfig(cpus=1.0, rpc=True, port=8081),
    }
    nanoray.init(cfg, default_node_id="A")
    banner("Init")
    print("mode: LOCAL multi-worker (nodes=A,B)\n")

    # 1) put/get
    ref = nanoray.put({"hello": "world"})
    val = nanoray.get(ref)
    show_rows("put/get", [val], n=1)

    # 2) remote function calls
    refs = [add.remote(i, i) for i in range(10)]
    refs = [r for r in refs if r] + nanoray.drain()
    vals = [nanoray.get(r) for r in refs]
    show_rows("add.remote results", vals, n=5)
    show_owners("add.remote owners", refs)

    # 3) Placement Group: PACK
    pg_pack = nanoray.create_placement_group(bundles=[{"cpus": 1.0}], strategy="PACK")
    refs_pack = [whoami.options(placement_group=pg_pack).remote() for _ in range(6)]
    refs_pack = [r for r in refs_pack if r] + nanoray.drain()
    vals_pack = [nanoray.get(r) for r in refs_pack]
    show_rows("PG=PACK whoami()", vals_pack, n=5)
    show_owners("PG=PACK owners", refs_pack)

    # 4) Placement Group: SPREAD
    pg_spread = nanoray.create_placement_group(
        bundles=[{"cpus": 1.0}, {"cpus": 1.0}],
        strategy="SPREAD",
    )
    refs_spread = []
    for i in range(8):
        refs_spread.append(
            whoami.options(placement_group=pg_spread, bundle_index=i % 2).remote()
        )
    refs_spread = [r for r in refs_spread if r] + nanoray.drain()
    vals_spread = [nanoray.get(r) for r in refs_spread]
    show_rows("PG=SPREAD whoami()", vals_spread, n=5)
    show_owners("PG=SPREAD owners", refs_spread)

    # 5) Actors + SPREAD: analyze texts on different bundles
    texts = [
        "nanoray is tiny but mighty",
        "actors keep heavy state in memory",
        "in contrast task is stateless",
        "placement groups influence where tasks land",
        "tiny runtime, big wins",
    ]

    a1_ref = TokenizerActor.options(
        placement_group=pg_spread, bundle_index=0
    ).remote()
    a2_ref = TokenizerActor.options(
        placement_group=pg_spread, bundle_index=1
    ).remote()
    a1 = nanoray.get(a1_ref)
    a2 = nanoray.get(a2_ref)

    # sticky calls to each actor
    r1 = a1.analyze.remote(texts[:3], topk=3)
    r2 = a2.analyze.remote(texts[2:], topk=3)
    r1, r2 = [r for r in (r1, r2) if r] + nanoray.drain()
    top1 = nanoray.get(r1)
    top2 = nanoray.get(r2)

    o1 = nanoray.get(a1.owner.remote())
    o2 = nanoray.get(a2.owner.remote())
    owners_actor = [a1_ref.owner_node_id, a2_ref.owner_node_id]

    banner("Actor (SPREAD) summary")
    print("actors owners:", owners_actor, "unique:", set(owners_actor))
    print("owner info:", {"a1": o1, "a2": o2})
    print("top-3 per actor:", {"bundle-0": top1, "bundle-1": top2})
    print()

    # cleanup
    nanoray.remove_placement_group(pg_pack.pg_id)
    nanoray.remove_placement_group(pg_spread.pg_id)
    nanoray.shutdown()
    banner("Shutdown")


if __name__ == "__main__":
    main()
