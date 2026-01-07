import heapq
from typing import Dict, List, Tuple, Optional, Protocol

from nanorlhf.nanoray.core.object_ref import ObjectRef
from nanorlhf.nanoray.core.placement import PlacementGroup, PlacementStrategy, Bundle
from nanorlhf.nanoray.core.task import Task
from nanorlhf.nanoray.scheduler.node_state import NodeState
from nanorlhf.nanoray.scheduler.policies import SchedulingPolicy


def sum_custom_resources(bundles: List[Bundle]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for b in bundles:
        if b.resources:
            for k, v in b.resources.items():
                out[k] = float(out.get(k, 0.0)) + float(v)
    return out


def is_dummy_bundle(b) -> bool:
    return b is None or (float(b.cpus) == 0.0 and float(b.gpus) == 0.0 and not (b.resources or {}))


class WorkerLike(Protocol):
    """
    Minimal execution surface the scheduler needs.

    Discussion:
        Q. What is `Protocol`?
            A structural interface (duck typing) that lets us avoid cyclic imports.
            Both `Worker` and `RemoteWorkerProxy` can satisfy this protocol.
    """

    def execute_task(self, task: Task) -> ObjectRef: ...


class Scheduler:
    """
    Minimal scheduler implementation

    Args:
        policy (SchedulingPolicy): Node selection strategy (e.g., `FIFO`, `RoundRobin`)
        nodes (Dict[str, Tuple[WorkerLike, Dict[str, Any]]]):
            Mapping of `node_id -> (worker, capacity_dict)`.
            `capacity_dict` fields: `{"cpus": float, "gpus": float, "resources": dict}`.

    Examples:
        >>> from nanorlhf.nanoray.scheduler.policies import RoundRobin
        >>> from nanorlhf.nanoray.core.object_store import ObjectStore
        >>> from nanorlhf.nanoray.core.task import Task
        >>> from nanorlhf.nanoray.runtime.worker import Worker
        >>> def add(x, y): return x + y
        >>> nodes = {
        ...   "A": (Worker(store=ObjectStore("A")), {"cpus": 1.0, "gpus": 0.0, "resources": {}}),
        ...   "B": (Worker(store=ObjectStore("B")), {"cpus": 2.0, "gpus": 0.0, "resources": {}}),
        ... }
        >>> sched = Scheduler(policy=RoundRobin(), nodes=nodes)
        >>> tasks = [Task.from_call(add, (i, i)) for i in range(4)]
        >>> refs = [sched.submit(s) for s in tasks] + sched.drain()
        >>> all(r is not None for r in refs)
        True

    Discussion:
        Q. What does the scheduler do end-to-end?
            (1) Accepts `Task`s
            (2) Chooses a node using a `SchedulingPolicy`
            (3) Executes on that node's `Worker`
            (4) Returns the produced `ObjectRef`

        Q. Why `WorkerLike`?
            To decouple placement from execution transport. A local `Worker` and a
            remote-RPC-backed `WorkerProxy` can both satisfy the same protocol,
            so the scheduler doesn't need to know *how* execution happens.
    """

    def __init__(self, policy: SchedulingPolicy, nodes: Dict[str, tuple[WorkerLike, Dict[str, float]]]):
        self.policy = policy
        self.nodes = nodes
        self.workers: Dict[str, WorkerLike] = {}
        self.state: Dict[str, NodeState] = {}

        order: List[str] = []
        for nid, (worker, cap) in nodes.items():
            self.workers[nid] = worker
            self.state[nid] = NodeState(
                total_cpus=cap.get("cpus", 1.0),
                total_gpus=cap.get("gpus", 0.0),
                total_custom=cap.get("resources", {}),
            )
            order.append(nid)

        self.order = order
        self.policy.set_node_order(order)

        self.queue: List[Tuple[int, Task]] = []  # (seq, task)
        self.seq = 0

        # placement groups
        self.placement_groups: Dict[str, PlacementGroup] = {}
        self.placement_group_assignment: Dict[str, Dict[object, str]] = {}

    def submit(self, task: Task) -> Optional[ObjectRef]:
        """
        Try to place and execute the task immediately.
        Otherwise, enqueue it for later placement.

        Args:
            task (Task): Declarative description of a remote function call.

        Returns:
            Optional[ObjectRef]: Result reference if placed now, else `None`.
        """
        ref = self.try_place(task)
        if ref is not None:
            return ref
        heapq.heappush(self.queue, (self.seq, task))
        self.seq += 1
        return None

    def drain(self) -> List[ObjectRef]:
        """
        Keep scheduling until queue is empty or no further progress can be made.

        Returns:
            List[ObjectRef]: Refs for tasks that were scheduled during draining.

        Discussion:
            Q. What is this method doing, step by step?
                We run in "rounds" (passes) over the pending queue:

                1) Start a round with `progressed=False` and an empty `pending`.

                2) Pop every task in order (FIFO).
                   For each task:
                     - Try to place it now (`_try_place`).
                     - If placed: append the returned `ObjectRef` to `produced`
                       and set `progressed=True`.
                     - If not placed: append the tuple back into `pending`.

                3) After inspecting the whole heap once, push every item in `pending`
                   back into the heap unchanged.

                4) If at least one task ran (`progressed=True`), we do another round,
                   because completed tasks may have freed resources and unlocked others.
                   If none ran (`progressed=False`), looping again would not change the
                   state, so we stop.

                In short:
                    - Each round tries to place every pending task once.
                    - Tasks that cannot be placed remain in the queue for future rounds.
                    - We stop when either the queue is empty or a full round makes no progress.

            Q. Does it terminate?
                Yes. Either the heap empties (everything placed) or a whole round
                makes no placements (`progressed=False`), so another round would be
                identical and we exit deterministically.
        """
        produced: List[ObjectRef] = []
        progressed = True

        while self.queue and progressed:
            progressed = False
            pending: List[Tuple[int, Task]] = []
            while self.queue:
                seq, task = heapq.heappop(self.queue)
                ref = self.try_place(task)
                if ref is None:
                    pending.append((seq, task))
                else:
                    produced.append(ref)
                    progressed = True
            for item in pending:
                heapq.heappush(self.queue, item)
        return produced

    def eligible_nodes(self, task: Task) -> List[str]:
        """
        Get the list of node IDs that can run the given task.

        Args:
            task (Task): The task to evaluate.

        Returns:
            List[str]: List of eligible node IDs.

        Discussion:
            Q. How are placement groups handled here?
                If the task is part of a placement group, we check the group's
                strategy (PACK or SPREAD) and the bundle requirements to filter
                eligible nodes accordingly.

            Q. What if the task is pinned to a specific node?
                If the task has a `pinned_node_id`, we only consider that node
                for eligibility, applying placement group constraints if applicable.
        """
        pg_id = getattr(task, "placement_group_id", None)
        bundle_index = getattr(task, "bundle_index", 0) or 0

        def task_fits_in_bundle(b) -> None:
            if task.num_cpus > float(b.cpus):
                raise ValueError(f"Task num_cpus exceeds bundle cpus: task={task.num_cpus}, bundle={b.cpus}")
            if task.num_gpus > float(b.gpus):
                raise ValueError(f"Task num_gpus exceeds bundle gpus: task={task.num_gpus}, bundle={b.gpus}")
            req = task.resources or {}
            cap = b.resources or {}
            for k, v in req.items():
                if float(v) > float(cap.get(k, 0.0)):
                    raise ValueError(f"Task resource exceeds bundle: key={k}, task={v}, bundle={cap.get(k, 0.0)}")

        def get_pg_and_bundle():
            if not pg_id:
                return None, None
            pg = self.placement_groups.get(pg_id, None)
            if pg is None:
                raise ValueError(
                    "PlacementGroup must be registered before use. "
                    "Use nanorlhf.nanoray.create_placement_group(...) to create it."
                )
            if bundle_index < 0 or bundle_index >= len(pg.bundles):
                raise ValueError(
                    f"bundle_index out of range: bundle_index={bundle_index}, num_bundles={len(pg.bundles)}"
                )
            b = pg.bundle(bundle_index)
            if not is_dummy_bundle(b):
                task_fits_in_bundle(b)
            return pg, b

        pg, bundle = get_pg_and_bundle()
        dummy = is_dummy_bundle(bundle)

        pinned = getattr(task, "pinned_node_id", None)
        if pinned:
            if pinned not in self.state:
                return []

            st = self.state[pinned]

            if not pg_id or dummy:
                return [pinned] if st.can_run(task) else []

            assign = self.placement_group_assignment.setdefault(pg_id, {})
            if pg.strategy == PlacementStrategy.PACK:
                locked = assign.get("__pack__")
                if locked and locked != pinned:
                    return []
                if locked == pinned:
                    return [pinned]
                can = st.can_reserve_bundle(
                    Bundle(
                        cpus=sum(float(b.cpus) for b in pg.bundles),
                        gpus=sum(float(b.gpus) for b in pg.bundles),
                        resources=sum_custom_resources(pg.bundles),
                    )
                )
                return [pinned] if can else []

            if pg.strategy == PlacementStrategy.SPREAD:
                chosen = assign.get(bundle_index)
                if chosen and chosen != pinned:
                    return []
                if chosen == pinned:
                    return [pinned]
                return [pinned] if st.can_reserve_bundle(bundle) else []

            return [pinned]

        if not pg_id or dummy:
            return [nid for nid, st in self.state.items() if st.can_run(task)]

        assign = self.placement_group_assignment.setdefault(pg_id, {})

        if pg.strategy == PlacementStrategy.PACK:
            locked = assign.get("__pack__")
            if locked:
                return [locked] if (locked in self.state) else []

            total_bundle = Bundle(
                cpus=sum(float(b.cpus) for b in pg.bundles),
                gpus=sum(float(b.gpus) for b in pg.bundles),
                resources=sum_custom_resources(pg.bundles),
            )
            return [nid for nid, st in self.state.items() if st.can_reserve_bundle(total_bundle)]

        if pg.strategy == PlacementStrategy.SPREAD:
            chosen = assign.get(bundle_index)
            if chosen:
                return [chosen] if (chosen in self.state) else []

            capacity_ok = [nid for nid, st in self.state.items() if st.can_reserve_bundle(bundle)]
            used = set(assign.values())
            prefer = [nid for nid in capacity_ok if nid not in used]
            return prefer or capacity_ok

        return list(self.state.keys())

    def try_place(self, task: Task) -> Optional[ObjectRef]:
        """
        Try to place and execute the task immediately.

        Args:
            task (Task): Declarative description of a remote function call.

        Returns:
            Optional[ObjectRef]: Result reference if placed now, else `None`.

        Discussion:
            Q. How are placement groups handled here?
                If the task is part of a placement group, we check the group's
                strategy (PACK or SPREAD) and the bundle requirements to
                reserve resources accordingly before execution.

            Q. What happens if the task cannot be placed?
                If no eligible node is found or if placement group constraints
                cannot be satisfied, the method returns `None`.

            Q. What if the task is successfully placed?
                The task is executed on the selected worker, and an `ObjectRef`
                to the result is returned.
        """
        cands = self.eligible_nodes(task)
        if not cands:
            return None

        nid = self.policy.select(cands)
        if nid is None:
            return None

        pg_id = getattr(task, "placement_group_id", None)
        bundle_index = getattr(task, "bundle_index", 0) or 0

        pg = None
        bundle = None
        if pg_id:
            pg = self.placement_groups.get(pg_id, None)
            if pg is None:
                raise ValueError(
                    "PlacementGroup must be registered before use. "
                    "Use nanorlhf.nanoray.create_placement_group(...) to create it."
                )
            if bundle_index < 0 or bundle_index >= len(pg.bundles):
                raise ValueError(
                    f"bundle_index out of range: bundle_index={bundle_index}, num_bundles={len(pg.bundles)}"
                )
            bundle = pg.bundle(bundle_index)

        use_reservation = bool(pg_id and pg is not None and not is_dummy_bundle(bundle))

        st = self.state[nid]

        if use_reservation:
            assign = self.placement_group_assignment.setdefault(pg_id, {})

            if pg.strategy == PlacementStrategy.PACK:
                locked = assign.get("__pack__")
                if locked is None:
                    for b in pg.bundles:
                        st.reserve_bundle(b)
                    assign["__pack__"] = nid
                elif locked != nid:
                    return None

            elif pg.strategy == PlacementStrategy.SPREAD:
                chosen = assign.get(bundle_index)
                if chosen is None:
                    st.reserve_bundle(bundle)
                    assign[bundle_index] = nid
                elif chosen != nid:
                    return None

            worker = self.workers[nid]
            return worker.execute_task(task)

        st.allocate(task)
        try:
            worker = self.workers[nid]
            return worker.execute_task(task)
        finally:
            st.release(task)

    def register_placement_group(self, pg: PlacementGroup):
        """
        Register a placement group so the scheduler can honor it.

        Args:
            pg (PlacementGroup): The placement group to register.
        """
        self.placement_groups[pg.pg_id] = pg
        self.placement_group_assignment.setdefault(pg.pg_id, {})

    def unregister_placement_group(self, pg_id: str):
        """
        Unregister a placement group and release its reserved resources.

        Args:
            pg_id (str): The ID of the placement group to unregister.

        Discussion:
            Q. What happens to resources reserved by the placement group?
                The method releases all resources reserved by the placement group
                on the nodes where they were allocated.
        """
        pg = self.placement_groups.get(pg_id, None)
        assign = self.placement_group_assignment.get(pg_id, {})

        if pg is not None:
            if pg.strategy == PlacementStrategy.PACK:
                locked = assign.get("__pack__")
                if locked and locked in self.state:
                    st = self.state[locked]
                    for b in pg.bundles:
                        st.release_bundle(b)

            elif pg.strategy == PlacementStrategy.SPREAD:
                for k, nid in list(assign.items()):
                    if k == "__pack__":
                        continue
                    if nid not in self.state:
                        continue
                    idx = int(k)  # noqa
                    if idx < 0 or idx >= len(pg.bundles):
                        continue
                    self.state[nid].release_bundle(pg.bundles[idx])

        self.placement_groups.pop(pg_id, None)
        self.placement_group_assignment.pop(pg_id, None)