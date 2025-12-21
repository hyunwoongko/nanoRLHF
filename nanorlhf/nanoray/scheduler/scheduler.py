import heapq
from typing import Dict, List, Tuple, Optional, Protocol

from nanorlhf.nanoray.core.object_ref import ObjectRef
from nanorlhf.nanoray.core.placement import PlacementGroup, PlacementStrategy
from nanorlhf.nanoray.core.task import Task
from nanorlhf.nanoray.scheduler.node_state import NodeState
from nanorlhf.nanoray.scheduler.policies import SchedulingPolicy


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

    def __init__(
        self,
        policy: SchedulingPolicy,
        nodes: Dict[str, tuple[WorkerLike, Dict[str, float]]]
    ):
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
        Find nodes that have enough available capacity for the given task.

        Args:
            task (Task): The task to check.

        Returns:
            List[str]: Node IDs that can run the task.
        """
        if getattr(task, "pinned_node_id", None):
            nid = task.pinned_node_id
            if nid in self.state and self.state[nid].can_run(task):
                return [nid]
            return []

        # 1) Filter by capacity (baseline)
        capacity_ok = [
            nid for nid, state in self.state.items() if state.can_run(task)
        ]

        # 2) If placement grouped, refine the candidate set
        pg_id = getattr(task, "placement_group_id", None)
        if not pg_id:
            return capacity_ok

        pg = self.placement_groups.get(pg_id, None)
        if pg is None:
            raise ValueError(
                "`PlacementGroup` must be registered with the scheduler before use. "
                "Use `nanorlhf.nanoray.create_placement_group(...)` rather than `PlacementGroup(...)` directly. "
                "The `create_placement_group(...)` function will register it automatically."
            )

        assign = self.placement_group_assignment.setdefault(pg_id, {})

        if pg.strategy == PlacementStrategy.PACK:
            # If group already packed to a node, stick to it.
            locked = assign.get("__pack__")
            if locked:
                # The task will only run on the locked node if it has capacity
                return [nid for nid in capacity_ok if nid == locked]
            else:
                # First placement will lock; for new keep capacity_ok
                return capacity_ok

        elif pg.strategy == PlacementStrategy.SPREAD:
            idx = getattr(task, "bundle_index", 0) or 0
            chosen = assign.get(idx)
            if chosen:
                # Already assigned -> stick to it if capacity allows.
                return [nid for nid in capacity_ok if nid == chosen]
            else:
                # Not yet assigned -> prefer unused nodes first.
                used = set(assign.values())
                prefer = [nid for nid in capacity_ok if nid not in used]
                return prefer or capacity_ok

        # Unknown placement strategy -> ignore
        return capacity_ok

    def try_place(self, task: Task) -> Optional[ObjectRef]:
        """
        Attempt to place the task on an eligible node using the scheduling policy.

        Args:
            task (Task): The task to place.

        Returns:
            Optional[ObjectRef]: Result reference if placed, else `None`.
        """
        cands = self.eligible_nodes(task)
        if not cands:
            return None

        nid = self.policy.select(cands)
        if nid is None:
            return None

        st = self.state[nid]
        st.allocate(task)
        try:
            worker = self.workers[nid]
            ref = worker.execute_task(task)
            # If PG, record the final assignment now that placement successful
            pg_id = getattr(task, "placement_group_id", None)
            if pg_id and pg_id in self.placement_groups:
                pg = self.placement_groups[pg_id]
                assign = self.placement_group_assignment.setdefault(pg_id, {})
                if pg.strategy == PlacementStrategy.PACK:
                    assign.setdefault("__pack__", nid)
                elif pg.strategy == PlacementStrategy.SPREAD:
                    idx = getattr(task, "bundle_index", 0) or 0
                    assign.setdefault(idx, nid)
            return ref
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
        Unregister a placement group.

        Args:
            pg_id (str): The ID of the placement group to unregister.
        """
        self.placement_groups.pop(pg_id, None)
        self.placement_group_assignment.pop(pg_id, None)
