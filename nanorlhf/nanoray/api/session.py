from typing import Dict, Tuple, Any, Optional, List, Union

from nanorlhf.nanoray.core.object_ref import ObjectRef
from nanorlhf.nanoray.core.object_store import ObjectStore
from nanorlhf.nanoray.core.placement import Bundle, PlacementStrategy, PlacementGroup
from nanorlhf.nanoray.core.task import Task
from nanorlhf.nanoray.network.router import Router
from nanorlhf.nanoray.network.rpc_client import RpcClient
from nanorlhf.nanoray.scheduler.policies import SchedulingPolicy
from nanorlhf.nanoray.scheduler.scheduler import Scheduler, WorkerLike
from nanorlhf.nanoray.utils import new_placement_group_id, task_result_object_id


class Session:
    """
    Runtime session that owns a `Scheduler` and exposes driver utilities.

    This class centralizes the "driver-side" API and networking hooks:
    it submits tasks to the scheduler, drains the queue, and retrieves
    values referenced by `ObjectRef`s across local and remote nodes.

    Args:
        policy (SchedulingPolicy): Node selection policy (e.g., FIFO/RoundRobin).
        nodes (Dict[str, Tuple[WorkerLike, Dict[str, Any]]]):
            Mapping of `node_id -> (worker, capacity_dict)`.
            `capacity_dict` fields: `{"cpus": float, "gpus": float, "resources": dict}`.
        local_workers (Dict[str, WorkerLike]): Mapping of local `node_id -> worker`.
        default_node_id (Optional[str]): Node to use for `put()` convenience.
        router (Optional[Router]): Router used for remote object placement/lookup.
        rpc (Optional[RpcClient]): RPC client used for remote object transfer.

    Attributes:
        scheduler (Scheduler): The scheduler instance.
        workers (Dict[str, WorkerLike]): All workers (local + remote proxies).
        local_workers (Dict[str, WorkerLike]): Only local workers (expose `.store`).
        default_node_id (str): Default node id for `put()` or caching.
        driver_store (Optional[ObjectStore]): Fallback cache when there is no local worker.
        router (Optional[Router]): Router for object routing (maybe None).
        rpc (Optional[RpcClient]): Rpc client for remote fetch (maybe None).
        aliases (Dict[str, str]): Mapping remote object_id -> local cached object_id.

    Examples:
        >>> # Minimal single-node session (local only)
        >>> from nanorlhf.nanoray.core.object_store import ObjectStore
        >>> from nanorlhf.nanoray.runtime.worker import Worker
        >>> from nanorlhf.nanoray.scheduler.policies import FIFO
        >>> nodes = {"A": (Worker(store=ObjectStore("A")), {"cpus": 1.0, "gpus": 0.0, "resources": {}})}
        >>> sess = Session(policy=FIFO(), nodes=nodes, default_node_id="A")
        >>> ref = sess.put(123)
        >>> sess.get(ref)
        123

    Discussion:
        Q. Why a "driver cache" `ObjectStore`?
            In multi-node tests where all workers are remote (no local `.store`),
            we still need a local place to cache remotely fetched bytes for fast
            subsequent `get()`. A private driver-side `ObjectStore("__driver__")`
            keeps the logic simple and explicit.

        Q. Why keep an alias map instead of reusing the remote id?
            Local stores generate their own `object_id`. After fetching bytes from
            a remote owner, we re-materialize the object locally; the alias map
            remembers that "remote_id -> local_id" for instant local hits later.

        Q. What is the `get()` lookup order?
            (1) alias cache → (2) owner-first if the owner is local →
            (3) scan local workers → (4) fetch from remote (router+rpc) and cache.
    """

    def __init__(
        self,
        policy: SchedulingPolicy,
        nodes: Dict[str, Tuple[WorkerLike, Dict[str, Any]]],
        local_workers: Dict[str, WorkerLike] = None,
        default_node_id: Optional[str] = None,
        router: Optional[Router] = None,
        rpc: Optional[RpcClient] = None,
    ):
        self.scheduler = Scheduler(policy=policy, nodes=nodes)

        # All workers (local + remote proxies)
        self.workers: Dict[str, WorkerLike] = {nid: w for nid, (w, _) in nodes.items()}

        # Only local workers (those exposing a `.store` attribute)
        self.local_workers: Dict[str, WorkerLike] = local_workers

        # Default node id resolution and driver cache setup
        if self.local_workers:
            self.default_node_id = (
                default_node_id if (default_node_id in self.local_workers) else next(iter(self.local_workers))
            )
            self.driver_store: Optional[ObjectStore] = None
        else:
            # No local workers: pick any known id for identification
            self.default_node_id = (
                default_node_id if (default_node_id in self.workers) else next(iter(self.workers))
            )
            # Driver-side cache for remote fetches
            self.driver_store = ObjectStore("__driver__")

        self.router = router
        self.rpc = rpc

        # Remote object_id -> local cached object_id
        self.aliases: Dict[str, str] = {}

    def cache_store(self) -> ObjectStore:
        """
        Return the store used to cache fetched objects locally.

        Returns:
            ObjectStore: Default local worker's store if present, otherwise the driver store.
        """
        if self.local_workers:
            return getattr(self.local_workers[self.default_node_id], "store")
        assert self.driver_store is not None
        return self.driver_store

    def submit(self, task: Task, blocking=False) -> Optional[ObjectRef]:
        """
        Submit a task to the scheduler (enqueue-only).

        The teaching scheduler is enqueue-only: this method always returns
        `None`. Call `drain()` to advance the queue and collect produced refs.

        Args:
            task (Task): Declarative description of a remote function call.
            blocking (bool): If True, use `submit_blocking()` to ensure an `ObjectRef` is returned.

        Returns:
            Optional[ObjectRef]: Always `None` in the enqueue-only model.

        Examples:
            >>> # build tasks, submit all, then drain()
            >>> refs = []
            >>> for i in range(3):
            ...     _ = sess.submit(Task.from_call(lambda x: x + 1, args=(i,)))
            >>> refs += sess.drain()
        """

        ref = self.scheduler.submit(task)
        if ref is not None:
            if blocking:
                self.get(ref)
            return ref
        if blocking is True:
            produced = self.scheduler.drain()
            if not produced:
                raise RuntimeError(
                    "Task could not be placed (no progress). " "Check resources/placement group/pin constraints."
                )
            expected_oid = task_result_object_id(task.task_id)
            chosen = None
            for candidate in produced:
                if candidate.object_id == expected_oid:
                    chosen = candidate
                    break
            if chosen is None:
                chosen = produced[-1]

            self.get(chosen)
            return chosen
        else:
            return None

    def drain(self) -> List[ObjectRef]:
        """
        Advance scheduling until the pending queue no longer progresses.

        Returns:
            List[ObjectRef]: Refs produced during draining.

        Examples:
            >>> # After submit(), use drain() to actually run and collect results
            >>> refs = sess.drain()
            >>> values = [sess.get(r) for r in refs]
        """
        return self.scheduler.drain()

    def put(self, value: Any, *, node_id: Optional[str] = None) -> ObjectRef:
        """
        Store a Python object and return an `ObjectRef`.

        If there is at least one local worker, the value is stored in that
        worker's `ObjectStore` (by `node_id` or by the default local node).
        If there are no local workers, the value is stored in the driver cache.

        Args:
            value (Any): Python object to store.
            node_id (Optional[str]): Target local node id; defaults to the default local node.

        Returns:
            ObjectRef: Handle to the stored value.

        Discussion:
            Q. Why allow `put()` with no local workers?
                For remote-only experiments, it is useful to still put values
                deterministically into the driver cache (e.g., for small config blobs).
        """
        if self.local_workers:
            nid = node_id if (node_id in self.local_workers) else self.default_node_id
            worker = self.local_workers[nid]
            return worker.store.put(value)  # type: ignore[attr-defined]
        return self.cache_store().put(value)

    def _get(self, ref: Optional[ObjectRef]) -> Any:
        """
        Retrieve the Python value for a given (possibly pending) reference.

        Lookup/drive order:
            0) If `ref is None`, drive scheduling once to materialize the most recent result
            1) alias cache
            2) owner-first (only if the owner is local)
            3) scan local workers
            4) drive scheduling rounds until no further progress
            5) remote fetch via router+rpc (then cache locally)

        Args:
            ref (Optional[ObjectRef]): Handle to fetch. If `None`, this means the call
                was enqueued but not placed yet; we will drive the scheduler once and
                use the most recently produced result as the target.

        Returns:
            Any: The stored Python object.

        Raises:
            RuntimeError: If object cannot be found locally and networking is not configured.

        Discussion:
            Q. Why accept `None`?
                In this teaching runtime, `remote()` may enqueue without immediate placement.
                Accepting `None` lets `get()` mimic Ray's UX by driving scheduling to
                materialize the most recent result and then fetching it.
        """

        # 0) If ref is None, drive once and take the most recently produced ref.
        if ref is None:
            produced = self.scheduler.drain()
            if not produced:
                raise RuntimeError(
                    "get(None) was called but no tasks progressed. "
                    "Make sure you submitted something before calling get()."
                )
            ref = produced[-1]

        def _try_local_lookup() -> Optional[Any]:
            # 1) alias cache fast-path
            local_id = self.aliases.get(ref.object_id)
            if local_id is not None:
                store = self.cache_store()
                if store.has(local_id):
                    return store.get(ObjectRef(object_id=local_id, owner_node_id=getattr(store, "node_id", None)))

            # 2) owner-first (when owner is local)
            if ref.owner_node_id and (ref.owner_node_id in self.local_workers):
                store = getattr(self.local_workers[ref.owner_node_id], "store")
                if store.has(ref.object_id):
                    return store.get(ref)

            # 3) scan local workers
            for w in self.local_workers.values():
                store = getattr(w, "store")
                if store.has(ref.object_id):
                    return store.get(ref)

            return None

        # First attempt
        val = _try_local_lookup()
        if val is not None:
            return val

        # 4) Drive scheduling in rounds until no progress
        progressed = True
        while val is None and progressed:
            produced = self.scheduler.drain()
            progressed = bool(produced)
            val = _try_local_lookup()

        if val is not None:
            return val

        # 5) remote fetch (requires router+rpc)
        if self.router is not None and self.rpc is not None:
            owner_id = self.router.route_object(ref) or ref.owner_node_id
            if owner_id is None:
                raise RuntimeError("No owner information available for remote get().")
            payload = self.rpc.get_object(owner_id, ref.object_id)
            cache = self.cache_store()
            local_ref = cache.put_bytes(payload)  # new local id
            self.aliases[ref.object_id] = local_ref.object_id
            return cache.get(local_ref)

        raise RuntimeError(
            "Object not found in local stores and networking hooks are not configured. "
            "Provide `router` and `rpc` to enable remote get()."
        )

    def get(self, ref: Union[ObjectRef, List[ObjectRef], None]) -> Union[Any, List[Any]]:
        """
        Retrieve multiple Python values for a list of (possibly pending) references.

        Args:
            ref (Union[ObjectRef, List[ObjectRef], None]): Handle(s) to fetch.
                If `None`, this means the call was enqueued but not placed yet;
                we will drive the scheduler once and use the most recently produced
                result as the target. If a list, all entries are fetched.

        Returns:
            Union[Any, List[Any]]: The stored Python object(s).
        """
        if not isinstance(ref, (list, tuple)):
            return self._get(ref)
        else:
            expected = len(ref)
            object_refs = [r for r in ref if r is not None]
            while len(object_refs) < expected:
                produced = self.drain()
                if not produced:
                    raise RuntimeError("`nanoray` calls could not be placed; no progress during drain.")
                object_refs.extend(produced)
            return [self._get(r) for r in object_refs[:expected]]

    def create_placement_group(
        self,
        bundles: List[Union[Bundle, Dict]],
        strategy: str = PlacementStrategy.PACK,
        *,
        pg_id: Optional[str] = None,
    ) -> PlacementGroup:
        """
        Create and register a placement group on this session's scheduler.

        Args:
            bundles (List[Union[Bundle, Dict]]): Resource bundles.
            strategy (str): Placement strategy, either `PACK` or `SPREAD`.
            pg_id (Optional[str]): Optional stable id; autogenerated if `None`.

        Returns:
            PlacementGroup: The created placement group.
        """
        assert strategy.upper() in (
            PlacementStrategy.PACK,
            PlacementStrategy.SPREAD,
        ), f"`strategy` must be either PACK or SPREAD, got {strategy}."

        pid = pg_id or new_placement_group_id()

        bundles_to_use = []
        for b in bundles:
            if isinstance(b, Bundle):
                bundles_to_use.append(b)
            elif isinstance(b, dict):
                bundles_to_use.append(Bundle(**b))
            else:
                raise ValueError(f"Each bundle must be a Bundle or dict, got {type(b)}.")

        pg = PlacementGroup(pg_id=pid, bundles=bundles_to_use, strategy=strategy)
        self.scheduler.register_placement_group(pg)
        return pg

    def remove_placement_group(self, pg_id: str):
        """
        Remove a placement group from this session's scheduler.

        Args:
            pg_id (str): id of the placement group to remove.
        """
        self.scheduler.unregister_placement_group(pg_id)


GLOBAL_SESSION: Optional[Session] = None


def init_session(
    policy: SchedulingPolicy,
    nodes: Dict[str, Tuple[WorkerLike, Dict[str, Any]]],
    local_workers: Dict[str, WorkerLike] = None,
    default_node_id: Optional[str] = None,
) -> Session:
    """
    Initialize the global session.

    Args:
        policy (SchedulingPolicy): Node selection policy (e.g., FIFO/RoundRobin).
        nodes (Dict[str, Tuple[WorkerLike, Dict[str, Any]]]):
            Mapping of `node_id -> (worker, capacity_dict)`.
            `capacity_dict` fields: `{"cpus": float, "gpus": float, "resources": dict}`.
        local_workers (Dict[str, WorkerLike]): Mapping of local `node_id -> worker`.
        default_node_id (Optional[str]): Node to use for `put()` convenience.

    Returns:
        Session: The created global session.

    Examples:
        >>> from nanorlhf.nanoray.scheduler.policies import RoundRobin
        >>> from nanorlhf.nanoray.core.object_store import ObjectStore
        >>> from nanorlhf.nanoray.runtime.worker import Worker
        >>> nodes = {
        ...   "A": (Worker(store=ObjectStore("A")), {"cpus": 1.0, "gpus": 0.0, "resources": {}}),
        ...   "B": (Worker(store=ObjectStore("B")), {"cpus": 1.0, "gpus": 0.0, "resources": {}}),
        ... }
        >>> _ = init_session(RoundRobin(), nodes, default_node_id="A")
    """
    global GLOBAL_SESSION
    GLOBAL_SESSION = Session(
        policy=policy,
        nodes=nodes,
        local_workers=local_workers,
        default_node_id=default_node_id,
    )
    return GLOBAL_SESSION


def get_session() -> Session:
    """
    Return the global session.

    Returns:
        Session: The global session.

    Raises:
        RuntimeError: If the global session has not been initialized.

    Examples:
        >>> _ = get_session()  # doctest: +ELLIPSIS
    """
    if GLOBAL_SESSION is None:
        raise RuntimeError("Global session is not initialized. Call `init_session(...)` first.")
    return GLOBAL_SESSION


def get(ref: Union[ObjectRef, List[ObjectRef]]) -> Union[Any, List[Any]]:
    """
    Convenience function: fetch a value via the global session.

    Args:
        ref (Union[ObjectRef, List[ObjectRef]]): Handle(s) to fetch.

    Returns:
        Union[Any, List[Any]]: The stored Python object(s).

    Examples:
        >>> # Single-node smoke
        >>> from nanorlhf.nanoray.core.object_store import ObjectStore
        >>> from nanorlhf.nanoray.runtime.worker import Worker
        >>> from nanorlhf.nanoray.scheduler.policies import FIFO
        >>> nodes = {"A": (Worker(store=ObjectStore("A")), {"cpus": 1.0, "gpus": 0.0, "resources": {}})}
        >>> _ = init_session(FIFO(), nodes, default_node_id="A")
        >>> ref = put(123)
        >>> get(ref)
        123
    """
    return get_session().get(ref)


def put(value: Any, *, node_id: Optional[str] = None) -> ObjectRef:
    """
    Convenience function: store a value via the global session.

    Args:
        value (Any): Python object to store.
        node_id (Optional[str]): Target local node id; defaults to the session default.

    Returns:
        ObjectRef: Handle to the stored value.

    Examples:
        >>> ref = put({"x": 1})  # doctest: +ELLIPSIS
    """
    return get_session().put(value, node_id=node_id)


def submit(task: Task, blocking: bool = False) -> Optional[ObjectRef]:
    """
    Convenience function: submit a task via the global session (enqueue-only).

    Args:
        task (Task): Declarative description of a remote function call.
        blocking (bool): If True, use `submit_blocking()` to ensure an `ObjectRef` is returned.

    Returns:
        Optional[ObjectRef]: Always `None` in the enqueue-only model.

    Examples:
        >>> def add(x, y): return x + y
        >>> _ = submit(Task.from_call(add, args=(3, 4)))
        >>> refs = drain()
        >>> get(refs[-1])
        7
    """
    return get_session().submit(task, blocking=blocking)


def drain() -> List[ObjectRef]:
    """
    Convenience function: advance scheduling in the global session.

    Returns:
        List[ObjectRef]: Refs produced during draining.

    Examples:
        >>> refs = drain()  # doctest: +ELLIPSIS
    """
    return get_session().drain()


def create_placement_group(
    bundles: List[Union[Bundle, Dict]], strategy: str = PlacementStrategy.PACK, pg_id: Optional[str] = None
) -> PlacementGroup:
    """
    Convenience function: create a placement group via the global session.

    Args:
        bundles (List[Union[Bundle, Dict]]): Resource bundles.
        strategy (str): Placement strategy, either `PACK` or `SPREAD`.
        pg_id (Optional[str]): Optional stable id; autogenerated if `None`.

    Returns:
        PlacementGroup: The created placement group.
    """
    return get_session().create_placement_group(bundles, strategy, pg_id=pg_id)


def remove_placement_group(pg_id: str):
    """
    Convenience function: remove a placement group via the global session.

    Args:
        pg_id (str): id of the placement group to remove.
    """
    return get_session().remove_placement_group(pg_id)
