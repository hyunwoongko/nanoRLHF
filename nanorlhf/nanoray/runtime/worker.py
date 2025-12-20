from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import nullcontext
from typing import Optional, Dict

from nanorlhf.nanoray.core.actor import ActorRef
from nanorlhf.nanoray.core.object_ref import ObjectRef
from nanorlhf.nanoray.core.object_store import ObjectStore
from nanorlhf.nanoray.core.serialization import dumps, loads
from nanorlhf.nanoray.core.task import Task
from nanorlhf.nanoray.utils import new_actor_id, task_result_object_id
import multiprocessing as mp

_PROCESS_ACTORS: Dict[str, object] = {}


def _invoke(fn, args, kwargs):
    """
    Top-level helper for process-based execution

    Notes:
        - Must be at module top-level so that it is picklable by `multiprocessing`.
        - Mirrors the core execution: `return fn(*args, **kwargs)`.
    """
    return fn(*args, **(kwargs or {}))


def _invoke_serialized(payload: bytes):
    fn, args, kwargs = loads(payload)
    return _invoke(fn, args, kwargs)


def _create_actor_process(actor_id: str, payload: bytes):
    cls, init_args, init_kwargs = loads(payload)
    instance = cls(*init_args, **(init_kwargs or {}))
    _PROCESS_ACTORS[actor_id] = instance
    return actor_id


def _call_actor_process(actor_id: str, method_name: str, payload: bytes):
    args, kwargs = loads(payload)
    instance = _PROCESS_ACTORS.get(actor_id)
    if instance is None:
        raise RuntimeError(f"Actor {actor_id} not found in process")
    method = getattr(instance, method_name, None)
    if method is None or not callable(method):
        raise AttributeError(f"Actor method {method_name} not found")
    return method(*args, **(kwargs or {}))


class Worker:
    """
    Minimal worker that executes `Task`s, stores results into an `ObjectStore`,
    and returns `ObjectRef`s to the caller.

    The worker can run regular function tasks in the current process (default)
    or submit them to a process pool for CPU-bound or isolation-friendly execution.

    **ActorCreate/ActorCall are always executed in-process** because actor instances
    live in this worker's memory.
    """

    def __init__(self, store: ObjectStore, node_id: Optional[str] = None):
        self.store = store
        self.node_id = node_id or store.node_id
        self._actors: Dict[str, object] = {}  # local actor registry
        self._task_executors: Dict[int, ThreadPoolExecutor] = {}
        self._actor_executors: Dict[str, ProcessPoolExecutor] = {}

    def execute_task(self, task: Task) -> ObjectRef:
        """
        Execute the given `Task` and return an `ObjectRef` to the result.

        Args:
            task (Task): Declarative description of a remote function call.

        Returns:
            ObjectRef: Handle to the value produced by `task.fn(*task.args, **task.kwargs)`.

        Notes:
            - Ownership: the produced object is stored on this worker's store, so the
              returned `ObjectRef.owner_node_id` will be `store.node_id`.

        Discussion:
            Q. Which code paths exist?
                1) ActorCreate   -> instantiate the actor locally and return `ActorRef`
                2) ActorCall     -> lookup instance locally and invoke method
                3) Regular call  -> run in-process or via process pool
        """
        ctx = getattr(task, "runtime_env", None)
        ctx_mgr = ctx.apply() if ctx is not None else nullcontext()

        try:
            with ctx_mgr:
                fn = task.fn

                # Actor creation
                if isinstance(fn, dict) and fn.get("kind") == "actor_create":
                    actor_id = new_actor_id()
                    cls = fn["cls"]
                    init_args = tuple(fn.get("args", ()))
                    init_kwargs = dict(fn.get("kwargs", {}) or {})

                    executor = ProcessPoolExecutor(task.max_concurrency or 1, mp_context=mp.get_context("spawn"))
                    self._actor_executors[actor_id] = executor
                    payload = dumps((cls, init_args, init_kwargs))
                    create_future = executor.submit(_create_actor_process, actor_id, payload)
                    result_future = Future()

                    def _on_created(done: Future):
                        try:
                            done.result()
                            result_future.set_result(
                                ActorRef(
                                    actor_id=actor_id,
                                    owner_node_id=self.node_id,
                                )
                            )
                        except Exception as e:
                            result_future.set_exception(e)

                    create_future.add_done_callback(_on_created)
                    return self.store.put_future(result_future, object_id=task_result_object_id(task.task_id))

                # Actor method call
                if isinstance(fn, dict) and fn.get("kind") == "actor_call":
                    actor_id = fn["actor_id"]
                    method_name = fn["method"]
                    executor = self._actor_executors.get(actor_id)
                    if executor is None:
                        raise RuntimeError(f"Actor {actor_id} not found on node {self.node_id}.")

                    payload = dumps((task.args, task.kwargs))
                    future = executor.submit(_call_actor_process, actor_id, method_name, payload)
                    return self.store.put_future(future, object_id=task_result_object_id(task.task_id))

                # Regular function call
                payload = dumps((fn, task.args, task.kwargs))
                max_concurrency = task.max_concurrency or 1
                if max_concurrency in self._task_executors:
                    executor = self._task_executors[max_concurrency]
                else:
                    executor = ThreadPoolExecutor(max_concurrency)
                    self._task_executors[max_concurrency] = executor

                future = executor.submit(_invoke_serialized, payload)
                return self.store.put_future(future, object_id=task_result_object_id(task.task_id))

        except Exception as e:
            raise RuntimeError(f"Task failed in worker@{self.node_id}") from e

    def rpc_read_object_bytes(self, object_id: str) -> bytes:
        """
        Return serialized bytes for a local object.

        Args:
            object_id (str): The local object id.

        Returns:
            bytes: Serialized payload.

        """
        return self.store.get_bytes(object_id)

    def rpc_execute_task(self, task: Task) -> ObjectRef:
        """
        Execute a task on behalf of a remote caller.

        Args:
            task (Task): The remote execution request.

        Returns:
            ObjectRef: Handle to the produced result (owned by this node).
        """
        ref = self.execute_task(task)
        return ObjectRef(
            object_id=ref.object_id,
            owner_node_id=self.store.node_id,
            size_bytes=self.store.get_size(ref.object_id),
        )
