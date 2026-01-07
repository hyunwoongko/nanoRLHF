from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from typing import Any, Dict, Optional

from nanorlhf.nanoray.api.remote import ActorRef
from nanorlhf.nanoray.core.object_ref import ObjectRef
from nanorlhf.nanoray.core.object_store import ObjectStore
from nanorlhf.nanoray.core.serialization import dumps, loads
from nanorlhf.nanoray.core.task import Task
from nanorlhf.nanoray.runtime.actor import ActorRuntime
from nanorlhf.nanoray.utils import task_result_object_id


def _invoke(payload: bytes):
    fn, args, kwargs = loads(payload)
    return fn(*args, **(kwargs or {}))


class Worker:
    def __init__(self, store: ObjectStore, node_id: Optional[str] = None):
        self.store = store
        self.node_id = node_id or store.node_id
        self.task_executors: Dict[int, ThreadPoolExecutor] = {}
        self.actors = ActorRuntime(node_id=self.node_id)

    def execute_task(self, task: Task) -> ObjectRef:
        """
        Execute a task locally.

        Args:
            task (Task): The task to execute.
        """
        ctx = getattr(task, "runtime_env", None)
        ctx_mgr = ctx.apply() if ctx is not None else nullcontext()

        with ctx_mgr:
            fn = task.fn

            if isinstance(fn, dict) and fn.get("kind") == "actor_create":
                return self.execute_actor_create(task, fn)

            if isinstance(fn, dict) and fn.get("kind") == "actor_call":
                return self.execute_actor_call(task, fn)

            return self.execute_task_call(task)

    def execute_actor_create(self, task: Task, fn: Dict[str, Any]) -> ObjectRef:
        """
        Execute an actor creation task.

        Args:
            task (Task): The task representing the actor creation.
            fn (Dict[str, Any]): The function descriptor containing actor class and init args.

        Returns:
            ObjectRef: A reference to the created actor.
        """
        actor_id, ready_future = self.actors.create(
            cls=fn["cls"],
            init_args=tuple(fn.get("args", ())),
            init_kwargs=dict(fn.get("kwargs", {}) or {}),
            max_concurrency=max(int(task.max_concurrency or 1), 1),
        )

        future = Future()

        def finish(done: Future):
            done.result()
            future.set_result(ActorRef(actor_id=actor_id, owner_node_id=self.node_id))

        ready_future.add_done_callback(finish)
        return self.store.put_future(future, object_id=task_result_object_id(task.task_id))

    def execute_actor_call(self, task: Task, fn: Dict[str, Any]) -> ObjectRef:
        """
        Execute an actor method call task.

        Args:
            task (Task): The task representing the actor method call.
            fn (Dict[str, Any]): The function descriptor containing actor_id and method name.

        Returns:
            ObjectRef: A reference to the result of the actor method call.
        """
        future = self.actors.call(
            actor_id=fn["actor_id"],
            call_id=task.task_id,
            method_name=fn["method"],
            args=task.args,
            kwargs=task.kwargs or {},
            max_concurrency=max(int(task.max_concurrency or 1), 1),
        )
        return self.store.put_future(future, object_id=task_result_object_id(task.task_id))

    def execute_task_call(self, task: Task) -> ObjectRef:
        """
        Execute a regular task call.

        Args:
            task (Task): The task to execute.

        Returns:
            ObjectRef: A reference to the result of the task execution.
        """
        payload = dumps((task.fn, task.args, task.kwargs))
        max_concurrency = max(int(task.max_concurrency or 1), 1)

        executor = self.task_executors.get(max_concurrency)
        if executor is None:
            executor = ThreadPoolExecutor(max_workers=max_concurrency)
            self.task_executors[max_concurrency] = executor

        future = executor.submit(_invoke, payload)
        return self.store.put_future(future, object_id=task_result_object_id(task.task_id))

    def rpc_read_object_bytes(self, object_id: str) -> bytes:
        """
        Fetch serialized object bytes for RPC.

        Args:
            object_id (str): The ID of the object to fetch.

        Returns:
            bytes: Raw object bytes.
        """
        return self.store.get_bytes(object_id)

    def rpc_execute_task(self, task: Task) -> ObjectRef:
        """
        Execute a task via RPC.

        Args:
            task (Task): The task to execute.

        Returns:
            ObjectRef: A reference to the result of the task execution.
        """
        ref = self.execute_task(task)
        return ObjectRef(
            object_id=ref.object_id,
            owner_node_id=self.store.node_id,
            size_bytes=self.store.get_size(ref.object_id),
        )

    def shutdown(self):
        """
        Shutdown the worker and its resources.
        """
        self.actors.shutdown()
        for _, executor in list(self.task_executors.items()):
            executor.shutdown(wait=False)
        self.task_executors.clear()
