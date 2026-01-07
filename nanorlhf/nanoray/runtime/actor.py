import multiprocessing as mp
import threading
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

import setproctitle
import torch
import torch.distributed as dist

from nanorlhf.nanoray.core.serialization import dumps, loads
from nanorlhf.nanoray.runtime.messages import (
    CallRequest,
    ResizeRequest,
    ShutdownRequest,
    CreatedResponse,
    ResizedResponse,
    ResultResponse,
    ShutdownDoneResponse,
)
from nanorlhf.nanoray.utils import new_actor_id


def actor_main_process(
    actor_id: str,
    create_payload: bytes,
    request_queue,
    response_queue,
    initial_max_concurrency: int,
):
    """
    Main process function for an actor.

    Args:
        actor_id (str): Unique identifier for the actor.
        create_payload (bytes): Serialized payload to create the actor instance.
        request_queue: Queue to receive requests.
        response_queue: Queue to send responses.
        initial_max_concurrency (int): Initial maximum concurrency level.
    """
    cls, init_args, init_kwargs = loads(create_payload)
    setproctitle.setproctitle(f"nanoray:{cls.__name__}")
    instance = cls(*init_args, **(init_kwargs or {}))

    device = torch.cuda.current_device() if torch.cuda.is_available() else None
    is_dist_initialized = dist.is_available() and dist.is_initialized()

    max_concurrency = max(int(initial_max_concurrency or 1), 1)
    parallel_thread_pool = ThreadPoolExecutor(max_workers=max_concurrency)
    serial_thread_pool = ThreadPoolExecutor(max_workers=1)
    lock = threading.Lock()

    created_response = CreatedResponse(actor_id=actor_id, max_concurrency=max_concurrency)
    response_queue.put(created_response)

    def submit(request: CallRequest):  # noqa
        def run():
            nonlocal is_dist_initialized
            is_dist_initialized = dist.is_available() and dist.is_initialized()

            if is_dist_initialized and device is not None:
                torch.cuda.set_device(device)

            args, kwargs = loads(request.payload)
            method = getattr(instance, request.method_name, None)
            if method is None or not callable(method):
                raise AttributeError(f"Actor method not found or not callable: {request.method_name}")
            return method(*args, **(kwargs or {}))

        with lock:
            selected = serial_thread_pool if is_dist_initialized else parallel_thread_pool
            future = selected.submit(run)

        future.add_done_callback(lambda done, call_id=request.call_id: send(call_id, done))

    def send(call_id: str, future: Future):
        try:
            value = future.result()
            response_queue.put(
                ResultResponse(
                    actor_id=actor_id,
                    call_id=call_id,
                    ok=True,
                    value_payload=dumps(value),
                    error_payload=None,
                )
            )
        except Exception as e:
            tb = traceback.format_exc()
            response_queue.put(
                ResultResponse(
                    actor_id=actor_id,
                    call_id=call_id,
                    ok=False,
                    value_payload=None,
                    error_payload=dumps((type(e).__name__, str(e), tb)),
                )
            )

    while True:
        # listen for requests
        request = request_queue.get()

        if isinstance(request, ShutdownRequest):
            break

        elif isinstance(request, ResizeRequest):
            requested = max(int(request.new_max_concurrency or 1), 1)
            with lock:
                if requested > max_concurrency:
                    old_pool = parallel_thread_pool
                    parallel_thread_pool = ThreadPoolExecutor(max_workers=requested)
                    max_concurrency = requested
                    old_pool.shutdown(wait=False)
            response_queue.put(ResizedResponse(actor_id=actor_id, max_concurrency=max_concurrency))

        elif isinstance(request, CallRequest):
            submit(request)

        else:
            raise RuntimeError(f"Unknown request type: {type(request)}")

    with lock:
        try:
            parallel_thread_pool.shutdown(wait=True)
        finally:
            serial_thread_pool.shutdown(wait=True)

    shutdown_done_response = ShutdownDoneResponse(actor_id=actor_id)
    response_queue.put(shutdown_done_response)


@dataclass
class ActorHandle:
    actor_id: str
    node_id: str
    request_q: Any
    response_q: Any
    process: mp.Process
    max_concurrency: int

    created_future: Future = field(default_factory=Future)
    pending: Dict[str, Future] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)
    listener_thread: threading.Thread | None = None
    is_closed: bool = False

    def start(self):
        """
        Start the actor process and the listener thread.
        """
        self.process.start()
        self.listener_thread = threading.Thread(target=self.listen_loop, daemon=True)
        self.listener_thread.start()

    def listen_loop(self):
        """
        Listen for responses from the actor process and handle them accordingly.
        """
        while True:
            response = self.response_q.get()

            if isinstance(response, CreatedResponse):
                with self.lock:
                    self.max_concurrency = int(response.max_concurrency or 1)
                if not self.created_future.done():
                    self.created_future.set_result(True)
            elif isinstance(response, ResizedResponse):
                with self.lock:
                    self.max_concurrency = int(response.max_concurrency or 1)
            elif isinstance(response, ResultResponse):
                with self.lock:
                    future = self.pending.pop(response.call_id, None)

                if future is None:
                    continue

                if response.ok:
                    future.set_result(loads(response.value_payload))
                else:
                    exc_name, exc_msg, tb = loads(response.error_payload)
                    future.set_exception(
                        RuntimeError(
                            f"Actor call failed (actor_id={response.actor_id}, call_id={response.call_id}) "
                            f"exc={exc_name}: {exc_msg}\n{tb}"
                        )
                    )
            elif isinstance(response, ShutdownDoneResponse):
                break
            else:
                raise RuntimeError(f"Unknown response type: {type(response)}")

    def submit(
        self,
        call_id: str,
        method_name: str,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
        max_concurrency: int,
    ) -> Future:
        """
        Submit a method call to the actor.

        Args:
            call_id (str): Unique identifier for the call.
            method_name (str): Name of the method to invoke.
            args (Tuple[Any, ...]): Positional arguments for the method.
            kwargs (Dict[str, Any]): Keyword arguments for the method.
            max_concurrency (int): Desired maximum concurrency level.

        Returns:
            Future: A future representing the result of the method call.
        """
        if self.is_closed:
            raise RuntimeError(f"Actor {self.actor_id} is closed.")

        desired = max(int(max_concurrency or 1), 1)
        if desired > self.max_concurrency:
            self.request_q.put(ResizeRequest(new_max_concurrency=desired))

        future = Future()
        with self.lock:
            self.pending[call_id] = future

        payload = dumps((args, kwargs))
        self.request_q.put(CallRequest(call_id=call_id, method_name=method_name, payload=payload))
        return future

    def shutdown(self) -> None:
        if self.is_closed:
            return
        self.is_closed = True
        self.request_q.put(ShutdownRequest())
        if self.process.is_alive():
            self.process.join(timeout=1.0)


class ActorRuntime:
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.actors: Dict[str, ActorHandle] = {}

    def create(
        self,
        cls,
        init_args: Tuple[Any, ...],
        init_kwargs: Dict[str, Any],
        max_concurrency: int,
    ) -> Tuple[str, Future]:
        """
        Create a new actor instance.

        Args:
            cls: The actor class to instantiate.
            init_args (Tuple[Any, ...]): Positional arguments for the actor's constructor.
            init_kwargs (Dict[str, Any]): Keyword arguments for the actor's constructor.
            max_concurrency (int): Maximum concurrency level for the actor.
        """
        actor_id = new_actor_id()
        max_concurrency = max(int(max_concurrency or 1), 1)

        mp_ctx = mp.get_context("spawn")
        request_q = mp_ctx.Queue()
        response_q = mp_ctx.Queue()

        create_payload = dumps((cls, init_args, init_kwargs))
        proc = mp_ctx.Process(
            target=actor_main_process,
            args=(actor_id, create_payload, request_q, response_q, max_concurrency),
            daemon=True,
        )

        handle = ActorHandle(
            actor_id=actor_id,
            node_id=self.node_id,
            request_q=request_q,
            response_q=response_q,
            process=proc,
            max_concurrency=max_concurrency,
        )
        self.actors[actor_id] = handle
        handle.start()
        ready = Future()

        def finish_created(done: Future):
            done.result()
            ready.set_result(True)

        handle.created_future.add_done_callback(finish_created)
        return actor_id, ready

    def call(
        self,
        actor_id: str,
        call_id: str,
        method_name: str,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
        max_concurrency: int,
    ) -> Future:
        """
        Call a method on the specified actor.

        Args:
            actor_id (str): Unique identifier for the actor.
            call_id (str): Unique identifier for the call.
            method_name (str): Name of the method to invoke.
            args (Tuple[Any, ...]): Positional arguments for the method.
            kwargs (Dict[str, Any]): Keyword arguments for the method.
            max_concurrency (int): Desired maximum concurrency level.
        """
        handle = self.actors.get(actor_id)
        if handle is None:
            raise RuntimeError(f"Actor {actor_id} not found on node {self.node_id}.")
        return handle.submit(call_id, method_name, args, kwargs, max_concurrency=max_concurrency)

    def shutdown(self) -> None:
        """
        Shutdown all actors managed by this runtime.
        """
        for _, handle in list(self.actors.items()):
            handle.shutdown()
        self.actors.clear()
