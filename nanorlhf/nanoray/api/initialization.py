import logging
import platform
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from nanorlhf.nanoray.api.session import Session, init_session
from nanorlhf.nanoray.core.object_store import ObjectStore
from nanorlhf.nanoray.network.proxy import RemoteWorkerProxy
from nanorlhf.nanoray.network.router import NodeRegistry, Router
from nanorlhf.nanoray.network.rpc_client import RpcClient
from nanorlhf.nanoray.network.rpc_server import RpcServer
from nanorlhf.nanoray.runtime.worker import Worker
from nanorlhf.nanoray.scheduler.policies import SchedulingPolicy, RoundRobin
from nanorlhf.nanoray.scheduler.scheduler import WorkerLike

logger = logging.getLogger(__name__)
NANORAY_BASE_PORT = 9200


@dataclass(frozen=True)
class NodeConfig:
    """
    Declarative spec for a node used by `init()`.

    Attributes:
        cpus (float): Advertised CPU capacity for scheduling.
        gpus (float): Advertised GPU capacity for scheduling.
        resources (Optional[Dict[str, float]]): Custom named resources.

        # Local node options
        rpc (bool): If True, start an RpcServer node.
        host (str): Bind host for RpcServer.
        port (Optional[int]): Bind port for RpcServer.
        token (Optional[str]): Bearer token for auth.
    """
    cpus: float = 1.0
    gpus: float = 0.0
    resources: Optional[Dict[str, float]] = None

    rpc: bool = False
    host: str = "127.0.0.1"
    port: Optional[int] = NANORAY_BASE_PORT
    token: Optional[str] = None


# Keep server handles so `shutdown()` can close them.
SERVERS: Dict[str, RpcServer] = {}
SERVER_THREADS: Dict[str, threading.Thread] = {}


def init(
    nodes: Optional[Dict[str, NodeConfig]] = None,
    *,
    policy: Optional[SchedulingPolicy] = None,
    default_node_id: Optional[str] = None,
):
    """
    Initialize a nanoray runtime (session + scheduler + optional RPC servers).

    This mirrors Ray's `ray.init()` spirit: if a node is marked local, we create a
    local `Worker` (and optionally start an `RpcServer`); if a node has an `address`,
    we register it and use a `RemoteWorkerProxy` to execute there.

    If `nodes` is None (zero-arg), we create a **single local node "Local"**,
    start an RPC server on an **ephemeral port**, and return a ready Session—
    similar to `ray.init()`.

    Args:
        nodes (Dict[str, NodeConfig]): Mapping `node_id -> NodeConfig`.
        policy (Optional[SchedulingPolicy]): Scheduling policy (default RoundRobin).
        default_node_id (Optional[str]): Default node for `Session.put()` and local cache.

    Returns:
        Session: A ready-to-use session with networking wired in.

    Examples:
        >>> # 1) Single local node + RPC server on port 8001
        >>> config = {
        ...   "A": NodeConfig(local=True, start_server=True, host="127.0.0.1", port=8001, token="tA")
        ... }
        >>> sess = init(config, policy=RoundRobin(), default_node_id="A")

        >>> # 2) Local node + one remote node
        >>> config = {
        ...   "A": NodeConfig(local=True,  start_server=True, host="127.0.0.1", port=8001, token="tA"),
        ...   "B": NodeConfig(local=False, address="http://10.0.0.2:8002", remote_token="tB", cpus=2.0),
        ... }
        >>> sess = init(config, default_node_id="A")

    Discussion:
        Q. What exactly does this set up?
            - Builds a `NodeRegistry` of node_id -> (address, token).
            - For local nodes: creates `ObjectStore` + `Worker`, and (optionally)
              starts an `RpcServer` in a background thread, then registers its address.
            - For remote nodes: registers their address and makes a `RemoteWorkerProxy`.
            - Assembles `nodes` for the `Scheduler` (mix of local `Worker` and `RemoteWorkerProxy`),
              creates a `Router` and `RpcClient`, and returns a `Session` with those injected.

        Q. How do I stop everything?
            Call `shutdown()`. It will stop any servers we started and clear globals.
    """
    pol = policy or RoundRobin()
    registry = NodeRegistry()

    # zero-arg default
    if nodes is None:
        nodes = {platform.node(): NodeConfig(rpc=True)}

    logger.info("Initializing nanoray runtime with nodes: %s", nodes)

    # Build WorkerLike map and capacity for Scheduler.
    sched_nodes: Dict[str, Tuple[WorkerLike, Dict[str, Any]]] = {}

    # Keep local workers to wire servers.
    local_workers: Dict[str, Worker] = {}

    # 1) Create local workers and (optionally) start local RPC servers.
    for nid, cfg in nodes.items():
        is_local = cfg.host in ("127.0.0.1", "localhost")
        if is_local:
            store = ObjectStore(nid)
            worker = Worker(store=store)
            local_workers[nid] = worker

            # If asked, start an RpcServer in a background thread and register endpoint.
            if cfg.rpc:
                port = cfg.port if cfg.port is not None else 0
                rpc_server = RpcServer(nid, worker, host=cfg.host, port=port, token=cfg.token)
                thread = threading.Thread(target=rpc_server.start, name=f"rpcd-{nid}", daemon=True)
                thread.start()
                SERVERS[nid] = rpc_server
                SERVER_THREADS[nid] = thread

                # Wait briefly until the HTTP server is bound, then read actual port
                actual_port: Optional[int] = None
                for _ in range(200):  # ~2s total
                    httpd = getattr(rpc_server, "_httpd", None)
                    if httpd is not None:
                        actual_port = int(getattr(httpd, "server_port"))
                        break
                    time.sleep(0.01)
                if actual_port is None:
                    raise RuntimeError(f"Failed to start RpcServer for node {nid}")

                registry.register(nid, f"http://{cfg.host}:{actual_port}", token=cfg.token)

    # 2) Build `WorkerLike` entries (local vs remote) and capacity dicts.
    rpc = RpcClient(registry=registry)
    for nid, cfg in nodes.items():
        is_local = cfg.host in ("127.0.0.1", "localhost")
        cap = {
            "cpus": cfg.cpus,
            "gpus": cfg.gpus,
            "resources": cfg.resources or {},
        }
        if is_local:
            if cfg.rpc:
                worker = RemoteWorkerProxy(node_id=nid, rpc=rpc)
            else:
                worker = local_workers[nid]
        else:
            worker = RemoteWorkerProxy(node_id=nid, rpc=rpc)
        sched_nodes[nid] = (worker, cap)

    # 3) Create Router and Session with networking hooks
    router = Router(registry=registry)
    sess = init_session(
        policy=pol,
        nodes=sched_nodes,
        local_workers=local_workers,
        default_node_id=default_node_id,
    )
    sess.router = router
    sess.rpc = rpc
    return sess


def shutdown():
    """
    Shutdown the nanoray runtime, including RPC servers and local workers.

    This is a best-effort cleanup; any exceptions are caught and ignored.
    1) Cleans up the global session, including placement groups and local workers.
    2) Stops any RPC servers that were started during `init()`.
    """
    logger.info("Shutting down nanoray runtime")

    # 1) placement group / session cleanup (best-effort)
    try:
        from nanorlhf.nanoray.api import session as session_api

        sess = getattr(session_api, "GLOBAL_SESSION", None)
        if sess is not None:
            # release placement group reservations
            for pg_id in list(getattr(sess.scheduler, "placement_groups", {}).keys()):
                try:
                    sess.remove_placement_group(pg_id)
                except Exception:
                    pass

            # shutdown local workers (actors/processes/threadpools)
            for w in (sess.local_workers or {}).values():
                try:
                    if hasattr(w, "shutdown"):
                        w.shutdown()
                except Exception:
                    pass

            session_api.GLOBAL_SESSION = None
    except Exception:
        pass

    # 2) Stop servers (best-effort)
    for nid, srv in list(SERVERS.items()):
        try:
            srv.stop()
        except Exception:
            pass
    SERVERS.clear()
    SERVER_THREADS.clear()