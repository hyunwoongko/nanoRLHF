import base64
import json
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Dict, Any, Optional
from urllib import request, error

from nanorlhf.nanoray.core.object_ref import ObjectRef
from nanorlhf.nanoray.core.serialization import dumps
from nanorlhf.nanoray.core.task import Task
from nanorlhf.nanoray.network.router import NodeRegistry


class RpcClient:
    """
    Minimal HTTP JSON RPC client.

    Args:
        registry (NodeRegistry): node_id -> (address, token)
        timeout_s (float): per-request timeout.
        retries (int): number of total attempts (>= 1).
    """

    def __init__(self, registry: NodeRegistry, timeout_s: float = 10.0, retries: int = 3):
        self._reg = registry
        self._timeout = float(timeout_s)
        self._retries = max(1, int(retries))
        self._executor = ThreadPoolExecutor()

    def _request(self, node_id: str, path: str, body: Dict[str, Any]) -> Dict[str, Any]:
        """
        Internal: send a JSON-RPC request to a node.

        Args:
            node_id (str): target node ID
            path (str): URL path (e.g., "/rpc/get_object")
            body (Dict[str, Any]): JSON-serializable request body

        Returns:
            Dict[str, Any]: JSON-deserialized response body
        """
        address, token = self._reg.get(node_id)
        url = f"{address}{path}"
        data = json.dumps(body).encode("utf-8")

        req = request.Request(
            url=url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Content-Length": f"{len(data)}",
            },
        )
        if token:
            req.add_header("Authorization", f"Bearer {token}")

        last_exc: Optional[Exception] = None

        for _ in range(self._retries):
            try:
                with request.urlopen(req, timeout=self._timeout) as resp:
                    raw = resp.read()
                    return json.loads(raw)
            except error.HTTPError as e:
                try:
                    # Try to parse JSON error body returned by server
                    body_txt = e.read().decode("utf-8", errors="replace")
                    return json.loads(body_txt)
                except Exception:
                    last_exc = RuntimeError(f"{e} (no JSON body)")
                    continue
            except Exception as e:
                last_exc = e
                continue

        raise RuntimeError(f"RPC request failed to {url}: {last_exc}")

    def _request_async(self, node_id: str, path: str, body: Dict[str, Any]) -> Future:
        """
        Fire-and-forget wrapper that executes `_request` in a thread pool.
        """
        return self._executor.submit(self._request, node_id, path, body)

    def get_object(self, node_id: str, object_id: str) -> bytes:
        """
        Fetch serialized object bytes from a remote node.

        Args:
            node_id (str): target node ID
            object_id (str): target object ID

        Returns:
            bytes: Raw object bytes
        """
        res = self._request_async(
            node_id=node_id,
            path="/rpc/get_object",
            body={"object_id": object_id},
        ).result()

        if not res.get("ok"):
            err = res.get("error", {})
            msg = err.get("message", err)
            tb = err.get("traceback", "")
            raise RuntimeError(f"Remote get_object failed: {msg}\n{tb}")
        return base64.b64decode(res["payload_b64"])

    def execute_task(self, node_id: str, task: Task) -> ObjectRef:
        """
        Send a task execution request to a remote node.

        Args:
            node_id (str): target node ID
            task (Dict[str, Any]): Task dictionary

        Returns:
            Dict[str, Any]: Task execution result
        """
        blob = dumps(task)

        res = self._request_async(
            node_id=node_id,
            path="/rpc/execute_task",
            body={"task_b64": base64.b64encode(blob).decode("ascii")},
        ).result()

        if not res.get("ok"):
            err = res.get("error", {})
            msg = err.get("message", err)
            tb = err.get("traceback", "")
            raise RuntimeError(f"Remote execute_task failed: {msg}\n--- Remote Traceback ---\n{tb}")

        ref_info = res["ref"]
        return ObjectRef(
            object_id=ref_info["object_id"],
            owner_node_id=ref_info["owner_node_id"],
            size_bytes=ref_info.get("size_bytes"),
        )
