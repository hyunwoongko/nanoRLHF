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
        self.registry = registry
        self.timeout = float(timeout_s)
        self.retries = max(1, int(retries))
        self.executor = ThreadPoolExecutor()

    def request(self, node_id: str, path: str, body: Dict[str, Any]) -> Dict[str, Any]:
        """
        Internal: send a JSON-RPC request to a node.

        Args:
            node_id (str): target node ID
            path (str): URL path (e.g., "/rpc/get_object")
            body (Dict[str, Any]): JSON-serializable request body

        Returns:
            Dict[str, Any]: JSON-deserialized response body
        """
        address, token = self.registry.get(node_id)
        url = f"{address}{path}"
        data = json.dumps(body).encode("utf-8")

        _request = request.Request(
            url=url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Content-Length": f"{len(data)}",
            },
        )
        if token:
            _request.add_header("Authorization", f"Bearer {token}")

        last_exception: Optional[Exception] = None

        for _ in range(self.retries):
            try:
                with request.urlopen(_request, timeout=self.timeout) as response:
                    raw = response.read()
                    return json.loads(raw)
            except error.HTTPError as e:
                try:
                    # Try to parse JSON error body returned by server
                    body_txt = e.read().decode("utf-8", errors="replace")
                    return json.loads(body_txt)
                except Exception:
                    last_exception = RuntimeError(f"{e} (no JSON body)")
                    continue
            except Exception as e:
                last_exception = e
                continue

        raise RuntimeError(f"RPC request failed to {url}: {last_exception}")

    def async_request(self, node_id: str, path: str, body: Dict[str, Any]) -> Future:
        """
        Fire-and-forget wrapper that executes `_request` in a thread pool.
        """
        return self.executor.submit(self.request, node_id, path, body)

    def get_object(self, node_id: str, object_id: str) -> bytes:
        """
        Fetch serialized object bytes from a remote node.

        Args:
            node_id (str): target node ID
            object_id (str): target object ID

        Returns:
            bytes: Raw object bytes
        """
        response = self.async_request(
            node_id=node_id,
            path="/rpc/get_object",
            body={"object_id": object_id},
        ).result()

        if not response.get("ok"):
            _error = response.get("error", {})
            message = _error.get("message", _error)
            traceback = _error.get("traceback", "")
            raise RuntimeError(f"Remote get_object failed: {message}\n{traceback}")
        return base64.b64decode(response["payload_b64"])

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

        response = self.async_request(
            node_id=node_id,
            path="/rpc/execute_task",
            body={"task_b64": base64.b64encode(blob).decode("ascii")},
        ).result()

        if not response.get("ok"):
            _error = response.get("error", {})
            message = _error.get("message", _error)
            traceback = _error.get("traceback", "")
            raise RuntimeError(f"Remote execute_task failed: {message}\n--- Remote Traceback ---\n{traceback}")

        ref = response["ref"]
        return ObjectRef(
            object_id=ref["object_id"],
            owner_node_id=ref["owner_node_id"],
            size_bytes=ref.get("size_bytes"),
        )
