import base64
import json
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from typing import Any, Dict, Optional

from nanorlhf.nanoray.core.serialization import loads
from nanorlhf.nanoray.core.task import Task
from nanorlhf.nanoray.runtime.worker import Worker


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    """
    Handle requests in a separate thread.

    Discussion:
        Q. What is `ThreadingMixIn`?
            It is a standard library mix-in that spawns a new thread for each
            incoming request, allowing concurrent handling. This is suitable
            for I/O-bound tasks typical in RPC servers.

        Q. Why not use `ProcessMixIn`?
            `ProcessMixIn` creates a new process per request, which is heavier
            and incurs more overhead. Moreover, CUDA and multiprocess forking
            do not play well together, making `ProcessMixIn` less suitable for

        Q. Why `daemon_threads = True`?
            Setting `daemon_threads` to True ensures that threads will not
            prevent the server from shutting down. When the main program
            exits, all daemon threads are terminated automatically.
    """
    # auto-terminate threads on server shutdown
    daemon_threads = True


class RpcServer:
    """
    Tiny HTTP JSON RPC server

    Args:
        node_id (str): This node's identity.
        worker (Worker): Executes tasks and reads local objects.
        host (str): Bind address (default "0.0.0.0").
        port (int): TCP port to listen on.
        token (Optional[str]): If set, requires `Authorization: Bearer <token>`.

    Examples:
        >>> server = RpcServer("A", worker, port=8001)
        >>> server.start()
        >>> # ... do something
        >>> server.stop()

    Discussion:
        Q. What does this server expose?
            POST /rpc/get_object:
                Request: {"object_id": str}
                Response: {"ok": true, "payload_b64": "..."}

            POST /rpc/execute_task:
                Request: {"task_b64": base64-encoded-bytes}
                Response: {"ok": true, "ref": {"object_id": "...", "owner_node_id": "...", "size_bytes": null}}

            When errors occur:
                Response: {"ok": false, "error": {"type":"...", "message":"...", "traceback":"..."}}
    """

    def __init__(
        self,
        node_id: str,
        worker: Worker,
        host: str = "0.0.0.0",
        port: int = 8000,
        token: Optional[str] = None
    ):
        self.node_id = node_id
        self.worker = worker
        self.host = host
        self.port = int(port)
        self._token = token
        self._httpd: Optional[ThreadingHTTPServer] = None

    def start(self):
        """
        Start the RPC server.

        Discussion:
            Q. What is `serve_forever(poll_interval=0.2)` function?
                It starts handling incoming requests in an infinite loop,
                checking for shutdown signals every 0.2 seconds. This allows
                the server to respond to shut down requests promptly.
        """
        handler_class = self._make_handler()
        self._httpd = ThreadingHTTPServer((self.host, self.port), handler_class)
        self._httpd.worker = self.worker
        self._httpd.node_id = self.node_id
        self._httpd.token = self._token
        self._httpd.serve_forever(poll_interval=0.2)

    def stop(self):
        """
        Stop the RPC server.
        """
        if self._httpd:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None

    def _make_handler(self):
        class Handler(BaseHTTPRequestHandler):
            """
            HTTP request handler for nanoray RPC.

            Attributes:
                server.worker (Worker): Local executor + object store adapter used by RPC endpoints.
                server.node_id (str): Logical id of this node (e.g., "A").
                server.token (Optional[str]): Bearer token; if set, requests must include
                    `Authorization: Bearer <token>`.

            Discussion:
                Q. What is this class?
                    A new `Handler` instance is created per request. With `ThreadingMixIn`,
                    each request runs in its own thread.

                Q. Why subclass `BaseHTTPRequestHandler`?
                    It provides a simple way to handle HTTP requests by overriding methods
                    like `do_GET`, `do_POST`, etc. Here, we only need `do_POST` for our RPC calls.

                Q. What does `self.server` refer to?
                    It refers to the instance of the HTTP server (`_ThreadingHTTPServer`)
                    that is handling the requests. We attached runtime objects (worker, node_id,
                    token) onto the instance; handlers access them via `self.server.*`.

                Q. What is the request lifecycle?
                    1) HTTP server accepts a connection and instantiates `Handler`.
                    2) `do_POST()` runs: auth → body parse → path dispatch → call worker.
                    3) The worker executes (`rpc_read_object_bytes` / `rpc_execute_task`).
                    4) We write a JSON response; keep-alive is handled by the stdlib.

                Q. How is the request body read?
                    We trust `Content-Length`, read exactly that many bytes, decode as UTF-8,
                    then `json.loads`. Chunked transfer is not implemented.

                Q. Why JSON + base64 instead of raw sockets or gRPC?
                    Because they are pure standard library, easy debugging,
                    headers help with auth/proxies. We can swap to gRPC/Protobuf later if desired;

                Q. Any security considerations?
                    - Bearer token only currently. For production, prefer mTLS/signed tokens.
                    - Deserialization trust: `execute_task` can run arbitrary code;
                        restrict access to trusted peers inside the cluster.
                    - Validate inputs and enforce length limits/rate-limits in real deployments.

                Q. What error model do clients see?
                    - 401 Unauthorized on auth failures
                    - 404 Not Found for unknown paths
                    - 500 Internal Server Error with:
                      `{"ok": false, "error": {"type","message","traceback"}}` (debug-friendly)

                Q. Performance caveats?
                    Base64 adds ~33% size overhead. For large payloads,
                    We can consider chunked/streaming transfers or
                    external object stores (e.g., presigned S3 URLs).
            """

            def log_message(self, format: str, *args: Any):
                """
                Override to suppress default logging.

                Discussion:
                    Q. Why override `log_message`?
                        The default implementation logs every request to stderr,
                        which can clutter output in high-throughput scenarios.
                        Overriding it to a no-op silences these logs.
                """

            # ------ Helper methods ------
            def _json(self) -> Dict[str, Any]:
                """
                Parse JSON body from the request.

                Returns:
                    Dict[str, Any]: Parsed JSON object.
                """
                length = int(self.headers.get("Content-Length", "0"))
                if length:
                    data = self.rfile.read(length)
                else:
                    raise ValueError("Missing or invalid Content-Length header")
                return json.loads(data.decode("utf-8"))

            def _auth_ok(self) -> bool:
                """
                Check if the request is authorized.

                Returns:
                    bool: True if authorized, False otherwise.
                """
                token = getattr(self.server, "token", None)
                if not token:
                    return True
                auth = self.headers.get("Authorization", "")
                return auth == f"Bearer {token}"

            def _send(self, code: int, obj: Dict[str, Any]):
                """
                Send a JSON response to the client.

                Args:
                    code (int): HTTP status code.
                    obj (Dict[str, Any]): JSON-serializable response object.
                """
                payload = json.dumps(obj).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            # ------ do_POST uses the above helper methods ------
            def do_POST(self):  # noqa
                """
                Handle POST requests for RPC endpoints.
                """
                if not self._auth_ok():
                    self._send(
                        HTTPStatus.UNAUTHORIZED, {
                            "ok": False,
                            "error": {
                                "type": "AuthError",
                                "message": "unauthorized",
                            }
                        }
                    )
                    return

                try:
                    if self.path == "/rpc/get_object":
                        """
                        Request: {"object_id": str}
                        Response: {"ok": true, "payload_b64": "..."}
                        """
                        body = self._json()
                        object_id = body.get("object_id")
                        if not object_id:
                            self._send(
                                HTTPStatus.BAD_REQUEST,
                                {"ok": False, "error": {"type": "BadRequest", "message": "missing 'object_id'"}}
                            )
                            return
                        try:
                            payload = self.server.worker.rpc_read_object_bytes(str(object_id))  # noqa
                        except KeyError:
                            self._send(
                                HTTPStatus.NOT_FOUND,
                                {"ok": False, "error": {"type": "NotFound", "message": f"object not found: {object_id}"}}
                            )
                            return
                        except Exception as e:
                            self._send(
                                HTTPStatus.INTERNAL_SERVER_ERROR,
                                {"ok": False, "error": {"type": type(e).__name__, "message": str(e), "traceback": traceback.format_exc()}}
                            )
                            return

                        b64 = base64.b64encode(payload).decode("ascii")
                        # because base64 only uses ASCII chars, so don't need to decode as UTF-8

                        self._send(
                            HTTPStatus.OK, {
                                "ok": True,
                                "payload_b64": b64,
                            }
                        )
                        return

                    elif self.path == "/rpc/execute_task":
                        """
                        Request: {"task_b64": base64-encoded-bytes}
                        Response: {"ok": true, "ref": {"object_id": "...", "owner_node_id": "...", "size_bytes": null}}
                        """
                        body = self._json()
                        task_b64 = body.get("task_b64")
                        if not task_b64:
                            self._send(
                                HTTPStatus.BAD_REQUEST,
                                {"ok": False, "error": {"type": "BadRequest", "message": "missing 'task_b64'"}}
                            )
                            return
                        try:
                            blob = base64.b64decode(task_b64.encode("ascii"))
                        except Exception as e:
                            self._send(
                                HTTPStatus.BAD_REQUEST,
                                {"ok": False, "error": {"type": "BadRequest", "message": "invalid base64", "traceback": traceback.format_exc()}}
                            )
                            return
                        try:
                            task: Task = loads(blob)
                        except Exception as e:
                            self._send(
                                HTTPStatus.BAD_REQUEST,
                                {"ok": False, "error": {"type": "BadRequest", "message": "invalid task payload", "traceback": traceback.format_exc()}}
                            )
                            return
                        try:
                            ref = self.server.worker.rpc_execute_task(task)  # noqa
                        except Exception as e:
                            self._send(
                                HTTPStatus.INTERNAL_SERVER_ERROR,
                                {"ok": False, "error": {"type": type(e).__name__, "message": str(e), "traceback": traceback.format_exc()}}
                            )
                            return

                        self._send(
                            HTTPStatus.OK, {
                                "ok": True,
                                "ref": {
                                    "object_id": ref.object_id,
                                    "owner_node_id": ref.owner_node_id,
                                    "size_bytes": ref.size_bytes,
                                }
                            }
                        )
                        return

                    self._send(
                        HTTPStatus.NOT_FOUND, {
                            "ok": False,
                            "error": {
                                "type": "NotFound",
                                "message": f"path {self.path!r} not found"
                            }
                        }
                    )

                except Exception as e:
                    tb = traceback.format_exc()
                    self._send(
                        HTTPStatus.INTERNAL_SERVER_ERROR, {
                            "ok": False,
                            "error": {
                                "type": type(e).__name__,
                                "message": str(e),
                                "traceback": tb,
                            }
                        }
                    )

        return Handler
