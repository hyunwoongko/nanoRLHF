# RPC

## Overview

One important fact in a distributed runtime is that the execution side (Worker) and the caller side (Session) may not live in the same process. If there are multiple nodes, a Task must travel over the network depending on which node's Worker will run it, and the execution result (the bytes pointed to by an ObjectRef) must also be read back over the network.

The RPC in this project provides only two minimal capabilities.

- Request Task execution on a remote node
- Fetch the bytes for a given object_id from a remote node

In other words, RPC is not part of scheduling policy or execution logic. It is the transport layer that moves Tasks and results between nodes.

## Components

- RemoteWorkerProxy
  - A proxy that the Scheduler can use like a Worker
  - Turns execute_task calls into remote-node RPC calls

- NodeRegistry
  - Stores the mapping node_id -> (address, token)

- Router
  - Decides which node to contact by looking at an ObjectRef
  - For now, it uses ref.owner_node_id as-is

- RpcClient
  - An HTTP JSON client
  - Sends execute_task and get_object requests

- RpcServer
  - An HTTP server running on each node
  - Exposes Worker.rpc_execute_task and Worker.rpc_read_object_bytes externally

## Big Picture

RPC provides one path for execution requests and another path for reading results.

```text
Execution request path

Scheduler
  -> (select a node)
  -> call Worker.execute_task(Task)

If it is a local node
  Scheduler -> Worker.execute_task

If it is a remote node
  Scheduler -> RemoteWorkerProxy.execute_task
           -> RpcClient.execute_task(node_id, task)
           -> (HTTP) RpcServer /rpc/execute_task
           -> Worker.rpc_execute_task(task)
           -> return ObjectRef
```

```text
Result read path

ObjectRef(object_id, owner_node_id)
  -> decide the target node via Router.route_object(ref)
  -> RpcClient.get_object(owner_node_id, object_id)
  -> (HTTP) RpcServer /rpc/get_object
  -> Worker.rpc_read_object_bytes(object_id)
  -> return bytes
```

## RemoteWorkerProxy

The Scheduler tries not to distinguish whether execution is on a local Worker or on a remote node. So it expects something that only needs to provide execute_task, and RemoteWorkerProxy fills that role.

```python
class RemoteWorkerProxy:
    def __init__(self, node_id: str, rpc: RpcClient):
        self.node_id = node_id
        self.rpc = rpc

    def execute_task(self, task: Task) -> ObjectRef:
        return self.rpc.execute_task(self.node_id, task)
```

There is only one core point. From the Scheduler's perspective, a local Worker call and a remote RPC call both look like the same execute_task interface.

## NodeRegistry and Router

NodeRegistry stores how to reach a node. Router decides where a given ref should be sent. Because the roles are separated, you can change routing policy without touching the registry storage structure.

```python
@dataclass
class NodeRegistry:
    _table: Dict[str, Tuple[str, Optional[str]]] = None

    def register(self, node_id: str, address: str, token: Optional[str] = None):
        self._table[node_id] = (address.rstrip("/"), token)

    def get(self, node_id: str) -> Optional[Tuple[str, Optional[str]]]:
        return self._table.get(node_id, None)
```

```python
class Router:
    def __init__(self, registry: NodeRegistry):
        self._registry = registry

    def route_object(self, ref: ObjectRef) -> Optional[str]:
        return ref.owner_node_id
```

Routing is currently simple. Since ObjectRef already carries who owns it, you can think of it as going to that owner node.

## RpcClient

RpcClient is a minimal implementation that sends HTTP POST JSON requests using the standard library. The two important endpoints are:

- /rpc/execute_task
- /rpc/get_object

A Task is not placed directly into JSON. Instead, the serialized bytes are base64-encoded and sent, because an HTTP JSON body cannot contain raw bytes directly.

```python
class RpcClient:
    def execute_task(self, node_id: str, task: Task) -> ObjectRef:
        blob = dumps(task)
        response = self.async_request(
            node_id=node_id,
            path="/rpc/execute_task",
            body={"task_b64": base64.b64encode(blob).decode("ascii")},
        ).result()

        ref = response["ref"]
        return ObjectRef(
            object_id=ref["object_id"],
            owner_node_id=ref["owner_node_id"],
            size_bytes=ref.get("size_bytes"),
        )

    def get_object(self, node_id: str, object_id: str) -> bytes:
        response = self.async_request(
            node_id=node_id,
            path="/rpc/get_object",
            body={"object_id": object_id},
        ).result()

        return base64.b64decode(response["payload_b64"])
```

There are two concepts you need to take away here.

- execute_task returns not the value itself, but an ObjectRef
- get_object returns the serialized bytes for object_id

So the return of remote execution is always a reference, and the value is read via a separate path.

## RpcServer

RpcServer is an HTTP server running on each node. It is a thin adapter that turns incoming external requests into Worker calls.

```text
POST /rpc/execute_task
  input: task_b64
  handling: deserialize task -> worker.rpc_execute_task(task)
  output: return ObjectRef as JSON

POST /rpc/get_object
  input: object_id
  handling: worker.rpc_read_object_bytes(object_id)
  output: return bytes base64-encoded
```

It uses ThreadingMixIn to assign a thread per request. This design is a simple way to get concurrency when RPC handling is I/O-heavy. The server itself does not schedule execution; it only calls the two RPC methods provided by the Worker.

## How Scheduler, Worker, and RPC Connect

The connecting logic in this project can be summarized like this.

- The Scheduler selects a node
- If execution is local, it directly calls Worker.execute_task
- If execution is remote, it calls RemoteWorkerProxy.execute_task
- RemoteWorkerProxy calls the remote RpcServer via RpcClient
- RpcServer calls Worker.rpc_execute_task to delegate execution
- The result comes back as an ObjectRef
- When a value is needed, Router determines owner_node_id and RpcClient.get_object fetches the bytes

## RPC: 3-line Summary

- RPC is the minimal transport layer that provides only Task execution requests and result-byte reads
- NodeRegistry handles connection info, and Router decides the destination based on refs
- RpcClient sends requests, and RpcServer converts requests into Worker calls