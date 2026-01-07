# initialzation

## Overview

init is the entry point that assembles one full set of the nanoray runtime on the driver. Concretely, it prepares Session, Scheduler, local Workers, networking (RPC), and routing information in one place.
After calling init, users can create remote calls, then use submit and get, and the execution flow continues end to end.

There is one key idea. It aligns the "execution path" so the scheduler can treat local execution and remote execution the same way by looking only at the WorkerLike interface.

## From the driver perspective

The driver is the process where user code runs. The driver does not execute work directly. Instead, it sets up runtime components, then submits Tasks and retrieves values. init creates the runtime handle the driver will use, then returns it as a Session.

```text
driver (user code)
  |
  | call init(...)
  v
Session (driver-side runtime handle)
  |
  | submit / drain / get
  v
Scheduler  ->  WorkerLike.execute_task(...)  ->  ObjectRef
```

## Components created by init

The main components prepared by init are:

- NodeRegistry: a store for node_id -> (address, token). The RPC client uses it to resolve destination addresses.
- RpcClient: a client that sends HTTP JSON RPC requests to a target node_id.
- Router: resolves the destination node based on the owner_node_id pointed to by an ObjectRef.
- Worker or RemoteWorkerProxy: execution endpoints that the Scheduler calls. Both behave like WorkerLike.
- Session and Scheduler: the center that lets the driver submit Tasks, drive the queue, and turn ObjectRef into values.

## Node configuration model

NodeConfig contains a node's "advertised capacity" and "communication method (RPC)".

- cpus, gpus, resources: capacity information the scheduler uses for placement decisions.
- rpc, host, port, token: settings for whether to start an RPC server on a local node, which address to bind, and whether to use an auth token.

With this project's simplified rule, locality is determined by whether host is "127.0.0.1" or "localhost". This is simplified by address for a teaching runtime.

## Local node initialization

If a node is considered local, init prepares the following for that node:

1) Create an ObjectStore
2) Create a Worker (owns that ObjectStore)
3) If cfg.rpc is True, start an RpcServer in a separate thread
4) After confirming the port actually bound by RpcServer, register the address into NodeRegistry

An important point is the design that "even a local node can use the RPC path". If cfg.rpc is True, the scheduler does not execute that local node by calling the Worker directly. Instead, it uses RemoteWorkerProxy and calls via RPC. This unifies local execution and remote execution under the same protocol, and from a teaching perspective the execution path becomes more consistent.

```text
local node (cfg.rpc = False)
  Scheduler -> Worker.execute_task(...)  (direct call)

local node (cfg.rpc = True)
  Scheduler -> RemoteWorkerProxy.execute_task(...)
           -> RpcClient -> RpcServer -> Worker.rpc_execute_task(...)
```

## Remote node initialization

A remote node does not create a local Worker. From the scheduler perspective, that node is represented by a single RemoteWorkerProxy.

- RemoteWorkerProxy.execute_task(Task)
  - calls RpcClient.execute_task(node_id, task)
  - the remote RpcServer receives it and runs Worker.rpc_execute_task
  - returns ObjectRef as the response

The remote node address must be in NodeRegistry. In this code, only local nodes directly start servers and call registry.register. If remote node addresses are needed, an additional flow must register them into NodeRegistry from outside. The current init implementation is simplified for teaching and only starts and registers servers based on whether host is local.

## Building the scheduler input (nodes)

init builds the nodes map passed to the Scheduler.

- key: node_id
- value: (WorkerLike, capacity_dict)

capacity_dict is the data the scheduler uses for placement decisions.

```text
capacity_dict example
  cpus: 4.0
  gpus: 1.0
  resources: {"ram_gb": 64.0}
```

The object placed in the WorkerLike slot is one of two:

- Worker for local direct execution
- RemoteWorkerProxy for execution via the RPC path

Because of this, the Scheduler does not need to know "how execution happens". After choosing a candidate node, the Scheduler only needs to call worker_like.execute_task(task).

## Session wiring

init calls init_session to create the global session (Session), then injects the following:

- local_workers: the list of local Workers. Used for Session.get local lookup, and for selecting the default store used by put.
- default_node_id: used when choosing the default node for put and the local cache.
- router, rpc: used for the remote get path. It decides the target node using ObjectRef.owner_node_id, fetches bytes over RPC, then stores them into the local cache.

In other words, init wires both "scheduling" and "remote access" hooks at once so the driver can use submit, drain, and get.

## Default single-node mode

If nodes is not provided, init does the following:

- uses platform.node() as node_id
- creates one local node with NodeConfig(rpc=True)
- if port is None, uses 0 so the OS selects an ephemeral port
- reads the bound port and registers it into the registry

This mode imitates ray.init() "start immediately" UX. Users can experience the same execution model on a single node with no extra configuration.

## shutdown overview

shutdown cleans up the global resources created by init.

- clears the global session variable
- attempts to unregister any registered placement groups
- if there are local Workers, attempts to call shutdown
- calls stop on started RpcServers to terminate them

This cleanup is implemented as best-effort. In a teaching runtime, what matters is the "setup" and the "flow", so it is written to keep cleaning up as much as possible even if some steps fail.

## initialization: 3-line Summary

- init prepares the runtime configuration for the driver (Session, Scheduler, execution endpoints, networking hooks) in one step
- the scheduler calls only WorkerLike.execute_task, and local direct execution and RPC execution are abstracted into the same shape
- if cfg.rpc is True, even a local node executes via the RPC path through RemoteWorkerProxy, making the execution model more consistent