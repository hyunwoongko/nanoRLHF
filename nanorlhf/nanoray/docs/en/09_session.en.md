# Session

## Overview

Session is the object that represents the runtime on the driver side. The entire flow where a user creates remote calls, submits them, and gets results goes through Session. The core role of Session is not to execute directly, but to drive execution via the Scheduler and to collect results as a driver coordinator.

Session has three responsibilities.

- Submit Tasks to the Scheduler
- Drive the Scheduler queue via drain when needed
- Provide a get path that retrieves values from ObjectRef

An important point is that Session does not call Worker.execute_task. Worker.execute_task is called inside the Scheduler after placement decisions are made, during the execution step.

## Driver

A driver is the central process where the user program runs in a distributed runtime. It typically does the following.

- Run user code and create Tasks
- Submit Tasks to the runtime and drive execution forward
- Query results via ObjectRef and compose the next work

If you think of Ray, the process where you run your Python script is the driver. Workers are executors that perform the actual execution, and the driver does not manage executors directly. Instead, it coordinates execution through the scheduler and the runtime API. In this project, Session serves as the runtime handle for the driver.

## Overall Relationship

```text
User code (driver process)
  -> remote(...) creates a Task
  -> Session.submit(Task)
       -> Scheduler.submit(Task)
            -> (if possible) Scheduler.try_place(Task)
                 -> eligible_nodes(Task)
                 -> policy.select(candidates)
                 -> Worker.execute_task(Task) or RemoteWorkerProxy.execute_task(Task)
                 -> returns ObjectRef
            -> (if not possible) enqueue it

  -> Session.drain()
       -> Scheduler.drain()
            -> (iterate the queue)
            -> Scheduler.try_place(Task)
                 -> calls Worker.execute_task(...)
                 -> produces a list of ObjectRef

  -> Session.get(ObjectRef)
       -> local lookup
       -> (if needed) drive execution via Scheduler.drain()
       -> (if needed) remote fetch via router+rpc, then local cache
```

Session is the gateway that drives the Scheduler inside the driver. Execution calls happen inside the Scheduler.

## submit and drain

Session.submit passes a Task to the Scheduler. The Scheduler may attempt immediate execution, and if it cannot, it enqueues the Task.

- submit(task)
  - Calls scheduler.submit(task)
  - If placement is possible now, an ObjectRef may be returned immediately
  - If placement is not possible, the Task enters the Scheduler queue and None is returned

Session.drain drives the Scheduler so that queued Tasks are executed as much as possible.

- drain()
  - Calls scheduler.drain()
  - Pops queued Tasks in order and attempts placement
  - For each Task that is placed, Worker.execute_task is called inside the Scheduler
  - Returns a list of ObjectRef produced by executed tasks

The blocking=True option is a driver convenience feature. If immediate placement fails, it runs drain to drive execution at least once, finds the ObjectRef for the requested Task among the produced results, and performs get as well. This does not execute directly, but secures the result by driving the Scheduler one more time.

## put

put stores a value into an ObjectStore and returns an ObjectRef. If there is a local worker, it uses that worker's store. If there is no local worker, it stores into driver_store (ObjectStore("__driver__")).

put is independent of Task execution. It is a feature for the driver to prepare values ahead of time. As a result, you can view put as a simple tool for managing input data or configuration values in an ObjectRef-based model.

## get

Session.get provides the path that turns an ObjectRef into an actual Python value. The key concept is that get is not just a lookup. If needed, it drives scheduling forward, fetches remotely over the network, and caches fetched objects locally.

In this project, get works in the following order.

```text
get(ref) lookup order

0) If ref is None
   - Call Scheduler.drain() once and select the most recent result ref

1) Check aliases cache
   - If there is a recorded mapping remote object_id -> local object_id
     return immediately from the local cache store

2) Owner-first (when the owner is local)
   - If ref.owner_node_id is a local worker id, look in that store first

3) Scan all local workers
   - If it exists in any local store, return the value

4) If still missing, drive scheduling
   - Repeatedly call Scheduler.drain() to advance execution
   - If execution progresses, the result may appear in a local store

5) If still missing, remote fetch
   - If router+rpc is configured, fetch bytes from the owner node
   - Re-materialize by storing bytes into the local cache_store via put_bytes
   - Record aliases[remote_id] = local_id to speed up later get calls
```

Here too, it does not call Worker.execute_task directly. get only drives execution by calling Scheduler.drain, and execution calls happen inside Scheduler.try_place.

## aliases and local cache

When you store an object fetched from remote into a local store, the local store issues a new object_id. So to create a local hit when the same remote object_id is requested again, you need a mapping from remote id to local id. That is aliases.

- remote object_id is an identifier in the remote node's namespace
- a local store issues a new local object_id
- aliases connects remote id to local id to make later get calls fast

If there is no local worker, driver_store serves as the cache.

## PlacementGroup driver API

Session provides a driver API to create PlacementGroups.

- create_placement_group
  - Creates a PlacementGroup from a list of Bundles and a strategy
  - Registers it via scheduler.register_placement_group

- remove_placement_group
  - Unregisters it via scheduler.unregister_placement_group

The key point is that Session does not manage placement directly. It is the gateway that registers inputs so the Scheduler can make placement decisions.

## Global session

GLOBAL_SESSION, init_session, and get_session are convenience layers to simplify driver UX. From the user's perspective, you can directly call functions like put, get, and submit, and internally they delegate to the global Session object.

## Session: 3-line Summary

- The driver is the central process where user code runs, and Session is the runtime handle of the driver
- Session owns the Scheduler and provides execution driving and result collection via submit, drain, and get
  - get combines local lookup, scheduling progress, remote fetch, and local cache into one flow