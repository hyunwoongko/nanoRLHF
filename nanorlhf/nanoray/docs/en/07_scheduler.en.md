# Scheduler

## Overview

The Scheduler is the component that receives a Task, chooses a node (node_id) to run it on, then delegates execution to that node's Worker. In a distributed runtime, the core is less about execution itself and more about making placement decisions consistently, and the Scheduler is responsible for that decision.

In this project, the Scheduler does the following.

- Accepts a Task as input
- Computes eligible candidate nodes
- Selects one among candidates via a SchedulingPolicy
- Calls Worker.execute_task on the selected node
- Returns the produced ObjectRef

## Overall Relationship

```text
RemoteFunction / ActorClass / ActorMethod
  -> create a Task
  -> Session.submit(Task)
  -> Scheduler.submit(Task)
     -> eligible_nodes(Task)
     -> policy.select(candidates)
     -> Worker.execute_task(Task)
     -> return ObjectRef
```

The Scheduler is not the execution engine. Execution is done by the Worker. The Scheduler makes placement decisions, then triggers execution.

## Execution Flow

Scheduler.submit attempts immediate execution, and if it cannot, it enqueues the task.

```text
1) submit(Task)
   - check whether it can run right now via try_place(Task)
   - if possible, immediately call Worker.execute_task -> return ObjectRef
   - if not possible, push (seq, Task) into the queue and return None

2) drain()
   - repeat until the queue is empty or no further progress is possible
   - in each round, try to place each Task in the queue at most once, in order
```

Because of this structure, the Scheduler has the following properties.

- Tasks that are immediately runnable execute without delay
- Tasks that are not runnable right away wait in the order specified by the Policy (FIFO or RoundRobin)
- When resources are freed, it retries in the next round

## Candidate Node Computation

The core of the Scheduler is eligible_nodes. This is where it computes which nodes a Task can run on. The constraints used for this computation fall into three main categories.

- Resource constraints: CPU, GPU, custom resources
- pinned_node_id: force a specific node
- PlacementGroup: reflect PACK, SPREAD intent

If pinned_node_id is present, the candidate set is effectively 0 or 1. If the node does not exist or resources do not fit, the candidate list becomes empty.

If a PlacementGroup is present, based on bundle_index it determines where that bundle should run, or if it is PACK, the whole group is locked to one node.

## SchedulingPolicy

SchedulingPolicy is the strategy that chooses one node given a list of candidate nodes. The Scheduler gives candidates to the policy, then uses the node the policy selects as-is.

- FIFO: chooses the first node from the global node order that appears in candidates
- RoundRobin: cycles through the global node order and alternates among candidates

What the policy does is simple. The Scheduler decides eligibility, and the policy only chooses among eligible candidates.

## Queueing Model

The Scheduler holds its internal queue as a heap. The priority is seq, and since seq increases, it is effectively FIFO.

- if immediate execution fails in submit, it pushes (seq, task)
- in drain, it pops and tries placement in order

drain is round-based. If in a round not a single task is placed, the next round is very likely to be identical, so it stops. That is why it terminates deterministically without an infinite loop.

## NodeState

NodeState tracks the current resource state of each node.

- total: the node's total capacity
- used: the amount currently consumed by running Tasks
- reserved: the amount pre-reserved for reasons such as PlacementGroup

A normal Task is allocate'd before execution and release'd when execution finishes. In this project, allocate and release wrap the Worker.execute_task call.

With a PlacementGroup, it is a bit different. PACK and SPREAD perform reserve_bundle at the bundle level, and they keep that reservation until it is released. In other words, a reservation can live longer than the lifetime of a single Task.

## PlacementGroup Handling

If a PlacementGroup exists, the Scheduler records placement decisions in placement_group_assignment.

- PACK
  - records the locked node id under the `__pack__` key
  - on first placement, reserves all bundles in the group on one node via reserve_bundle and locks it
- SPREAD
  - records which node each bundle_index was assigned to
  - for bundles not yet assigned, it prefers nodes that have not been used, among feasible nodes

The important point is that a PlacementGroup is an intent. The Scheduler concretizes it through candidate computation and reservations so that later Task placement does not wobble.

## WorkerLike

The Scheduler only needs to know a minimal interface called WorkerLike. That is, the Scheduler does not need to distinguish whether execution is done by a local Worker or a remote proxy. It only needs execute_task.

This design separates placement logic from execution transport. The key message is that the Scheduler should be independent of how execution is transported.

## Scheduler: 3-line Summary

- The Scheduler receives a Task, decides an execution node, then calls Worker.execute_task
- eligible_nodes determines feasibility, and SchedulingPolicy handles selection among candidates
- NodeState tracks used and reserved resources, and PlacementGroup fixes intent via reservations and recorded assignments