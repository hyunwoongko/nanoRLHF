# Distributed Programming

Because large-scale models are so large, you often have to split the model across multiple GPUs. Then, the split pieces of the model need to communicate over the network and exchange values with each other. Distributing such large resources across multiple computers or multiple devices to process them is called distributed processing. This time, we will learn the basics of distributed programming using PyTorch.

## Multi-processing with PyTorch

Before the distributed programming tutorial, we will go through a tutorial on a multi-processing application implemented with PyTorch. Concepts like threads and processes are typically covered in an operating systems course if you majored in Computer Science, so we will skip them here. If you are not familiar with these concepts, I recommend searching on Google or reading an article like https://www.backblaze.com/blog/whats-the-diff-programs-processes-and-threads/ first.

### Basic terminology
- Node: You can generally think of this as a computer. For example, 3 nodes means 3 computers.
- Global Rank: Originally it refers to process priority, but here you can think of it as the GPU ID.
- Local Rank: Originally it refers to process priority within a node, but here you can think of it as the GPU ID within a node.
- World Size: It refers to the number of processes.

![](https://github.com/hyunwoongko/nanoRLHF/blob/main/nanorlhf/nanotron/docs/assets/process_terms.png?raw=true)

### How to run a multi-process application
There are two main ways to run a multi-process application implemented with PyTorch.

1. Your code becomes the main process and forks a specific function into subprocesses.
2. The PyTorch launcher becomes the main process and forks the entire user code into subprocesses.

We will cover both methods. Here, the expression fork means that one process becomes the parent and runs multiple subprocesses simultaneously.

#### 1) Your code becomes the main process and forks a specific function into subprocesses
In this approach, your code is the main process, and a specific function is forked as subprocesses.

![](https://github.com/hyunwoongko/nanoRLHF/blob/main/nanorlhf/nanotron/docs/assets/multi_process_1.png?raw=true)

In general, there are two ways to fork subprocesses: Spawn and Fork.

- Spawn
  - Does not inherit the main process resources, and allocates only the necessary resources newly to the subprocess.
  - Slower but safer.
- Fork
  - Shares all resources of the main process with the subprocess and starts the process.
  - Faster but riskier.

p.s. In practice, there is also the Forkserver method, but we omit it because it is uncommon and not frequently used.

```python
import torch.multiprocessing as mp

def fn(rank, param1, param2):
    print(f"{param1} {param2} - rank: {rank}")


if __name__ == "__main__":
    processes = []
    mp.set_start_method("spawn")

    for rank in range(4):
        process = mp.Process(target=fn, args=(rank, "A0", "B1"))
        process.daemon = False
        process.start()
        processes.append(process)

    for process in processes:
        process.join()
```

Using the torch.multiprocessing.spawn function makes this much easier.

```python
import torch.multiprocessing as mp

def fn(rank, param1, param2):
    print(f"{param1} {param2} - rank: {rank}")


# Main process
if __name__ == "__main__":
    mp.spawn(
        fn=fn,
        args=("A0", "B1"),
        nprocs=4,
        join=True,
        daemon=False,
        start_method="spawn",
    )
```

The mp.spawn function internally creates multiple subprocesses and runs the fn function in each subprocess. The args argument contains additional arguments passed to fn. The nprocs argument indicates how many subprocesses to create.

```python
def start_processes(fn, args=(), nprocs=1, join=True, daemon=False, start_method='spawn'):
    _python_version_check()
    mp = multiprocessing.get_context(start_method)
    error_queues = []
    processes = []
    for i in range(nprocs):
        error_queue = mp.SimpleQueue()
        process = mp.Process(
            target=_wrap,
            args=(fn, i, args, error_queue),
            daemon=daemon,
        )
        process.start()
        error_queues.append(error_queue)
        processes.append(process)

    context = ProcessContext(processes, error_queues)
    if not join:
        return context

    # Loop on join until it returns True or raises an exception.
    while not context.join():
        pass
```

Later, in the nanoverl project, we will perform distributed programming through nanoray. At that time, nanoray internally creates multiple processes and runs remote functions in each process.

```python
from nanorlhf import nanoray


@nanoray.remote
def fn(rank, param1, param2):
    print(f"{param1} {param2} - rank: {rank}")


configs = {
    "rpc-node-1": nanoray.NodeConfig(rpc=True, port=8092),
    "rpc-node-2": nanoray.NodeConfig(rpc=True, port=8093),
}

nanoray.init(configs, default_node_id="rpc-node-1")

refs = []
for rank in range(4):
    ref = fn.remote(rank, "A0", "B1")
    refs.append(ref)
nanoray.get(refs)
nanoray.shutdown()
```

#### 2) The PyTorch launcher becomes the main process and forks the entire user code into subprocesses

This is a very convenient method where the multiprocessing launcher built into torch runs the entire user code as subprocesses.

![](https://github.com/hyunwoongko/nanoRLHF/blob/main/nanorlhf/nanotron/docs/assets/multi_process_2.png?raw=true)

With the command torchrun --nproc_per_node=N your_script.py, you can create N processes and run the your_script.py script in each process.

```bash
import os

# Variables like RANK, LOCAL_RANK, WORLD_SIZE are set automatically.
print(f"hello world, {os.environ['RANK']}")
```

## Distributed Programming with PyTorch

### Message Passing

Message passing is a technique where multiple processes that do not share the same address space exchange indirect information called messages so they can send and receive data. For example, if Process-1 is coded to send data with a certain tag into a message queue, and Process-2 is coded to receive that data, then the two processes can exchange data without sharing memory.

![](https://github.com/hyunwoongko/nanoRLHF/blob/main/nanorlhf/nanotron/docs/assets/message_passing.png?raw=true)

### MPI (Massage Passing Interface)

MPI refers to the standard interface for message passing. MPI defines many operations used for message passing between processes (e.g. broadcast, reduce, scatter, gather, ...), and a 대표적인 open-source implementation is OpenMPI.

![](https://github.com/hyunwoongko/nanoRLHF/blob/main/nanorlhf/nanotron/docs/assets/open_mpi.png?raw=true)

### NCCL & GLOO
In practice, you tend to use libraries like nccl or gloo rather than openmpi.

- NCCL (NVIDIA Collective Communication Library)
  - A GPU-focused message passing library developed by NVIDIA (pronounced nickel)
  - It is known to offer much higher performance on NVIDIA GPUs compared to other tools.
- GLOO (Facebook's Collective Communication Library)
  - A message passing library developed by Facebook.
  - In torch, it is mainly recommended for CPU distributed processing.

Unless you have a special reason to use openmpi, you can use nccl or gloo. Use nccl for GPUs and gloo for CPUs. For more details, refer to https://pytorch.org/docs/stable/distributed.html.

### torch.distributed package

Directly using gloo, nccl, openmpi can be a good experience. However, due to time constraints, we cannot cover all of them, so we will proceed using the torch.distributed package that wraps them. In real practice, you usually program with higher-level packages like torch.distributed rather than using nccl directly.

### Process Group
Managing many processes is difficult, so process groups are used to make management easier. When you call init_process_group, a default_pg (process group) that includes all processes is created. The init_process_group function that initializes the process group must be executed in subprocesses, and if you want to create a group containing only processes you choose, call new_group.

```python
import torch.multiprocessing as mp
import torch.distributed as dist
import os


def fn(rank, world_size):
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    group = dist.new_group([_ for _ in range(world_size)])
    print(f"{group} - rank: {rank}")


if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["WORLD_SIZE"] = "4"

    mp.spawn(
        fn=fn,
        args=(4,),
        nprocs=4,
        join=True,
        daemon=False,
        start_method="spawn",
    )
```
```python
python3 process_group.py
```

### P2P (Point to Point) Communication
P2P (point to point) communication is communication where data is sent from one specific process to another, and you can use the send and recv functions in the torch.distributed package.

![](https://github.com/hyunwoongko/nanoRLHF/blob/main/nanorlhf/nanotron/docs/assets/p2p.png?raw=true)

```python
import torch
import torch.distributed as dist

dist.init_process_group("nccl")

if dist.get_rank() == 0:
    tensor = torch.randn(2, 2)
    dist.send(tensor, dst=1)

elif dist.get_rank() == 1:
    tensor = torch.zeros(2, 2)
    print(f"rank 1 before: {tensor}\n")
    dist.recv(tensor, src=0)
    print(f"rank 1 after: {tensor}\n")

else:
    raise RuntimeError("wrong rank")
```
```text
torchrun --nproc_per_node=2 p2p.py
```

Note that these communicate synchronously. For asynchronous (non-blocking) communication, use isend and irecv. Because they work asynchronously, you must wait with wait() until the communication from the other process is finished before accessing the data.

```python
import torch
import torch.distributed as dist

dist.init_process_group("nccl")

if dist.get_rank() == 0:
    tensor = torch.randn(2, 2)
    request = dist.isend(tensor, dst=1)
elif dist.get_rank() == 1:
    tensor = torch.zeros(2, 2)
    request = dist.irecv(tensor, src=0)
else:
    raise RuntimeError("wrong rank")

request.wait()

print(f"rank {dist.get_rank()}: {tensor}")
```
```text
torchrun --nproc_per_node=2 p2p_non_blocking.py
```

### Collective Communication
Collective Communication means collective communication where multiple processes participate. There are many operations, but the basic set consists of four operations: broadcast, scatter, gather, reduce.

![](https://github.com/hyunwoongko/nanoRLHF/blob/main/nanorlhf/nanotron/docs/assets/collective.png?raw=true)

In addition, we will look at a total of 8 operations including composite operations like all-reduce, all-gather, reduce-scatter, and the synchronization operation barrier. Also, if you want to run these operations in asynchronous mode, set the async_op parameter to True when performing each operation.

#### Broadcast

Broadcast is an operation that copies data from a specific process to all processes in the group.

![](https://github.com/hyunwoongko/nanoRLHF/blob/main/nanorlhf/nanotron/docs/assets/broadcast.png?raw=true)

```python
import torch
import torch.distributed as dist

dist.init_process_group("nccl")
rank = dist.get_rank()
torch.cuda.set_device(rank)

if rank == 0:
    tensor = torch.randn(2, 2).to(torch.cuda.current_device())
else:
    tensor = torch.zeros(2, 2).to(torch.cuda.current_device())

print(f"before rank {rank}: {tensor}\n")
dist.broadcast(tensor, src=0)
print(f"after rank {rank}: {tensor}\n")
```
```text
torchrun --nproc_per_node=2 broadcast.py
```

Broadcast can also be used for P2P communication like send and recv.

```python
import torch
import torch.distributed as dist

dist.init_process_group("nccl")

if dist.get_rank() == 0:
    tensor = torch.randn(2, 2).to(torch.cuda.current_device())
    dist.broadcast(tensor, src=0)
elif dist.get_rank() == 1:
    tensor = torch.zeros(2, 2).to(torch.cuda.current_device())
    dist.broadcast(tensor, src=0)
else:
    raise RuntimeError("wrong rank")
```
```text
torchrun --nproc_per_node=2 broadcast_p2p.py
```

This produces the same effect as sending data from rank 0 to rank 1.

#### Reduce

Reduce is an operation that applies a certain operation to the data held by each process and gathers the output onto a single device. The operations are typically sum, max, min, and so on.

![](https://github.com/hyunwoongko/nanoRLHF/blob/main/nanorlhf/nanotron/docs/assets/reduce.png?raw=true)

```python
import torch
import torch.distributed as dist

dist.init_process_group("nccl")
rank = dist.get_rank()
torch.cuda.set_device(rank)

tensor = torch.ones(2, 2).to(torch.cuda.current_device()) * rank
# rank==0 => [[0, 0], [0, 0]]
# rank==1 => [[1, 1], [1, 1]]
# rank==2 => [[2, 2], [2, 2]]
# rank==3 => [[3, 3], [3, 3]]

dist.reduce(tensor, op=torch.distributed.ReduceOp.SUM, dst=0)

if rank == 0:
    print(tensor)
```
```text
torchrun --nproc_per_node=4 reduce_sum.py
```

```python
import torch
import torch.distributed as dist

dist.init_process_group("nccl")
rank = dist.get_rank()
torch.cuda.set_device(rank)

tensor = torch.ones(2, 2).to(torch.cuda.current_device()) * rank
# rank==0 => [[0, 0], [0, 0]]
# rank==1 => [[1, 1], [1, 1]]
# rank==2 => [[2, 2], [2, 2]]
# rank==3 => [[3, 3], [3, 3]]

dist.reduce(tensor, op=torch.distributed.ReduceOp.MAX, dst=0)

if rank == 0:
    print(tensor)
```
```text
torchrun --nproc_per_node=4 reduce_max.py
```

#### Scatter
Scatter is an operation that splits multiple elements and distributes them to each device.

![](https://github.com/hyunwoongko/nanoRLHF/blob/main/nanorlhf/nanotron/docs/assets/scatter.png?raw=true)

```python
import torch
import torch.distributed as dist

dist.init_process_group("nccl")
rank = dist.get_rank()
torch.cuda.set_device(rank)


output = torch.zeros(1)
print(f"before rank {rank}: {output}\n")

if rank == 0:
    inputs = torch.tensor([10.0, 20.0, 30.0, 40.0])
    inputs = torch.split(inputs, dim=0, split_size_or_sections=1)
    # (tensor([10]), tensor([20]), tensor([30]), tensor([40]))
    dist.scatter(output, scatter_list=list(inputs), src=0)
else:
    dist.scatter(output, src=0)

print(f"after rank {rank}: {output}\n")
```
```text
torchrun --nproc_per_node=4 scatter.py
```

#### Gather
Gather is an operation that gathers elements scattered across multiple devices onto a single device.

![](https://github.com/hyunwoongko/nanoRLHF/blob/main/nanorlhf/nanotron/docs/assets/gather.png?raw=true)

```python
import torch
import torch.distributed as dist

dist.init_process_group("nccl")
rank = dist.get_rank()
torch.cuda.set_device(rank)

input = torch.ones(1) * rank
# rank==0 => [0]
# rank==1 => [1]
# rank==2 => [2]
# rank==3 => [3]

if rank == 0:
    outputs_list = [torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(1)]
    dist.gather(input, gather_list=outputs_list, dst=0)
    print(outputs_list)
else:
    dist.gather(input, dst=0)
```

#### All-Reduce
You can think of operations with All- in front as performing the operation and then broadcasting the result to all devices. As shown below, all-reduce performs reduce and then copies the computed result to all devices.

![](https://github.com/hyunwoongko/nanoRLHF/blob/main/nanorlhf/nanotron/docs/assets/all_reduce.png?raw=true)

```python
import torch
import torch.distributed as dist

dist.init_process_group("nccl")
rank = dist.get_rank()
torch.cuda.set_device(rank)

tensor = torch.ones(2, 2).to(torch.cuda.current_device()) * rank
# rank==0 => [[0, 0], [0, 0]]
# rank==1 => [[1, 1], [1, 1]]
# rank==2 => [[2, 2], [2, 2]]
# rank==3 => [[3, 3], [3, 3]]

dist.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)

print(f"rank {rank}: {tensor}\n")
```
```text
torchrun --nproc_per_node=4 all_reduce_sum.py
```

```python
import torch
import torch.distributed as dist

dist.init_process_group("nccl")
rank = dist.get_rank()
torch.cuda.set_device(rank)

tensor = torch.ones(2, 2).to(torch.cuda.current_device()) * rank
# rank==0 => [[0, 0], [0, 0]]
# rank==1 => [[1, 1], [1, 1]]
# rank==2 => [[2, 2], [2, 2]]
# rank==3 => [[3, 3], [3, 3]]

dist.all_reduce(tensor, op=torch.distributed.ReduceOp.MAX)

print(f"rank {rank}: {tensor}\n")
```
```text
torchrun --nproc_per_node=4 all_reduce_max.py
```

#### All-Gather
All-gather performs gather and then copies the computed result to all devices.

![](https://github.com/hyunwoongko/nanoRLHF/blob/main/nanorlhf/nanotron/docs/assets/all_gather.png?raw=true)

```python
import torch
import torch.distributed as dist

dist.init_process_group("nccl")
rank = dist.get_rank()
torch.cuda.set_device(rank)

input = torch.ones(1).to(torch.cuda.current_device()) * rank
# rank==0 => [0]
# rank==1 => [1]
# rank==2 => [2]
# rank==3 => [3]

outputs_list = [
    torch.zeros(1, device=torch.device(torch.cuda.current_device())),
    torch.zeros(1, device=torch.device(torch.cuda.current_device())),
    torch.zeros(1, device=torch.device(torch.cuda.current_device())),
    torch.zeros(1, device=torch.device(torch.cuda.current_device())),
]

dist.all_gather(tensor_list=outputs_list, tensor=input)
print(outputs_list)
```
```text
torchrun --nproc_per_node=4 all_gather.py
```

#### Reduce-Scatter
Reduce-scatter splits the computed result and distributes it to each device after performing reduce. As the name suggests, it is an operation that reduces and then scatters the result.

![](https://github.com/hyunwoongko/nanoRLHF/blob/main/nanorlhf/nanotron/docs/assets/reduce_scatter.png?raw=true)

```python
import torch
import torch.distributed as dist

dist.init_process_group("nccl")
rank = dist.get_rank()
torch.cuda.set_device(rank)

input_list = torch.tensor([1, 10, 100, 1000]).to(torch.cuda.current_device()) * rank
input_list = torch.split(input_list, dim=0, split_size_or_sections=1)
# rank==0 => [0, 00, 000, 0000]
# rank==1 => [1, 10, 100, 1000]
# rank==2 => [2, 20, 200, 2000]
# rank==3 => [3, 30, 300, 3000]

output = torch.tensor([0], device=torch.device(torch.cuda.current_device()),)

dist.reduce_scatter(
    output=output,
    input_list=list(input_list),
    op=torch.distributed.ReduceOp.SUM,
)

print(f"rank {rank}: {output}\n")
```
```text
torchrun --nproc_per_node=4 reduce_scatter.py
```

#### Barrier
Barrier is used to synchronize processes. A process that reaches the barrier first waits until all processes have executed up to that point.

```python
import time
import torch.distributed as dist

dist.init_process_group("nccl")
rank = dist.get_rank()

if rank == 0:
    seconds = 0
    while seconds <= 3:
        time.sleep(1)
        seconds += 1
        print(f"rank 0 - seconds: {seconds}\n")

print(f"rank {rank}: no-barrier\n")
dist.barrier()
print(f"rank {rank}: barrier\n")
```
```text
torchrun --nproc_per_node=4 barrier.py
```

## Distributed Programming: 3-line Summary
- Distributed programming is a technique that distributes work across multiple computers or devices.
- In PyTorch, distributed programming is supported through the torch.distributed package, and there are representative P2P communication and collective communication patterns.
- Representative collective communication operations include broadcast, reduce, scatter, gather, all-reduce, all-gather, reduce-scatter, and barrier.