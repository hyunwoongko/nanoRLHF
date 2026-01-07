# Distributed Programming

Large-scale 모델은 크기가 크기 때문에 여러대의 GPU에 쪼개서 모델을 올려야 합니다. 그리고 쪼개진 각 모델의 조각들끼리 네트워크로 통신을 하면서 값을 주고 받아야 합니다. 이렇게 커다란 리소스를 여러대의 컴퓨터 혹은 여러대의 장비에 분산시켜서 처리하는 것을 '분산처리'라고 합니다. 이번에는 PyTorch를 이용한 분산 프로그래밍의 기초에 대해 알아보겠습니다.

## Multi-processing with PyTorch

분산프로그래밍 튜토리얼에 앞서 PyTorch로 구현된 Multi-processing 애플리케이션에 대한 튜토리얼을 진행합니다. 쓰레드 및 프로세스의 개념 등은 Computer Scienece 전공자라면 운영체제 시간에 배우는 것들이니 생략하도록 하겠습니다. 만약 이러한 개념에 대해 잘 모르신다면, 구글에 검색하시거나 https://www.backblaze.com/blog/whats-the-diff-programs-processes-and-threads/ 와 같은 글을 먼저 읽어보는 것을 추천드립니다.

### 기본 용어
- 노드 (Node): 일반적으로 컴퓨터라고 생각하시면 됩니다. 노드 3대라고 하면 컴퓨터 3대를 의미합니다.
- 글로벌 랭크 (Global Rank):  원래는 프로세스의 우선순위를 의미하지만 여기에서는 GPU의 ID라고 보시면 됩니다.
- 로컬 랭크 (Local Rank): 원래는 한 노드내에서의 프로세스 우선순위를 의미하지만 여기에서는 노드내의 GPU ID라고 보시면 됩니다.
- 월드 사이즈 (World Size): 프로세스의 개수를 의미합니다.

![](https://github.com/hyunwoongko/nanoRLHF/blob/main/nanorlhf/nanotron/docs/assets/process_terms.png?raw=true)

### Multi-process application 실행 방법
PyTorch로 구현된 Multi-process 애플리케이션을 실행시키는 방법은 크게 두가지가 있습니다.

1. 사용자의 코드가 메인프로세스가 되어 특정 함수를 서브프로세스로 분기한다.
2. PyTorch 런처가 메인프로세스가 되어 사용자 코드 전체를 서브프로세스로 분기한다.

이 두가지 방법에 대해 모두 알아보겠습니다. 이때, '분기한다.'라는 표현이 나오는데, 이는 한 프로세스가 부모가 되어 여러개의 서브프로세스를 동시에 실행시키는 것을 의미합니다.

#### 1) 사용자의 코드가 메인프로세스가 되어 특정 함수를 서브프로세스로 분기한다.
이 방식은 사용자의 코드가 메인프로세스가 되며 특정 function을 서브프로세스로써 분기하는 방식입니다.

![](https://github.com/hyunwoongko/nanoRLHF/blob/main/nanorlhf/nanotron/docs/assets/multi_process_1.png?raw=true)

일반적으로 Spawn과 Fork 등 두가지 방식으로 서브프로세스를 분기 할 수 있습니다.

- Spawn
  - 메인프로세스의 자원을 물려주지 않고 필요한 만큼의 자원만 서브프로세스에게 새로 할당.
  - 속도가 느리지만 안전한 방식.
- Fork
  - 메인프로세스의 모든 자원을 서브프로세스와 공유하고 프로세스를 시작.
  - 속도가 빠르지만 위험한 방식.

p.s. 실제로는 Forkserver 방식도 있지만 자주 사용되지 않는 생소한 방식이기에 생략합니다.

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

`torch.multiprocessing.spawn` 함수를 이용하면 이 과정을 훨씬 쉽게 진행 할 수 있습니다.

```python
import torch.multiprocessing as mp

def fn(rank, param1, param2):
    print(f"{param1} {param2} - rank: {rank}")


# 메인 프로세스
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

`mp.spawn` 함수는 내부적으로 여러개의 서브프로세스를 생성하고, 각각의 서브프로세스에서 `fn` 함수를 실행시킵니다. `args` 인자는 `fn` 함수에 전달되는 추가적인 인자들입니다. `nprocs` 인자는 생성할 서브프로세스의 개수를 의미합니다.

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

우리는 이후에 `nanoverl` 프로젝트에서 `nanoray`를 통해 분산 프로그래밍을 수행하게 됩니다. 이때 `nanoray`는 내부적으로 여러개의 프로세스를 생성하고, 각각의 프로세스에서 원격 함수를 실행시킵니다.

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

#### 2) PyTorch 런처가 메인프로세스가 되어 사용자 코드 전체를 서브프로세스로 분기한다.

이 방식은 torch에 내장된 멀티프로세싱 런처가 사용자 코드 전체를 서브프로세스로 실행시켜주는 매우 편리한 방식입니다.

![](https://github.com/hyunwoongko/nanoRLHF/blob/main/nanorlhf/nanotron/docs/assets/multi_process_2.png?raw=true)

`torchrun --nproc_per_node=N your_script.py` 명령어를 통해 N개의 프로세스를 생성하고, 각각의 프로세스에서 `your_script.py` 스크립트를 실행시킬 수 있습니다.

```bash
import os

# RANK, LOCAL_RANK, WORLD_SIZE 등의 변수가 자동으로 설정됩니다.
print(f"hello world, {os.environ['RANK']}")
```

## Distributed Programming with PyTorch

### Message Passing

메시지 패싱이란 동일한 주소공간을 공유하지 않는 여러 프로세스들이 데이터를 주고 받을 수 있도록 메시지라는 간접 정보를 주고 받는 것입니다. 예를 들면 Process-1이 특정 태그가 달린 데이터를 메시지 큐에 send하도록, Process-2가 해당 데이터를 receive하도록 코딩해놓으면 두 프로세스가 공유하는 메모리 공간 없이도 데이터를 주고 받을 수 있죠.

![](https://github.com/hyunwoongko/nanoRLHF/blob/main/nanorlhf/nanotron/docs/assets/message_passing.png?raw=true)

### MPI (Massage Passing Interface)

MPI는 Message Passing에 대한 표준 인터페이스를 의미합니다. MPI에는 Process간의 Message Passing에 사용되는 여러 연산(e.g. broadcast, reduce, scatter, gather, ...)이 정의되어 있으며 대표적으로 OpenMPI라는 오픈소스가 존재합니다.

![](https://github.com/hyunwoongko/nanoRLHF/blob/main/nanorlhf/nanotron/docs/assets/open_mpi.png?raw=true)

### NCCL & GLOO
실제로는 openmpi 보다는 nccl이나 gloo 같은 라이브러리를 사용하게 됩니다.

- NCCL (NVIDIA Collective Communication Library)
  - NVIDIA에서 개발한 GPU 특화 Message Passing 라이브러리 ('nickel'이라고 읽음)
  - NVIDIA GPU에서 사용시, 다른 도구에 비해 월등히 높은 성능을 보여주는 것으로 알려져있습니다.
- GLOO (Facebook's Collective Communication Library)
  - Facebook에서 개발된 Message Passing 라이브러리.
  - torch에서는 주로 CPU 분산처리에 사용하라고 추천하고 있습니다.

openmpi를 써야할 특별한 이유가 있는 것이 아니라면 nccl이나 gloo를 사용하는데, 
GPU에서 사용시 nccl, CPU에서 사용시 gloo를 사용하시면 됩니다. 
더 자세한 정보는 https://pytorch.org/docs/stable/distributed.html 이곳을 참고하세요.

### torch.distributed 패키지

gloo, nccl, openmpi 등을 직접 사용해보는 것은 분명 좋은 경험이 될 것입니다. 
그러나 시간 관계상 이들을 모두 다룰 수는 없고, 이들을 wrapping 하고 있는 `torch.distributed` 패키지를 사용하여 진행하겠습니다. 
실제로 활용 단으로 가면 nccl 등을 직접 사용하지 않고 대부분의 경우 `torch.distributed `등의 하이레벨 패키지를 사용하여 프로그래밍 하게 됩니다.

### Process Group
많은 프로세스를 관리하는 것은 어려운 일입니다. 따라서 프로세스 그룹을 만들어서 관리를 용이하게 합니다. 
`init_process_group`를 호출하면 전체 프로세스가 속한 `default_pg(process group)`가 만들어집니다. 
프로세스 그룹을 초기화하는 `init_process_group` 함수는 반드시 서브프로세스에서 실행되어야 하며, 만약 추가로 사용자가 원하는 프로세스들만 모아서 그룹을 생성하려면 `new_group`을 호출하면 됩니다.

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
P2P (Point to point, 점 대 점) 통신은 특정 프로세스에서 다른 프로세스 데이터를 전송하는 통신이며 `torch.distributed` 패키지의 `send`, `recv` 함수를 활용하여 통신할 수 있습니다.

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
```
torchrun --nproc_per_node=2 p2p.py
```

주의할 것은 이들이 동기적으로 통신한다는 것입니다. 
비동기 통신(non-blocking)에는 `isend`, `irecv`를 이용합니다. 
이들은 비동기적으로 작동하기 때문에 `wait()` 함수를 통해 다른 프로세스의 통신이 끝날때 까지 기다리고 난 뒤에 접근해야합니다.

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
```
torchrun --nproc_per_node=2 p2p_non_blocking.py
```

### Collective Communication
Collective Communication은 집합통신이라는 뜻으로 여러 프로세스가 참여하여 통신하는 것을 의미합니다. 
다양한 연산들이 있지만 기본적으로 아래와 같은 4개의 연산(`broadcast`, `scatter`, `gather`, `reduce`)이 기본 세트입니다.

![](https://github.com/hyunwoongko/nanoRLHF/blob/main/nanorlhf/nanotron/docs/assets/collective.png?raw=true)

여기에 추가로 all-reduce, all-gather, reduce-scatter 등의 복합 연산과 동기화 연산인 barrier까지 총 8개 연산에 대해 알아보겠습니다. 
추가로 만약 이러한 연산들을 비동기 모드로 실행하려면 각 연산 수행시 `async_op` 파라미터를 True로 설정하면 됩니다.

#### Broadcast

Broadcast는 특정 프로세스에 있는 데이터를 그룹내의 모든 프로세스에 복사하는 연산입니다.

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
```
torchrun --nproc_per_node=2 broadcast.py
```

broadcast는 `send`, `recv`처럼 P2P 통신을 위해 사용할 수도 있습니다.

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
```
torchrun --nproc_per_node=2 broadcast_p2p.py
```

이렇게 하면 rank 0에서 rank 1로 데이터를 보내는 것과 동일한 효과를 냅니다.

#### Reduce

Reduce는 각 프로세스가 가진 데이터로 특정 연산을 수행해서 출력을 하나의 디바이스로 모아주는 연산입니다. 연산은 주로 sum, max, min 등이 가능합니다.

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
```
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
```
torchrun --nproc_per_node=4 reduce_max.py
```

#### Scatter
Scatter는 여러개의 element를 쪼개서 각 device에 뿌려주는 연산입니다.

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
```
torchrun --nproc_per_node=4 scatter.py
```

#### Gather
Gather는 여러개의 device에 흩어져 있는 element들을 한 device로 모아주는 연산입니다.

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
이름 앞에 All- 이 붙은 연산들은 해당 연산을 수행 한뒤, 결과를 모든 디바이스로 broadcast한다고 생각하셔도 됩니다. 
아래 그림처럼 All-reduce는 reduce를 수행한 뒤, 계산된 결과를 모든 디바이스로 복사합니다.

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
```
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
```
torchrun --nproc_per_node=4 all_reduce_max.py
```

#### All-Gather
All-gather는 gather를 수행한 뒤, 계산된 결과를 모든 디바이스로 복사합니다.

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
```
torchrun --nproc_per_node=4 all_gather.py
```

#### Reduce-Scatter
Reduce-scatter는 scatter를 수행한 뒤, 계산된 결과를 쪼개서 각 디바이스에 뿌려줍니다.
이름처럼 reduce하고 그 뒤에 scatter하는 연산입니다.

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
```
torchrun --nproc_per_node=4 reduce_scatter.py
```

#### Barrier
Barrier는 프로세스를 동기화 하기 위해 사용됩니다. 먼저 barrier에 도착한 프로세스는 모든 프로세스가 해당 지점까지 실행되는 것을 기다립니다.

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
```
torchrun --nproc_per_node=4 barrier.py
```

## Distributed Programming: 3줄 요약
- 분산 프로그래밍은 여러대의 컴퓨터 혹은 장비에 작업을 분산시켜 처리하는 기법입니다.
- PyTorch에서는 `torch.distributed` 패키지를 통해 분산 프로그래밍을 지원하며 대표적으로 P2P 통신과 Collective 통신이 있습니다.
- 대표적인 Collective 통신 연산으로는 broadcast, reduce, scatter, gather, all-reduce, all-gather, reduce-scatter, barrier 등이 있습니다.