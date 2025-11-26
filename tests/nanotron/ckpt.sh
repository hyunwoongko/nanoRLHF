torchrun --nproc_per_node=4 ckpt.py --tp 4 --pp 1
torchrun --nproc_per_node=4 ckpt.py --tp 1 --pp 4
torchrun --nproc_per_node=4 ckpt.py --tp 2 --pp 2
torchrun --nproc_per_node=8 ckpt.py --tp 2 --pp 2 --dp 2 --stg 1
torchrun --nproc_per_node=8 ckpt.py --tp 1 --pp 1 --dp 8 --stg 3


