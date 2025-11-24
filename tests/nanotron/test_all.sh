#torchrun --nproc_per_node=4 train.py --tp 4 --pp 1 --dp 1 --stg 0  # tp only
#torchrun --nproc_per_node=4 train.py --tp 1 --pp 4 --dp 1 --stg 0  # pp only
#torchrun --nproc_per_node=4 train.py --tp 1 --pp 1 --dp 4 --stg 1  # dp only stg=1
#torchrun --nproc_per_node=4 train.py --tp 1 --pp 1 --dp 4 --stg 2  # dp only stg=2
torchrun --nproc_per_node=4 train.py --tp 1 --pp 1 --dp 4 --stg 3  # dp only stg=3

#torchrun --nproc_per_node=4 train.py --tp 2 --pp 2 --dp 1 --stg 0  # tp + pp stg=0
#torchrun --nproc_per_node=4 train.py --tp 2 --pp 1 --dp 2 --stg 0  # tp + dp stg=0
#torchrun --nproc_per_node=4 train.py --tp 1 --pp 2 --dp 2 --stg 0  # pp + dp stg=0

#torchrun --nproc_per_node=4 train.py --tp 2 --pp 1 --dp 2 --stg 1  # tp + dp stg=1
#torchrun --nproc_per_node=4 train.py --tp 1 --pp 2 --dp 2 --stg 1  # pp + dp stg=1
#torchrun --nproc_per_node=4 train.py --tp 2 --pp 1 --dp 2 --stg 2  # tp + dp stg=2
torchrun --nproc_per_node=8 train.py --tp 4 --pp 1 --dp 2 --stg 3  # tp + dp stg=3

#torchrun --nproc_per_node=8 train.py --tp 2 --pp 2 --dp 2 --stg 1  # tp + pp + dp stg=1

python3 plot.py
