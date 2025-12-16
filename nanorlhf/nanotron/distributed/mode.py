from enum import Enum


class ParallelMode(Enum):
    """Enum class for parallelization mode."""

    GLOBAL = "global"
    DATA = "data"
    TENSOR = "tensor"
    PIPELINE = "pipeline"
    TIED_EMBEDDING = "tied_embedding"

    ROLLOUT_TENSOR = "rollout_tensor"
    ROLLOUT_DATA = "rollout_data"
