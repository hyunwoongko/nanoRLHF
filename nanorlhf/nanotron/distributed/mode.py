from enum import Enum


class ParallelMode(Enum):
    """Enum class for parallelization mode."""

    GLOBAL = "global"
    DATA = "data"
    MODEL = "model"
    TENSOR = "tensor"
    PIPELINE = "pipeline"
    TIED_EMBEDDING = "tied_embedding"
