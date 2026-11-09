"""C-MAPSS acquisition and preparation interfaces."""

from aeromaintain.data.pipeline import (
    DataContract,
    DataPipelineError,
    PrepareResult,
    prepare_dataset,
)

__all__ = [
    "DataContract",
    "DataPipelineError",
    "PrepareResult",
    "prepare_dataset",
]
