"""Data loading, preprocessing, and download utilities for UWB datasets."""

__all__ = [
    "UWBDataProcessor",
    "PreprocessingStats",
    "UWBDataset",
    "create_dataloaders",
    "download_dataset",
]


def __getattr__(name: str):
    if name in ("UWBDataProcessor", "PreprocessingStats"):
        from uwb_moe.data.processor import UWBDataProcessor, PreprocessingStats
        return locals()[name]
    if name in ("UWBDataset", "create_dataloaders"):
        from uwb_moe.data.dataset import UWBDataset, create_dataloaders
        return locals()[name]
    if name == "download_dataset":
        from uwb_moe.data.download import download_dataset
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

