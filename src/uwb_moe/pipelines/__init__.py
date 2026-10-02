"""Training and evaluation execution pipelines."""

__all__ = [
    "Trainer",
    "train_pipeline",
    "Evaluator",
    "evaluate_pipeline",
]


def __getattr__(name: str):
    if name in ("Trainer", "train_pipeline"):
        from uwb_moe.pipelines.train import Trainer, train_pipeline
        return locals()[name]
    if name in ("Evaluator", "evaluate_pipeline"):
        from uwb_moe.pipelines.evaluate import Evaluator, evaluate_pipeline
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

