"""
Sequence Models package for Phase 4.
Exports StudentGRU, StudentLSTM, StudentTCN, StudentTransformer, and build_model factory.
"""

import inspect
from src.seq.models.gru import StudentGRU
from src.seq.models.lstm import StudentLSTM
from src.seq.models.tcn import StudentTCN
from src.seq.models.transformer import StudentTransformer


def _filter_kwargs(cls, kwargs):
    sig = inspect.signature(cls.__init__)
    valid_params = set(sig.parameters.keys()) - {"self"}
    filtered = {k: v for k, v in kwargs.items() if k in valid_params}
    # Map common aliases: hidden_size -> channels for TCN; channels -> hidden_size for RNNs
    if "hidden_size" in kwargs and "channels" in valid_params and "channels" not in filtered:
        filtered["channels"] = kwargs["hidden_size"]
    if "channels" in kwargs and "hidden_size" in valid_params and "hidden_size" not in filtered:
        filtered["hidden_size"] = kwargs["channels"]
    return filtered


def build_model(model_name: str, **kwargs):
    name = model_name.lower()
    if name == "gru":
        return StudentGRU(**_filter_kwargs(StudentGRU, kwargs))
    elif name == "lstm":
        return StudentLSTM(**_filter_kwargs(StudentLSTM, kwargs))
    elif name in {"tcn", "cnn", "1dcnn"}:
        return StudentTCN(**_filter_kwargs(StudentTCN, kwargs))
    elif name in {"transformer", "trans"}:
        return StudentTransformer(**_filter_kwargs(StudentTransformer, kwargs))
    else:
        raise ValueError(f"Unknown model name: '{model_name}'. Choose from 'gru', 'lstm', 'tcn', 'transformer'.")


__all__ = [
    "StudentGRU",
    "StudentLSTM",
    "StudentTCN",
    "StudentTransformer",
    "build_model",
]
