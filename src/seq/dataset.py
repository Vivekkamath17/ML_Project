"""
PyTorch Dataset and DataLoader module for Phase 4 Sequence Models.
Returns: (x, mask, static, y_dropout, y_onset, onset_mask, y_score, score_mask, learner_key)
Optimized with pre-tensorized storage for high-throughput DataLoader iteration.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class StudentSequenceDataset(Dataset):
    """
    Sequence dataset for student trajectory modeling.
    Returns:
      x: torch.FloatTensor of shape (W, F)
      mask: torch.BoolTensor of shape (W,)
      static: torch.FloatTensor of shape (F_static,)
      y_dropout: torch.FloatTensor scalar (0.0 or 1.0)
      y_onset: torch.FloatTensor scalar (0.0 or 1.0)
      onset_mask: torch.BoolTensor scalar
      y_score: torch.FloatTensor scalar (final_score)
      score_mask: torch.BoolTensor scalar
      learner_key: str
    """

    def __init__(self, data_dict: Dict[str, Any], indices: Optional[np.ndarray] = None):
        self.indices = indices if indices is not None else np.arange(len(data_dict["X"]))

        # Convert sliced arrays to contiguous PyTorch tensors upfront for zero per-item conversion overhead
        x_sub = np.ascontiguousarray(data_dict["X"][self.indices], dtype=np.float32)
        m_sub = np.ascontiguousarray(data_dict["M"][self.indices], dtype=bool)
        s_sub = np.ascontiguousarray(data_dict["S"][self.indices], dtype=np.float32)
        yd_sub = np.ascontiguousarray(data_dict["y_dropout"][self.indices], dtype=np.float32)
        yo_sub = np.ascontiguousarray(data_dict["y_onset"][self.indices], dtype=np.float32)
        om_sub = np.ascontiguousarray(data_dict["onset_mask"][self.indices], dtype=bool)
        fs_sub = np.ascontiguousarray(data_dict["final_score"][self.indices], dtype=np.float32)
        sm_sub = np.ascontiguousarray(data_dict["score_mask"][self.indices], dtype=bool)

        self.X = torch.from_numpy(x_sub)
        self.M = torch.from_numpy(m_sub)
        self.S = torch.from_numpy(s_sub)
        self.y_dropout = torch.from_numpy(yd_sub)
        self.y_onset = torch.from_numpy(yo_sub)
        self.onset_mask = torch.from_numpy(om_sub)
        self.final_score = torch.from_numpy(fs_sub)
        self.score_mask = torch.from_numpy(sm_sub)

        meta = data_dict["meta"]
        if "learner_key" in meta.columns:
            self.learner_keys = meta.iloc[self.indices]["learner_key"].to_numpy()
        else:
            self.learner_keys = np.array([f"learner_{i}" for i in self.indices])

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[Any, ...]:
        return (
            self.X[idx],
            self.M[idx],
            self.S[idx],
            self.y_dropout[idx],
            self.y_onset[idx],
            self.onset_mask[idx],
            self.final_score[idx],
            self.score_mask[idx],
            self.learner_keys[idx],
        )


def collate_fn(batch: List[Tuple]) -> Dict[str, Any]:
    """Collate sequence batch into stacked tensors."""
    x = torch.stack([item[0] for item in batch])
    mask = torch.stack([item[1] for item in batch])
    static = torch.stack([item[2] for item in batch])
    y_dropout = torch.stack([item[3] for item in batch])
    y_onset = torch.stack([item[4] for item in batch])
    onset_mask = torch.stack([item[5] for item in batch])
    final_score = torch.stack([item[6] for item in batch])
    score_mask = torch.stack([item[7] for item in batch])
    learner_keys = [item[8] for item in batch]

    return {
        "x": x,
        "mask": mask,
        "static": static,
        "y_dropout": y_dropout,
        "y_onset": y_onset,
        "onset_mask": onset_mask,
        "y_score": final_score,
        "score_mask": score_mask,
        "learner_keys": learner_keys,
    }


def create_dataloader(
    dataset: StudentSequenceDataset,
    batch_size: int = 256,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    """Create DataLoader for sequence dataset."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=False,
    )
