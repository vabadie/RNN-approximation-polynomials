r"""NumPy implementation of Definition \label{def:spreadInput}."""

from __future__ import annotations

import numpy as np

from .elman_rnn import Tensor


class SpreadInputOperator:
    r"""Implements (\spreadInputOp x)[t] = x * 1_{t=0} on finite horizons."""

    def __call__(self, x: Tensor, T: int) -> Tensor:
        return self.forward(x, T)

    def forward(self, x: Tensor, T: int) -> Tensor:
        """Create the terms 0 to T of spread input sequence."""
        if T < 0:
            raise ValueError(f"T must be >= 0, got T={T}.")

        x = np.asarray(x)
        if x.ndim == 1:
            d = x.shape[0]
            seq = np.zeros((T + 1, d), dtype=x.dtype)
            seq[0] = x
            return seq

        if x.ndim == 2:
            batch_size, d = x.shape
            seq = np.zeros((batch_size, T + 1, d), dtype=x.dtype)
            seq[:, 0, :] = x
            return seq

        raise ValueError("x must have shape (d,) or (B, d).")
