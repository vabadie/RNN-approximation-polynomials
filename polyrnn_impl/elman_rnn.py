r"""NumPy implementation of Definition \label{def:elman_rnn}.

Implements:
- Hidden state operator  H
- Output mapping        O
- RNN operator          R = O \circ H
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

Tensor = np.ndarray


def _asarray(x: Tensor | list[float], *, dtype: np.dtype | type | None = None) -> Tensor:
    """Return a detached NumPy array with an optional dtype override."""
    if isinstance(x, np.ndarray):
        if dtype is None:
            return np.array(x, copy=True)
        return np.array(x, dtype=dtype, copy=True)
    return np.array(x, dtype=dtype if dtype is not None else np.float64, copy=True)


@dataclass(frozen=True)
class RNNSize:
    """Size tuple (d, d', m) used in the paper."""

    input_dim: int
    output_dim: int
    hidden_dim: int


@dataclass(frozen=True)
class RNNWeights:
    """Container for the Elman RNN weights (A_h, b_h, A_x, A_o, b_o)."""

    A_h: Tensor
    b_h: Tensor
    A_x: Tensor
    A_o: Tensor
    b_o: Tensor

    @staticmethod
    def _format_tensor(name: str, tensor: Tensor) -> str:
        matrix_str = str(tensor)
        meta = f"shape={tensor.shape}, dtype={tensor.dtype}"
        return f"{name}:\n{matrix_str}\n  [{meta}]"

    def __repr__(self) -> str:
        sections = [
            "RNNWeights(",
            self._format_tensor("A_h", self.A_h),
            self._format_tensor("b_h", self.b_h),
            self._format_tensor("A_x", self.A_x),
            self._format_tensor("A_o", self.A_o),
            self._format_tensor("b_o", self.b_o),
            ")",
        ]
        return "\n".join(sections)


class HiddenStateOperator:
    r"""Hidden state operator H from Definition \label{def:elman_rnn}."""

    def __init__(
        self,
        A_h: Tensor,
        A_x: Tensor,
        b_h: Tensor,
    ) -> None:
        A_h = _asarray(A_h)
        A_x = _asarray(A_x, dtype=A_h.dtype)
        b_h = _asarray(b_h, dtype=A_h.dtype)

        if A_h.ndim != 2 or A_x.ndim != 2 or b_h.ndim != 1:
            raise ValueError("A_h, A_x must be matrices and b_h a vector.")

        hidden_dim_h0, hidden_dim_h1 = A_h.shape
        hidden_dim_x, input_dim = A_x.shape
        if hidden_dim_h0 != hidden_dim_h1:
            raise ValueError("A_h must be square with shape (m, m).")
        if hidden_dim_x != hidden_dim_h0:
            raise ValueError("A_x must have shape (m, d) with same m as A_h.")
        if b_h.shape[0] != hidden_dim_h0:
            raise ValueError("b_h must have shape (m,).")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim_h0
        self.A_h = A_h
        self.A_x = A_x
        self.b_h = b_h

    def __call__(self, x_seq: Tensor, h_init: Optional[Tensor] = None) -> Tensor:
        return self.forward(x_seq, h_init=h_init)

    def step(self, h_prev: Tensor, x_t: Tensor) -> Tensor:
        """Single update h[t] = ReLU(A_h h[t-1] + A_x x[t] + b_h)."""
        return np.maximum(h_prev @ self.A_h.T + x_t @ self.A_x.T + self.b_h, 0.0)

    def forward(self, x_seq: Tensor, h_init: Optional[Tensor] = None) -> Tensor:
        """Apply H to an input sequence.

        Args:
            x_seq: Array of shape (T, d) or (B, T, d)
            h_init: Optional initial hidden state with shape (m,) or (B, m).
                    If omitted, uses h[-1] = 0 as in the paper.

        Returns:
            Hidden-state sequence with shape (T, m) or (B, T, m).
        """
        x_seq = _asarray(x_seq, dtype=self.A_h.dtype)
        added_batch = False
        if x_seq.ndim == 2:
            x_seq = x_seq[None, :, :]
            added_batch = True

        if x_seq.ndim != 3:
            raise ValueError("x_seq must have shape (T, d) or (B, T, d).")

        batch_size, horizon, input_dim = x_seq.shape
        if input_dim != self.input_dim:
            raise ValueError(f"Expected input dimension d={self.input_dim}, got d={input_dim}.")

        if h_init is None:
            h_t = np.zeros((batch_size, self.hidden_dim), dtype=x_seq.dtype)
        else:
            h_init = _asarray(h_init, dtype=x_seq.dtype)
            if h_init.ndim == 1:
                h_t = np.broadcast_to(h_init[None, :], (batch_size, self.hidden_dim)).copy()
            elif h_init.ndim == 2 and h_init.shape[0] == batch_size:
                h_t = h_init.copy()
            else:
                raise ValueError("h_init must have shape (m,) or (B, m).")

        h_list = []
        for t in range(horizon):
            h_t = self.step(h_t, x_seq[:, t, :])
            h_list.append(h_t)

        h_seq = np.stack(h_list, axis=1)
        if added_batch:
            h_seq = h_seq[0]
        return h_seq


class OutputMapping:
    r"""Output map O(h) = A_o h + b_o from Definition \label{def:elman_rnn}."""

    def __init__(self, A_o: Tensor, b_o: Tensor) -> None:
        A_o = _asarray(A_o)
        b_o = _asarray(b_o, dtype=A_o.dtype)
        if A_o.ndim != 2 or b_o.ndim != 1:
            raise ValueError("A_o must be a matrix and b_o a vector.")
        out_dim, hidden_dim = A_o.shape
        if b_o.shape[0] != out_dim:
            raise ValueError("b_o must have shape (d',).")

        self.output_dim = out_dim
        self.hidden_dim = hidden_dim
        self.A_o = A_o
        self.b_o = b_o

    def __call__(self, h: Tensor) -> Tensor:
        return self.forward(h)

    def forward(self, h: Tensor) -> Tensor:
        """Map h of shape (..., m) to y of shape (..., d')."""
        h = _asarray(h, dtype=self.A_o.dtype)
        if h.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"Expected hidden dimension m={self.hidden_dim}, got m={h.shape[-1]}."
            )
        return h @ self.A_o.T + self.b_o


class ElmanRNN:
    """RNN operator R = O H with per-time-step output as in the paper."""

    def __init__(
        self,
        A_h: Tensor,
        A_x: Tensor,
        A_o: Tensor,
        b_h: Tensor,
        b_o: Tensor,
    ) -> None:
        self.hidden_operator = HiddenStateOperator(A_h=A_h, A_x=A_x, b_h=b_h)
        self.output_mapping = OutputMapping(A_o=A_o, b_o=b_o)

        if self.output_mapping.hidden_dim != self.hidden_operator.hidden_dim:
            raise ValueError("A_o must have hidden dimension m matching A_h/A_x/b_h.")

        self.size = RNNSize(
            input_dim=self.hidden_operator.input_dim,
            output_dim=self.output_mapping.output_dim,
            hidden_dim=self.hidden_operator.hidden_dim,
        )

        self.weights = RNNWeights(
            A_h=self.hidden_operator.A_h,
            b_h=self.hidden_operator.b_h,
            A_x=self.hidden_operator.A_x,
            A_o=self.output_mapping.A_o,
            b_o=self.output_mapping.b_o,
        )

    def __call__(self, x_seq: Tensor, h_init: Optional[Tensor] = None) -> Tensor:
        return self.forward(x_seq, h_init=h_init)

    def forward(self, x_seq: Tensor, h_init: Optional[Tensor] = None) -> Tensor:
        """Compute (R x)[t] = O((H x)[t]) for all t."""
        h_seq = self.hidden_operator(x_seq, h_init=h_init)
        return self.output_mapping(h_seq)
