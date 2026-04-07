r"""Polynomial map utilities from the paper.

- Definition \label{def:mappings-fl}
- Lemma \label{lm:polyMap}
"""

from __future__ import annotations

import numpy as np

from .elman_rnn import Tensor


def poly_map(level: int, x: Tensor) -> Tensor:
    r"""Apply the map f_level from Definition \label{def:mappings-fl}."""
    if level < 1:
        raise ValueError(f"level must be >= 1, got {level}.")

    x = np.asarray(x)
    if level == 1:
        if x.ndim == 0:
            x_work = x[None]
        elif x.ndim >= 1 and x.shape[-1] == 1:
            x_work = np.squeeze(x, axis=-1)
        else:
            x_work = x
        return np.stack([x_work**2, x_work], axis=-1)

    if x.ndim < 1:
        raise ValueError("For level >= 2, x must have at least one dimension.")

    m = 2 ** (level - 2)
    n_in = m + 1
    if x.shape[-1] != n_in:
        raise ValueError(
            f"poly_map level={level} expects last dim {n_in}, got {x.shape[-1]}."
        )

    special = x[..., m - 1] * x[..., m]
    squares = x[..., :m] ** 2
    adjacent = x[..., : m - 1] * x[..., 1:m]
    identity = x[..., m]

    y = np.empty((*x.shape[:-1], 2 * m + 1), dtype=x.dtype)
    y[..., 0] = special
    y[..., 1 : 2 * m : 2] = squares
    if m > 1:
        y[..., 2 : 2 * m - 1 : 2] = adjacent
    y[..., 2 * m] = identity
    return y


def poly_map_compose(level: int, x: Tensor) -> Tensor:
    r"""Compute f_level ∘ ... ∘ f_1 at x."""
    if level < 1:
        raise ValueError(f"level must be >= 1, got {level}.")

    z = np.asarray(x)
    for ell in range(1, level + 1):
        z = poly_map(ell, z)
    return z


def poly_map_compose_closed_form(level: int, x: Tensor) -> Tensor:
    r"""Closed form from Lemma \label{lm:polyMap} for f_level ∘ ... ∘ f_1."""
    if level < 1:
        raise ValueError(f"level must be >= 1, got {level}.")

    x = np.asarray(x)
    if x.ndim == 0:
        x_work = x[None]
    elif x.ndim >= 1 and x.shape[-1] == 1:
        x_work = np.squeeze(x, axis=-1)
    else:
        x_work = x

    start = 2 ** (level - 1) + 1
    end = 2**level
    monoms = [x_work**p for p in range(start, end + 1)]
    monoms.append(x_work)
    return np.stack(monoms, axis=-1)
