r"""Concrete RNN constructions from theorem proofs of the paper.

- Theorem 8 (squaring RNN)
- Theorem 12 (multiplication RNN)
- Lemma 22
- Lemma 23
- Lemma \label{lm:rnn_approx_polymap}
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .elman_rnn import ElmanRNN, RNNWeights, Tensor
from .rnn_transformations import (
    clipping_rnn,
    linear_map_rnn_input,
    linear_map_rnn_output,
    multiconcat_rnn,
    parallel_rnn_from_list,
)


def squaring_rnn(
    D: float,
    *,
    dtype: np.dtype | type = np.float64,
    device: Optional[object] = None,
) -> RNNWeights:
    """Return exact weights from the proof of Theorem 8."""
    del device
    if D < 1:
        raise ValueError(f"Theorem requires D >= 1, got D={D}.")

    half = np.array(0.5, dtype=dtype)
    quarter = np.array(0.25, dtype=dtype)
    D_t = np.array(float(D), dtype=dtype)

    A_h = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, float(half), -1.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, float(half), -1.0, 0.0, -1.0, 0.0],
            [1.0, 1.0, -float(half), 1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, float(quarter), -float(half)],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=dtype,
    )
    b_h = np.array([0.0, 0.0, 0.0, 0.0, 0.0, float(half), 1.0], dtype=dtype)
    A_x = (1.0 / D_t) * np.array(
        [[1.0], [-1.0], [0.0], [0.0], [0.0], [0.0], [0.0]],
        dtype=dtype,
    )
    A_o = (D_t**2) * np.array(
        [[0.0, 0.0, -float(half), 1.0, 1.0, 0.0, 0.0]],
        dtype=dtype,
    )
    b_o = np.array([0.0], dtype=dtype)

    return RNNWeights(A_h=A_h, b_h=b_h, A_x=A_x, A_o=A_o, b_o=b_o)


def squaring_rnn_weights(
    D: float,
    *,
    dtype: np.dtype | type = np.float64,
    device: Optional[object] = None,
) -> ElmanRNN:
    """Instantiate the exact theorem RNN from its proof-defined weights."""
    w = squaring_rnn(D, dtype=dtype, device=device)
    return ElmanRNN(A_h=w.A_h, A_x=w.A_x, A_o=w.A_o, b_h=w.b_h, b_o=w.b_o)


def square_rnn_from_theorem(
    D: float,
    *,
    dtype: np.dtype | type = np.float64,
    device: Optional[object] = None,
) -> ElmanRNN:
    """Backward-compatible alias for the theorem-defined squaring RNN constructor."""
    return squaring_rnn_weights(D, dtype=dtype, device=device)


def multiplication_rnn_from_theorem(
    D: float,
    *,
    dtype: np.dtype | type = np.float64,
    device: Optional[object] = None,
) -> ElmanRNN:
    r"""Instantiate the RNN from Theorem \label{thm:multiplication_net}."""
    del device
    if D < 1:
        raise ValueError(f"Theorem requires D >= 1, got D={D}.")

    square1 = squaring_rnn_weights(D, dtype=dtype)
    square2 = squaring_rnn_weights(D, dtype=dtype)
    rnn_parallel = parallel_rnn_from_list([square1, square2])

    A_in = 0.5 * np.array([[1.0, 1.0], [1.0, -1.0]], dtype=dtype)
    rnn_tilde = linear_map_rnn_input(rnn_parallel, A_in)

    A_out = np.array([[1.0, -1.0]], dtype=dtype)
    b_out = np.array([0.0], dtype=dtype)
    return linear_map_rnn_output(rnn_tilde, A_out, b_out)


def zero_rnn_from_definition(
    input_dim: int,
    *,
    dtype: np.dtype | type = np.float64,
    device: Optional[object] = None,
) -> ElmanRNN:
    r"""Dummy RNN from the proof of Corollary 17."""
    del device
    if input_dim <= 0:
        raise ValueError(f"input_dim must be positive, got {input_dim}.")

    return ElmanRNN(
        A_h=np.zeros((1, 1), dtype=dtype),
        A_x=np.zeros((1, input_dim), dtype=dtype),
        b_h=np.zeros((1,), dtype=dtype),
        A_o=np.zeros((1, 1), dtype=dtype),
        b_o=np.zeros((1,), dtype=dtype),
    )


def identity_rnn(
    *,
    dtype: np.dtype | type = np.float64,
    device: Optional[object] = None,
) -> ElmanRNN:
    r"""RNN from Lemma 2."""
    del device
    return ElmanRNN(
        A_h=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=dtype),
        A_x=np.array([[1.0], [-1.0]], dtype=dtype),
        b_h=np.zeros((2,), dtype=dtype),
        A_o=np.array([[1.0, -1.0]], dtype=dtype),
        b_o=np.zeros((1,), dtype=dtype),
    )


def square_and_identity_rnn(
    D: float,
    *,
    dtype: np.dtype | type = np.float64,
    device: Optional[object] = None,
) -> ElmanRNN:
    r"""RNN from Lemma 23."""
    del device
    if D < 1:
        raise ValueError(f"Lemma requires D >= 1, got D={D}.")

    rnn_square = squaring_rnn_weights(D, dtype=dtype)
    rnn_identity = identity_rnn(dtype=dtype)
    rnn_parallel = parallel_rnn_from_list([rnn_square, rnn_identity])
    A_in = np.array([[1.0], [1.0]], dtype=dtype)
    return linear_map_rnn_input(rnn_parallel, A_in)


def _selector_matrix_for_polymap_level(
    level: int,
    *,
    dtype: np.dtype | type,
    device: Optional[object],
) -> Tensor:
    """Selector matrix A used in the proof of Lemma 24."""
    del device
    if level < 2:
        raise ValueError(f"level must be >= 2, got {level}.")

    m = 2 ** (level - 2)
    n_in = m + 1
    n_out = 3 * m + 1
    A = np.zeros((n_out, n_in), dtype=dtype)

    row = 0
    A[row, m - 1] = 1.0
    row += 1
    A[row, m] = 1.0
    row += 1

    for k in range(1, m + 1):
        A[row, k - 1] = 1.0
        row += 1
        if k <= m - 1:
            A[row, k - 1] = 1.0
            row += 1
            A[row, k] = 1.0
            row += 1

    A[row, m] = 1.0
    return A


def polymap_rnn(
    level: int,
    D: float,
    *,
    dtype: np.dtype | type = np.float64,
    device: Optional[object] = None,
) -> ElmanRNN:
    r"""Construct RNN \mathcal{R}_D^\ell from Lemma 24."""
    del device
    if level < 2:
        raise ValueError(f"Lemma applies to level >= 2, got level={level}.")
    if D < 1:
        raise ValueError(f"Lemma requires D >= 1, got D={D}.")

    D_level = D ** (2 ** (level - 1))
    rnn_mult = multiplication_rnn_from_theorem(D_level, dtype=dtype)
    rnn_square = squaring_rnn_weights(D_level, dtype=dtype)
    rnn_id = identity_rnn(dtype=dtype)

    m = 2 ** (level - 2)
    rnns = [rnn_mult]
    for idx in range(1, m + 1):
        rnns.append(rnn_square)
        if idx <= m - 1:
            rnns.append(rnn_mult)
    rnns.append(rnn_id)

    rnn_tmp = parallel_rnn_from_list(rnns)
    A_in = _selector_matrix_for_polymap_level(level, dtype=dtype, device=None)
    return linear_map_rnn_input(rnn_tmp, A_in)


def _build_powers_hidden_system(
    L: int,
    D: float,
    *,
    hid_bound: Optional[float] = None,
    dtype: np.dtype | type = np.float64,
    device: Optional[object] = None,
) -> tuple[object, ElmanRNN, list[Tensor], list[Tensor]]:
    """Internal builder for Definition 25 and dependents."""
    del device
    if L < 1:
        raise ValueError(f"L must be >= 1, got L={L}.")
    if D < 1:
        raise ValueError(f"D must be >= 1, got D={D}.")

    rnns = [square_and_identity_rnn(D, dtype=dtype)]
    for ell in range(2, L + 1):
        rnns.append(polymap_rnn(ell, D, dtype=dtype))

    if hid_bound is None:
        hid_bound = max(2.0, float(D) ** (2**L))

    rnn_full, A_list, b_list = multiconcat_rnn(rnns, hid_bound=hid_bound)
    hid_op = rnn_full.hidden_operator
    return hid_op, rnn_full, A_list, b_list


def powers_hidden_operator(
    L: int,
    D: float,
    *,
    hid_bound: Optional[float] = None,
    dtype: np.dtype | type = np.float64,
    device: Optional[object] = None,
):
    r"""Definition 25."""
    hid_op, _, _, _ = _build_powers_hidden_system(
        L=L,
        D=D,
        hid_bound=hid_bound,
        dtype=dtype,
        device=device,
    )
    return hid_op


def powers_output_mapping(
    L: int,
    D: float,
    *,
    hid_bound: Optional[float] = None,
    dtype: np.dtype | type = np.float64,
    device: Optional[object] = None,
) -> tuple[Tensor, Tensor]:
    r"""Construct (A_o^\pi, b_o^\pi) from Lemma 29."""
    del device
    if L < 1:
        raise ValueError(f"L must be >= 1, got L={L}.")
    if D < 1:
        raise ValueError(f"D must be >= 1, got D={D}.")

    _, _, A_list, b_list = _build_powers_hidden_system(
        L=L,
        D=D,
        hid_bound=hid_bound,
        dtype=dtype,
    )

    P = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=dtype)
    Ao_blocks: list[Tensor] = [P @ A_list[0]]
    bo_blocks: list[Tensor] = [P @ b_list[0]]

    for ell in range(2, L + 1):
        rows_keep = 2 ** (ell - 1)
        S = np.concatenate(
            [
                np.eye(rows_keep, dtype=dtype),
                np.zeros((rows_keep, 1), dtype=dtype),
            ],
            axis=1,
        )
        Ao_blocks.append(S @ A_list[ell - 1])
        bo_blocks.append(S @ b_list[ell - 1])

    return np.concatenate(Ao_blocks, axis=0), np.concatenate(bo_blocks, axis=0)


def final_powers_rnn(
    L: int,
    D: float,
    *,
    hid_bound: Optional[float] = None,
    dtype: np.dtype | type = np.float64,
    device: Optional[object] = None,
    return_aux: bool = False,
):
    r"""Construct RNN \mathcal{R}_{D,L}^{\pi} from Theorem 30."""
    del device
    if L < 1:
        raise ValueError(f"L must be >= 1, got L={L}.")
    if D < 1:
        raise ValueError(f"D must be >= 1, got D={D}.")

    hid_op, rnn_hidden, A_list, b_list = _build_powers_hidden_system(
        L=L,
        D=D,
        hid_bound=hid_bound,
        dtype=dtype,
    )
    A_o_pi, b_o_pi = powers_output_mapping(
        L=L,
        D=D,
        hid_bound=hid_bound,
        dtype=dtype,
    )

    rnn_final = ElmanRNN(
        A_h=rnn_hidden.weights.A_h,
        A_x=rnn_hidden.weights.A_x,
        A_o=A_o_pi,
        b_h=rnn_hidden.weights.b_h,
        b_o=b_o_pi,
    )

    if return_aux:
        return rnn_final, hid_op, A_list, b_list, A_o_pi, b_o_pi
    return rnn_final


def polynomial_rnn(
    coefficients: Tensor | list[float],
    D: float,
    *,
    clip_B: Optional[float] = None,
    hid_bound: Optional[float] = None,
    dtype: np.dtype | type = np.float64,
    device: Optional[object] = None,
    return_aux: bool = False,
):
    r"""Construct \rnnPolynom from Theorem 33."""
    del device
    if D < 1:
        raise ValueError(f"D must be >= 1, got D={D}.")

    coeff = np.asarray(coefficients, dtype=dtype).reshape(-1)
    if coeff.size == 0:
        raise ValueError("coefficients must contain at least one element [a0].")
    N = coeff.size - 1

    if N <= 1:
        rnn_id = identity_rnn(dtype=dtype)
        a1 = coeff[1] if N == 1 else np.array(0.0, dtype=dtype)
        A = np.array([[float(a1)]], dtype=dtype)
        b = np.array([float(coeff[0])], dtype=dtype)
        rnn_exact = linear_map_rnn_output(rnn_id, A, b)
        if return_aux:
            return rnn_exact, rnn_exact, float(abs(coeff[0]))
        return rnn_exact

    L = int(math.ceil(math.log2(N)))
    n_pow = 2**L

    a_ext = np.zeros((n_pow,), dtype=dtype)
    a_ext[:N] = coeff[1:]
    a0 = coeff[0]

    rnn_powers = final_powers_rnn(L=L, D=D, hid_bound=hid_bound, dtype=dtype)
    A = a_ext[None, :]
    b = np.array([float(a0)], dtype=dtype)
    rnn_tilde = linear_map_rnn_output(rnn_powers, A, b)

    if clip_B is None:
        powers = np.arange(0, N + 1, dtype=np.int64)
        B = float(np.sum(np.abs(coeff) * (float(D) ** powers)))
    else:
        B = float(clip_B)
    if B <= 0:
        raise ValueError(f"clip_B must be > 0, got {B}.")

    rnn_poly = clipping_rnn(rnn_tilde, B=B)
    if return_aux:
        return rnn_poly, rnn_tilde, B
    return rnn_poly
