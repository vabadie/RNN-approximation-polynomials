r"""Implementations of RNN transformation lemmas.

- Lemma 9 (parallel RNN construction)
- Lemma 10 (linear map on RNN input)
- Lemma 11 (affine map on RNN output)
- Definition 14 (concatenation of two RNNs)
- Lemma 16 (recursive tree construction for concatenation of multiple RNNs)
- Lemma \label{lem:outSmoothing}
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .elman_rnn import ElmanRNN, Tensor


def _common_dtype(*arrays: Tensor) -> np.dtype:
    return np.result_type(*arrays)


def _block_diag(blocks: Sequence[Tensor]) -> Tensor:
    rows = sum(block.shape[0] for block in blocks)
    cols = sum(block.shape[1] for block in blocks)
    dtype = _common_dtype(*blocks)
    out = np.zeros((rows, cols), dtype=dtype)
    row = 0
    col = 0
    for block in blocks:
        r, c = block.shape
        out[row : row + r, col : col + c] = block
        row += r
        col += c
    return out


def parallel_rnn_from_list(rnns: Sequence[ElmanRNN]) -> ElmanRNN:
    """Construct the block-diagonal parallel RNN from Lemma 9."""
    if len(rnns) == 0:
        raise ValueError("rnns must contain at least one ElmanRNN.")

    A_h = _block_diag([rnn.weights.A_h for rnn in rnns])
    A_x = _block_diag([rnn.weights.A_x for rnn in rnns])
    A_o = _block_diag([rnn.weights.A_o for rnn in rnns])
    b_h = np.concatenate([rnn.weights.b_h for rnn in rnns], axis=0)
    b_o = np.concatenate([rnn.weights.b_o for rnn in rnns], axis=0)

    return ElmanRNN(A_h=A_h, A_x=A_x, A_o=A_o, b_h=b_h, b_o=b_o)


def linear_map_rnn_input(rnn: ElmanRNN, A: Tensor) -> ElmanRNN:
    r"""Lemma 10: absorb input linear map into A_x."""
    A = np.asarray(A, dtype=rnn.weights.A_x.dtype)
    if A.ndim != 2:
        raise ValueError("A must be a matrix of shape (d, d').")
    d, _ = A.shape
    if d != rnn.size.input_dim:
        raise ValueError(
            f"A has incompatible first dimension {d}; expected {rnn.size.input_dim}."
        )

    return ElmanRNN(
        A_h=rnn.weights.A_h,
        A_x=rnn.weights.A_x @ A,
        A_o=rnn.weights.A_o,
        b_h=rnn.weights.b_h,
        b_o=rnn.weights.b_o,
    )


def linear_map_rnn_output(rnn: ElmanRNN, A: Tensor, b: Tensor) -> ElmanRNN:
    r"""Lemma 11: absorb affine output map."""
    A = np.asarray(A, dtype=rnn.weights.A_o.dtype)
    b = np.asarray(b, dtype=rnn.weights.A_o.dtype)
    if A.ndim != 2:
        raise ValueError("A must be a matrix of shape (d', d).")
    if b.ndim != 1:
        raise ValueError("b must be a vector of shape (d',).")
    d_prime, d = A.shape
    if d != rnn.size.output_dim:
        raise ValueError(
            f"A has incompatible second dimension {d}; expected {rnn.size.output_dim}."
        )
    if b.shape[0] != d_prime:
        raise ValueError("b must have first dimension matching A rows (d').")

    return ElmanRNN(
        A_h=rnn.weights.A_h,
        A_x=rnn.weights.A_x,
        A_o=A @ rnn.weights.A_o,
        b_h=rnn.weights.b_h,
        b_o=A @ rnn.weights.b_o + b,
    )


def _switch_net_weights(*, dtype: np.dtype | type) -> tuple[Tensor, Tensor]:
    """Return (switchNetA, switchNetB) from Lemma 13."""
    switch_A = np.array(
        [
            [-4.0, 2.0, 0.0, 0.0, 0.0],
            [-4.0, 2.0, 0.0, 2.0, -0.5],
            [0.0, 0.0, 0.5, 0.0, -1.0],
            [0.0, 1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=dtype,
    )
    switch_B = np.array([-1.0, 0.5, 1.0, -2.0, 1.0], dtype=dtype)
    return switch_A, switch_B


def concat_rnn(
    rnn_g: ElmanRNN,
    rnn_f: ElmanRNN,
    *,
    bound_out_f: float,
    bound_hid_g: float,
) -> tuple[ElmanRNN, Tensor, Tensor]:
    r"""Definition 14."""
    if bound_out_f <= 0 or bound_hid_g <= 0:
        raise ValueError("bound_out_f and bound_hid_g must be > 0.")
    if rnn_g.size.input_dim != rnn_f.size.output_dim:
        raise ValueError(
            "Size mismatch: rnn_g input dimension must equal rnn_f output dimension."
        )

    m_f = rnn_f.size.hidden_dim
    o_f = rnn_f.size.output_dim
    d_f = rnn_f.size.input_dim
    m_g = rnn_g.size.hidden_dim
    o_g = rnn_g.size.output_dim

    Ahf, Axf, Aof, bhf, bof = (
        rnn_f.weights.A_h,
        rnn_f.weights.A_x,
        rnn_f.weights.A_o,
        rnn_f.weights.b_h,
        rnn_f.weights.b_o,
    )
    Ahg, Axg, Aog, bhg, bog = (
        rnn_g.weights.A_h,
        rnn_g.weights.A_x,
        rnn_g.weights.A_o,
        rnn_g.weights.b_h,
        rnn_g.weights.b_o,
    )

    dtype = _common_dtype(Ahf, Axf, Aof, bhf, bof, Ahg, Axg, Aog, bhg, bog)
    switch_A, switch_B = _switch_net_weights(dtype=dtype)

    M = m_f + o_f + o_f + m_g + 5
    A_h = np.zeros((M, M), dtype=dtype)
    b_h = np.zeros((M,), dtype=dtype)
    A_x = np.zeros((M, d_f), dtype=dtype)
    A_o = np.zeros((o_g, M), dtype=dtype)

    s_hf = slice(0, m_f)
    s_up = slice(m_f, m_f + o_f)
    s_um = slice(m_f + o_f, m_f + 2 * o_f)
    s_hg = slice(m_f + 2 * o_f, m_f + 2 * o_f + m_g)
    s_sw = slice(m_f + 2 * o_f + m_g, M)

    A_h[s_hf, s_hf] = Ahf
    A_h[s_up, s_hf] = Aof
    A_h[s_um, s_hf] = -Aof
    A_h[s_hg, s_up] = Axg
    A_h[s_hg, s_um] = -Axg
    A_h[s_hg, s_hg] = Ahg
    A_h[s_sw, s_sw] = switch_A

    switch_readout = np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=dtype)
    A_h[s_up, s_sw] = bound_out_f * np.ones((o_f, 1), dtype=dtype) @ switch_readout[None, :]
    A_h[s_um, s_sw] = bound_out_f * np.ones((o_f, 1), dtype=dtype) @ switch_readout[None, :]
    A_h[s_hg, s_sw] = -bound_hid_g * np.ones((m_g, 1), dtype=dtype) @ switch_readout[None, :]

    b_h[s_hf] = bhf
    b_h[s_up] = bof - bound_out_f * np.ones((o_f,), dtype=dtype)
    b_h[s_um] = -bof - bound_out_f * np.ones((o_f,), dtype=dtype)
    b_h[s_hg] = bhg
    b_h[s_sw] = switch_B

    A_x[s_hf, :] = Axf
    A_o[:, s_hg] = Aog
    b_o = bog

    rnn_concat = ElmanRNN(A_h=A_h, A_x=A_x, A_o=A_o, b_h=b_h, b_o=b_o)

    submat_in = np.zeros((m_f, M), dtype=dtype)
    submat_in[:, s_hf] = np.eye(m_f, dtype=dtype)

    submat_out = np.zeros((m_g, M), dtype=dtype)
    submat_out[:, s_hg] = np.eye(m_g, dtype=dtype)

    return rnn_concat, submat_in, submat_out


def multiconcat_rnn_tree(
    rnns: Sequence[ElmanRNN],
    *,
    hid_bound: float,
) -> tuple[ElmanRNN, list[Tensor], list[Tensor]]:
    r"""Recursive construction from Lemma 16."""
    n = len(rnns)
    if n == 0:
        raise ValueError("rnns must contain at least one RNN.")
    if hid_bound <= 0:
        raise ValueError("hid_bound must be > 0.")
    if (n & (n - 1)) != 0:
        raise ValueError(f"Number of RNNs must be a power of two, got n={n}.")

    for i in range(n - 1):
        if rnns[i + 1].size.input_dim != rnns[i].size.output_dim:
            raise ValueError(
                "Size mismatch in chain at indices "
                f"{i+1}->{i+2}: input_dim(next)={rnns[i+1].size.input_dim} "
                f"!= output_dim(curr)={rnns[i].size.output_dim}."
            )

    if n == 1:
        r = rnns[0]
        return r, [r.weights.A_o], [r.weights.b_o]

    if n == 2:
        r1, r2 = rnns[0], rnns[1]
        r_tree, submat_in, submat_out = concat_rnn(
            r2, r1, bound_out_f=hid_bound, bound_hid_g=hid_bound
        )
        A1 = r1.weights.A_o @ submat_in
        b1 = r1.weights.b_o
        A2 = r2.weights.A_o @ submat_out
        b2 = r2.weights.b_o
        return r_tree, [A1, A2], [b1, b2]

    half = n // 2
    r_a, A_a, b_a = multiconcat_rnn_tree(rnns[:half], hid_bound=hid_bound)
    r_b, A_b, b_b = multiconcat_rnn_tree(rnns[half:], hid_bound=hid_bound)

    r_tree, submat_in, submat_out = concat_rnn(
        r_b, r_a, bound_out_f=hid_bound, bound_hid_g=hid_bound
    )

    A_list: list[Tensor] = []
    b_list: list[Tensor] = []
    for A_ell, b_ell in zip(A_a, b_a):
        A_list.append(A_ell @ submat_in)
        b_list.append(b_ell)
    for A_ell, b_ell in zip(A_b, b_b):
        A_list.append(A_ell @ submat_out)
        b_list.append(b_ell)

    return r_tree, A_list, b_list


def _zero_rnn_for_transformations(
    input_dim: int,
    *,
    dtype: np.dtype | type,
) -> ElmanRNN:
    """Dummy zero RNN from Corollary 17."""
    return ElmanRNN(
        A_h=np.zeros((1, 1), dtype=dtype),
        A_x=np.zeros((1, input_dim), dtype=dtype),
        A_o=np.zeros((1, 1), dtype=dtype),
        b_h=np.zeros((1,), dtype=dtype),
        b_o=np.zeros((1,), dtype=dtype),
    )


def multiconcat_rnn(
    rnns: Sequence[ElmanRNN],
    *,
    hid_bound: float,
) -> tuple[ElmanRNN, list[Tensor], list[Tensor]]:
    r"""Corollary 17 via padding + Lemma tree construction."""
    L = len(rnns)
    if L == 0:
        raise ValueError("rnns must contain at least one RNN.")
    if hid_bound <= 0:
        raise ValueError("hid_bound must be > 0.")

    for i in range(L - 1):
        if rnns[i + 1].size.input_dim != rnns[i].size.output_dim:
            raise ValueError(
                "Size mismatch in chain at indices "
                f"{i+1}->{i+2}: input_dim(next)={rnns[i+1].size.input_dim} "
                f"!= output_dim(curr)={rnns[i].size.output_dim}."
            )

    dtype = rnns[0].weights.A_h.dtype
    log_layer = (L - 1).bit_length()
    padded_len = 1 << log_layer
    num_dummy = padded_len - L
    rnns_padded = list(rnns)
    if num_dummy > 0:
        first_dummy_in = rnns[-1].size.output_dim
        rnns_padded.append(_zero_rnn_for_transformations(first_dummy_in, dtype=dtype))
        for _ in range(num_dummy - 1):
            rnns_padded.append(_zero_rnn_for_transformations(1, dtype=dtype))

    rnn_tree, A_all, b_all = multiconcat_rnn_tree(rnns_padded, hid_bound=hid_bound)
    return rnn_tree, A_all[:L], b_all[:L]


def clipping_rnn(rnn: ElmanRNN, *, B: float) -> ElmanRNN:
    r"""Construct the RNN from Lemma 31."""
    if B <= 0:
        raise ValueError(f"B must be > 0, got B={B}.")
    if rnn.size.output_dim != 1:
        raise ValueError(
            "Lemma 31 applies to scalar-output RNNs only (output_dim=1)."
        )

    Ah = rnn.weights.A_h
    Ax = rnn.weights.A_x
    bh = rnn.weights.b_h
    Ao = rnn.weights.A_o
    bo = rnn.weights.b_o

    dtype = Ah.dtype
    m = rnn.size.hidden_dim
    d = rnn.size.input_dim

    switch_A, switch_B = _switch_net_weights(dtype=dtype)
    switch_readout = np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=dtype)
    B_t = np.array(float(B), dtype=dtype)

    s1 = slice(0, m)
    s2 = slice(m, m + 2)
    s3 = slice(m + 2, m + 4)
    s4 = slice(m + 4, m + 6)
    s5 = slice(m + 6, m + 11)
    M = m + 11

    A_h_p = np.zeros((M, M), dtype=dtype)
    A_x_p = np.zeros((M, d), dtype=dtype)
    b_h_p = np.zeros((M,), dtype=dtype)

    A_h_p[s1, s1] = Ah
    A_x_p[s1, :] = Ax
    b_h_p[s1] = bh

    A_h_p[s2, s1] = np.concatenate([Ao, -Ao], axis=0)
    A_h_p[s2, s5] = B_t * np.ones((2, 1), dtype=dtype) @ switch_readout[None, :]
    b_h_p[s2] = np.array([bo[0] - B_t, -bo[0] - B_t], dtype=dtype)

    A_h_p[s3, s1] = np.concatenate([Ao, -Ao], axis=0)
    b_h_p[s3] = np.array([bo[0] - B_t, -bo[0] - B_t], dtype=dtype)

    A_h_p[s4, s2] = np.eye(2, dtype=dtype)
    A_h_p[s4, s3] = -np.eye(2, dtype=dtype)
    A_h_p[s4, s4] = np.eye(2, dtype=dtype)
    A_h_p[s4, s5] = -B_t * np.ones((2, 1), dtype=dtype) @ switch_readout[None, :]

    A_h_p[s5, s5] = switch_A
    b_h_p[s5] = switch_B

    A_o_p = np.zeros((1, M), dtype=dtype)
    A_o_p[0, s2] = np.array([1.0, -1.0], dtype=dtype)
    A_o_p[0, s3] = np.array([-1.0, 1.0], dtype=dtype)
    A_o_p[0, s4] = np.array([1.0, -1.0], dtype=dtype)
    b_o_p = np.zeros((1,), dtype=dtype)

    return ElmanRNN(A_h=A_h_p, A_x=A_x_p, A_o=A_o_p, b_h=b_h_p, b_o=b_o_p)
