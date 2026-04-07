from pathlib import Path
import sys

import numpy as np

# Allow direct execution: `python tests/test_elman_rnn.py`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from polyrnn_impl import (
    ElmanRNN,
    SpreadInputOperator,
    clipping_rnn,
    concat_rnn,
    final_powers_rnn,
    identity_rnn,
    linear_map_rnn_input,
    linear_map_rnn_output,
    multiconcat_rnn,
    multiconcat_rnn_tree,
    multiplication_rnn_from_theorem,
    parallel_rnn_from_list,
    poly_map,
    poly_map_compose,
    poly_map_compose_closed_form,
    polynomial_rnn,
    polymap_rnn,
    powers_hidden_operator,
    powers_output_mapping,
    square_and_identity_rnn,
    square_rnn_from_theorem,
    squaring_rnn_weights,
    zero_rnn_from_definition,
)


def tensor(x, dtype=np.float64):
    return np.array(x, dtype=dtype)


def assert_allclose(a, b, atol=1e-8):
    np.testing.assert_allclose(a, b, atol=atol, rtol=0.0)


def test_elman_rnn_shapes_and_recursion():
    rnn = ElmanRNN(
        A_h=tensor([[0.5]]),
        A_x=tensor([[1.0]]),
        A_o=tensor([[2.0]]),
        b_h=tensor([0.0]),
        b_o=tensor([1.0]),
    )
    y = rnn(tensor([[1.0], [0.0], [0.0]]))
    assert_allclose(y, tensor([[3.0], [2.0], [1.5]]))


def test_weights_container_exposes_all_elman_tensors():
    A_h = tensor([[0.5]])
    A_x = tensor([[1.0]])
    A_o = tensor([[2.0]])
    b_h = tensor([0.0])
    b_o = tensor([1.0])
    rnn = ElmanRNN(A_h=A_h, A_x=A_x, A_o=A_o, b_h=b_h, b_o=b_o)
    assert_allclose(rnn.weights.A_h, A_h)
    assert_allclose(rnn.weights.A_x, A_x)
    assert_allclose(rnn.weights.b_h, b_h)
    assert_allclose(rnn.weights.A_o, A_o)
    assert_allclose(rnn.weights.b_o, b_o)


def test_spread_input_operator_unbatched():
    spread = SpreadInputOperator()
    seq = spread(tensor([2.0, -1.0]), T=4)
    expected = tensor([[2.0, -1.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    assert_allclose(seq, expected)


def test_spread_input_operator_batched():
    spread = SpreadInputOperator()
    seq = spread(tensor([[2.0], [-3.0]]), T=3)
    expected = tensor([[[2.0], [0.0], [0.0], [0.0]], [[-3.0], [0.0], [0.0], [0.0]]])
    assert_allclose(seq, expected)


def test_rnn_with_spread_input_matches_manual_sequence():
    rnn = ElmanRNN(
        A_h=tensor([[0.5]]),
        A_x=tensor([[1.0]]),
        A_o=tensor([[2.0]]),
        b_h=tensor([0.0]),
        b_o=tensor([1.0]),
    )
    spread = SpreadInputOperator()
    y_spread = rnn(spread(tensor([1.25]), T=5))
    y_manual = rnn(tensor([[1.25], [0.0], [0.0], [0.0], [0.0], [0.0]]))
    assert_allclose(y_spread, y_manual)


def test_rnn_weights_repr_is_human_readable():
    rnn = ElmanRNN(
        A_h=tensor([[0.5]]),
        A_x=tensor([[1.0]]),
        A_o=tensor([[2.0]]),
        b_h=tensor([0.0]),
        b_o=tensor([1.0]),
    )
    rep = repr(rnn.weights)
    assert "RNNWeights(" in rep
    assert "A_h:" in rep
    assert "A_x:" in rep
    assert "A_o:" in rep
    assert "b_h:" in rep
    assert "b_o:" in rep
    assert "shape=" in rep
    assert "dtype=" in rep


def test_square_rnn_weights_match_theorem_proof_exactly():
    w = squaring_rnn_weights(2.0)
    assert_allclose(
        w.weights.A_h,
        tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 0.5, -1.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 0.5, -1.0, 0.0, -1.0, 0.0],
                [1.0, 1.0, -0.5, 1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.25, -0.5],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ),
    )
    assert_allclose(w.weights.b_h, tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0]))
    assert_allclose(w.weights.A_x, tensor([[0.5], [-0.5], [0.0], [0.0], [0.0], [0.0], [0.0]]))
    assert_allclose(w.weights.A_o, tensor([[0.0, 0.0, -2.0, 4.0, 4.0, 0.0, 0.0]]))
    assert_allclose(w.weights.b_o, tensor([0.0]))


def test_square_rnn_object_has_expected_sizes():
    rnn = square_rnn_from_theorem(D=1.0)
    assert rnn.size.input_dim == 1
    assert rnn.size.output_dim == 1
    assert rnn.size.hidden_dim == 7


def test_multiplication_rnn_object_has_expected_sizes():
    rnn = multiplication_rnn_from_theorem(D=1.0)
    assert rnn.size.input_dim == 2
    assert rnn.size.output_dim == 1
    assert rnn.size.hidden_dim == 14


def test_parallel_rnn_matches_individual_outputs_on_same_sequence():
    rnn1 = ElmanRNN(
        A_h=tensor([[0.3]]),
        A_x=tensor([[1.0]]),
        A_o=tensor([[2.0]]),
        b_h=tensor([0.1]),
        b_o=tensor([0.0]),
    )
    rnn2 = ElmanRNN(
        A_h=tensor([[0.5, 0.0], [0.1, 0.2]]),
        A_x=tensor([[1.0, -1.0], [0.5, 0.25]]),
        A_o=tensor([[1.0, -0.5]]),
        b_h=tensor([0.0, 0.2]),
        b_o=tensor([1.0]),
    )
    rnn_parallel = parallel_rnn_from_list([rnn1, rnn2])
    x1 = tensor([[1.0], [0.2], [0.0], [0.3]])
    x2 = tensor([[0.1, -0.2], [0.3, 0.4], [0.0, 0.0], [1.0, -1.0]])
    x_cat = np.concatenate([x1, x2], axis=1)
    y_cat_expected = np.concatenate([rnn1(x1), rnn2(x2)], axis=1)
    assert_allclose(rnn_parallel(x_cat), y_cat_expected)


def test_parallel_rnn_matches_hidden_states_blockwise():
    rnn1 = ElmanRNN(tensor([[0.2]]), tensor([[1.0]]), tensor([[1.0]]), tensor([0.0]), tensor([0.0]))
    rnn2 = ElmanRNN(tensor([[0.4]]), tensor([[2.0]]), tensor([[1.0]]), tensor([0.0]), tensor([0.0]))
    rp = parallel_rnn_from_list([rnn1, rnn2])
    x1 = tensor([[0.5], [0.0], [0.0]])
    x2 = tensor([[1.0], [0.0], [0.0]])
    x_cat = np.concatenate([x1, x2], axis=1)
    h_expected = np.concatenate([rnn1.hidden_operator(x1), rnn2.hidden_operator(x2)], axis=1)
    assert_allclose(rp.hidden_operator(x_cat), h_expected)


def test_linear_map_rnn_input_matches_original_on_transformed_input():
    rnn = ElmanRNN(
        tensor([[0.3, 0.0], [0.1, 0.2]]),
        tensor([[1.0, -1.0], [0.5, 0.25]]),
        tensor([[1.0, -0.5]]),
        tensor([0.0, 0.2]),
        tensor([1.0]),
    )
    A = tensor([[1.0, 0.0, 2.0], [0.0, -1.0, 1.0]])
    rnn_prime = linear_map_rnn_input(rnn, A)
    x_prime_seq = tensor([[0.5, -1.0, 2.0], [0.0, 0.0, 0.0], [1.0, 2.0, -1.0]])
    assert_allclose(rnn_prime(x_prime_seq), rnn(x_prime_seq @ A.T))


def test_linear_map_rnn_output_matches_affine_transform():
    rnn = ElmanRNN(tensor([[0.2]]), tensor([[1.0]]), tensor([[1.0], [2.0]]), tensor([0.0]), tensor([0.5, -1.0]))
    A = tensor([[1.0, -1.0], [0.5, 0.25]])
    b = tensor([0.1, -0.2])
    x_seq = tensor([[1.0], [0.0], [0.5]])
    y = rnn(x_seq)
    y_expected = y @ A.T + b
    assert_allclose(linear_map_rnn_output(rnn, A, b)(x_seq), y_expected, atol=1e-7)


def test_multiplication_rnn_basic_numeric_sanity():
    spread = SpreadInputOperator()
    rnn = multiplication_rnn_from_theorem(D=1.0)
    x = tensor([0.6, -0.25])
    y = np.squeeze(rnn(spread(x, T=12)), axis=-1)
    target = x[0] * x[1]
    assert np.isclose(y[-1], target, atol=5e-3)


def test_concat_rnn_has_definition_size_and_runs():
    rnn_f = ElmanRNN(tensor([[0.4]]), tensor([[1.0]]), tensor([[1.0]]), tensor([0.0]), tensor([0.0]))
    rnn_g = ElmanRNN(
        tensor([[0.2, 0.0], [0.1, 0.3]]),
        tensor([[1.0], [0.5]]),
        tensor([[1.0, -0.5]]),
        tensor([0.0, 0.1]),
        tensor([0.0]),
    )
    rnn_c, submat_in, submat_out = concat_rnn(rnn_g, rnn_f, bound_out_f=2.0, bound_hid_g=3.0)
    expected_m = rnn_f.size.hidden_dim + 2 * rnn_f.size.output_dim + rnn_g.size.hidden_dim + 5
    assert rnn_c.size.hidden_dim == expected_m
    assert submat_in.shape == (rnn_f.size.hidden_dim, expected_m)
    assert submat_out.shape == (rnn_g.size.hidden_dim, expected_m)
    y = rnn_c(SpreadInputOperator()(tensor([0.25]), T=6))
    assert y.shape == (7, 1)


def test_concat_rnn_requires_matching_fg_dimensions():
    rnn_f = ElmanRNN(tensor([[0.4]]), tensor([[1.0]]), tensor([[1.0], [2.0]]), tensor([0.0]), tensor([0.0, 0.0]))
    rnn_g = ElmanRNN(tensor([[0.2]]), tensor([[1.0]]), tensor([[1.0]]), tensor([0.0]), tensor([0.0]))
    try:
        concat_rnn(rnn_g, rnn_f, bound_out_f=1.0, bound_hid_g=1.0)
        assert False
    except ValueError:
        pass


def test_multiconcat_rnn_tree_base_case_matches_proof_formula():
    rnn1 = ElmanRNN(tensor([[0.4]]), tensor([[1.0]]), tensor([[1.0]]), tensor([0.0]), tensor([0.3]))
    rnn2 = ElmanRNN(
        tensor([[0.2, 0.0], [0.1, 0.3]]),
        tensor([[1.0], [0.5]]),
        tensor([[1.0, -0.5]]),
        tensor([0.0, 0.1]),
        tensor([0.7]),
    )
    tree, A_list, b_list = multiconcat_rnn_tree([rnn1, rnn2], hid_bound=2.0)
    concat_tree, submat_in, submat_out = concat_rnn(rnn2, rnn1, bound_out_f=2.0, bound_hid_g=2.0)
    assert tree.size.hidden_dim == concat_tree.size.hidden_dim
    assert_allclose(A_list[0], rnn1.weights.A_o @ submat_in)
    assert_allclose(b_list[0], rnn1.weights.b_o)
    assert_allclose(A_list[1], rnn2.weights.A_o @ submat_out)
    assert_allclose(b_list[1], rnn2.weights.b_o)


def test_multiconcat_rnn_tree_recursive_shapes_for_4_rnns():
    rs = [ElmanRNN(tensor([[0.2]]), tensor([[1.0]]), tensor([[1.0]]), tensor([0.0]), tensor([0.0])) for _ in range(4)]
    tree, A_list, b_list = multiconcat_rnn_tree(rs, hid_bound=2.0)
    assert len(A_list) == 4
    assert len(b_list) == 4
    for ell in range(4):
        assert A_list[ell].shape[1] == tree.size.hidden_dim
        assert A_list[ell].shape[0] == rs[ell].size.output_dim
        assert b_list[ell].shape[0] == rs[ell].size.output_dim


def test_multiconcat_rnn_tree_requires_power_of_two():
    rnn = ElmanRNN(tensor([[0.2]]), tensor([[1.0]]), tensor([[1.0]]), tensor([0.0]), tensor([0.0]))
    try:
        multiconcat_rnn_tree([rnn, rnn, rnn], hid_bound=2.0)
        assert False
    except ValueError:
        pass


def test_multiconcat_rnn_corollary_handles_non_power_of_two():
    rs = [ElmanRNN(tensor([[0.2]]), tensor([[1.0]]), tensor([[1.0]]), tensor([0.0]), tensor([0.0])) for _ in range(3)]
    tree, A_list, b_list = multiconcat_rnn(rs, hid_bound=2.0)
    assert len(A_list) == 3
    assert len(b_list) == 3
    for ell in range(3):
        assert A_list[ell].shape[1] == tree.size.hidden_dim
        assert b_list[ell].shape[0] == rs[ell].size.output_dim


def test_zero_rnn_from_definition_outputs_zero():
    rnn0 = zero_rnn_from_definition(input_dim=3)
    y = np.squeeze(rnn0(SpreadInputOperator()(tensor([0.7, -1.2, 3.4]), T=8)), axis=-1)
    assert_allclose(y, np.zeros_like(y))


def test_identity_rnn_properties():
    rnn_id = identity_rnn()
    x_seq = SpreadInputOperator()(tensor([-0.75]), T=8)
    h = rnn_id.hidden_operator(x_seq)
    y = np.squeeze(rnn_id(x_seq), axis=-1)
    assert_allclose(y, np.full_like(y, fill_value=-0.75))
    assert_allclose(np.max(np.abs(h), axis=1), np.full((9,), 0.75))


def test_square_and_identity_rnn_sizes():
    rnn = square_and_identity_rnn(D=1.0)
    assert rnn.size.input_dim == 1
    assert rnn.size.output_dim == 2
    assert rnn.size.hidden_dim == 9


def test_square_and_identity_rnn_behavior():
    rnn = square_and_identity_rnn(D=1.0)
    x = tensor([0.6])
    y = rnn(SpreadInputOperator()(x, T=12))
    assert np.isclose(y[-1, 0], x[0] ** 2, atol=5e-3)
    assert_allclose(y[:, 1], np.full_like(y[:, 1], x.item()))


def test_poly_map_definition_level_1():
    x = tensor([-2.0, -0.5, 0.0, 1.5])
    assert_allclose(poly_map(1, x), np.stack([x**2, x], axis=-1))


def test_poly_map_definition_level_3_matches_formula():
    x = tensor([2.0, 3.0, 5.0])
    expected = tensor([x[1] * x[2], x[0] ** 2, x[0] * x[1], x[1] ** 2, x[2]])
    assert_allclose(poly_map(3, x), expected)


def test_poly_map_compose_matches_closed_form_from_lemma():
    x = np.linspace(-1.0, 1.0, 23)
    assert_allclose(poly_map_compose(4, x), poly_map_compose_closed_form(4, x), atol=1e-6)


def test_polymap_rnn_level_2_sizes_and_behavior():
    rnn = polymap_rnn(level=2, D=1.0)
    assert rnn.size.input_dim == 2
    assert rnn.size.output_dim == 3
    assert rnn.size.hidden_dim <= 40
    x = tensor([0.4, -0.7])
    target = poly_map(2, x)
    assert_allclose(rnn(SpreadInputOperator()(x, T=12))[-1], target, atol=6e-3)


def test_polymap_rnn_level_3_batch_accuracy():
    rnn = polymap_rnn(level=3, D=1.0)
    x_batch = tensor([[0.0, 0.0, 0.0], [0.1, -0.4, 0.8], [-0.9, 0.5, -0.2], [1.0, -1.0, 1.0]])
    target = poly_map(3, x_batch)
    y = rnn(SpreadInputOperator()(x_batch, T=12))
    assert_allclose(y[:, -1, :], target, atol=6e-3)


def test_powers_hidden_operator_matches_full_rnn_hidden_states():
    hid_op = powers_hidden_operator(L=3, D=1.0)
    rnn_full, _, A_list, b_list, _, _ = final_powers_rnn(L=3, D=1.0, return_aux=True)
    x_seq = SpreadInputOperator()(tensor([0.4]), T=8)
    assert_allclose(hid_op(x_seq), rnn_full.hidden_operator(x_seq))
    assert len(A_list) == 3
    assert len(b_list) == 3


def test_powers_output_mapping_shapes():
    A_o_pi, b_o_pi = powers_output_mapping(L=3, D=1.0)
    assert A_o_pi.shape[0] == 8
    assert b_o_pi.shape[0] == 8


def test_final_powers_rnn_basic_behavior():
    rnn = final_powers_rnn(L=3, D=1.0)
    x = tensor([0.4])
    y = rnn(SpreadInputOperator()(x, T=62))[-1]
    expected = tensor([x.item() ** p for p in range(1, 9)], dtype=y.dtype)
    assert_allclose(y, expected, atol=2e-2)


def test_clipping_rnn_increases_hidden_size_by_11():
    base = identity_rnn()
    smooth = clipping_rnn(base, B=1.0)
    assert smooth.size.hidden_dim == base.size.hidden_dim + 11
    assert smooth.size.output_dim == 1


def test_clipping_rnn_holds_dyadic_output_without_clipping():
    smooth = clipping_rnn(identity_rnn(), B=1.0)
    y = np.squeeze(smooth(SpreadInputOperator()(tensor([0.7]), T=14)), axis=-1)
    assert_allclose(y[3:15], np.full_like(y[3:15], 0.7), atol=1e-6)


def test_clipping_rnn_clips_to_interval():
    smooth = clipping_rnn(identity_rnn(), B=1.0)
    y = np.squeeze(smooth(SpreadInputOperator()(tensor([2.0]), T=14)), axis=-1)
    assert_allclose(y[3:15], np.full_like(y[3:15], 1.0), atol=1e-6)


def test_polynomial_rnn_linear_exact_case():
    rnn = polynomial_rnn([2.5, -1.2], D=1.0)
    y = np.squeeze(rnn(SpreadInputOperator()(tensor([0.4]), T=8)), axis=-1)
    assert_allclose(y, np.full_like(y, 2.5 - 1.2 * 0.4), atol=1e-6)


def test_polynomial_rnn_cubic_sanity():
    coeff = [1.0, 2.0, -3.0, 1.0]
    rnn = polynomial_rnn(coeff, D=1.0)
    y = np.squeeze(rnn(SpreadInputOperator()(tensor([0.6]), T=31)), axis=-1)
    target = coeff[0] + coeff[1] * 0.6 + coeff[2] * 0.6**2 + coeff[3] * 0.6**3
    assert np.isclose(y[-1], target, atol=8e-2)
