"""Public API for the RNNApproxPoly reference implementation.

The package is organized into four layers:

- core RNN primitives in `elman_rnn.py`,
- structural transformations in `rnn_transformations.py`,
- proof-driven constructions in `rnn_constructions.py`,
- polynomial-map helpers in `poly_maps.py`.

The goal of this module is to provide a single import surface for the objects
that are most useful in experiments, tests, and notebooks.
"""

from .elman_rnn import ElmanRNN, HiddenStateOperator, OutputMapping, RNNSize, RNNWeights
from .rnn_constructions import (
    final_powers_rnn,
    identity_rnn,
    multiplication_rnn_from_theorem,
    polynomial_rnn,
    powers_output_mapping,
    powers_hidden_operator,
    polymap_rnn,
    square_and_identity_rnn,
    square_rnn_from_theorem,
    squaring_rnn,
    squaring_rnn_weights,
    zero_rnn_from_definition,
)
from .poly_maps import poly_map, poly_map_compose, poly_map_compose_closed_form
from .rnn_transformations import (
    concat_rnn,
    clipping_rnn,
    linear_map_rnn_input,
    linear_map_rnn_output,
    multiconcat_rnn,
    multiconcat_rnn_tree,
    parallel_rnn_from_list,
)
from .spread_input import SpreadInputOperator

__all__ = [
    # Core RNN objects.
    "ElmanRNN",
    "HiddenStateOperator",
    "OutputMapping",
    "RNNSize",
    "RNNWeights",
    "SpreadInputOperator",

    # Structural transformations.
    "parallel_rnn_from_list",
    "concat_rnn",
    "multiconcat_rnn",
    "multiconcat_rnn_tree",
    "clipping_rnn",
    "linear_map_rnn_input",
    "linear_map_rnn_output",

    # Polynomial map utilities.
    "poly_map",
    "poly_map_compose",
    "poly_map_compose_closed_form",

    # Proof-driven constructions.
    "identity_rnn",
    "square_and_identity_rnn",
    "polymap_rnn",
    "powers_hidden_operator",
    "powers_output_mapping",
    "final_powers_rnn",
    "polynomial_rnn",
    "square_rnn_from_theorem",
    "squaring_rnn",
    "squaring_rnn_weights",
    "multiplication_rnn_from_theorem",
    "zero_rnn_from_definition",
]
