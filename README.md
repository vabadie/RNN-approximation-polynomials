# RNN-approximation-polynomials

You can open this project and run it in Binder (this might take a few minutes to load at first).

[![Open In Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/vabadie/RNN-approximation-polynomials/HEAD?labpath=notebooks%2Frnn_polynomial_approximation.ipynb)

Reference implementation for the RNN constructions used in the paper https://arxiv.org/abs/2511.15326

This folder contains three things:

1. A small NumPy package, `polyrnn_impl`, implementing the basic Elman-RNN
   model and the constructive lemmas/theorems from the paper.
2. A notebook, `notebooks/rnn_polynomial_approximation.ipynb`, for numerical
   experiments and visual checks.
3. A test suite, `tests/test_elman_rnn.py`, covering the core constructions.

The project is designed so that someone reading the paper can:

- instantiate the exact proof-defined RNNs,
- compose them using the lemmas from the paper,
- run numerical experiments in double precision or `mpmath`,
- and compare the numerical behavior against the theoretical bounds.

## Project layout

```text
RNN-approximation-polynomials/
├── polyrnn_impl/
│   ├── elman_rnn.py
│   ├── rnn_transformations.py
│   ├── rnn_constructions.py
│   ├── poly_maps.py
│   └── spread_input.py
├── notebooks/
│   ├── rnn_polynomial_approximation.ipynb
│   └── rnn_polynomial_approximation_support.py
├── tests/
│   └── test_elman_rnn.py
├── environment.yml
└── pyproject.toml
```

## Main concepts

The core model is the Elman RNN from the paper:

- hidden-state operator `H`
  - `h[-1] = 0`
  - `h[t] = ReLU(A_h h[t-1] + A_x x[t] + b_h)`
- output map `O`
  - `y[t] = A_o h[t] + b_o`
- recurrent network `R = O ∘ H`

The package then builds increasingly structured objects on top of this base:

- exact proof-defined constructions such as `squaring_rnn`,
  `multiplication_rnn_from_theorem`, and `polynomial_rnn`,
- structural transformations such as `parallel_rnn_from_list`,
  `concat_rnn`, `multiconcat_rnn`, and `clipping_rnn`,
- algebraic helpers such as `poly_map` and `poly_map_compose`,
- notebook helpers for plotting approximation trajectories and errors.

## Quick start

If you want to explore the notebook without installing anything locally, start with Binder.

### 0. Open online

You can launch the notebook directly in Binder, with no local installation:

- [Open `notebooks/rnn_polynomial_approximation.ipynb` in Binder](https://mybinder.org/v2/gh/vabadie/RNN-approximation-polynomials/HEAD?labpath=notebooks%2Frnn_polynomial_approximation.ipynb)

Binder will build the environment from [.binder/requirements.txt](/Users/valentinabadie/Desktop/Work/Research/Papers/Papers-in-process/RNNApproxPoly/RNN-2026/RNN-approximation-polynomials/.binder/requirements.txt) and open JupyterLab on the notebook. The first launch can take a few minutes while the environment is built, but this Binder-specific setup is intentionally slimmer than the full local development environment.

### 1. Bootstrap everything with one command

Recommended for most GitHub users:

```bash
./scripts/bootstrap_venv.sh
```

This creates a local virtual environment in `./.venv`, installs the project in
editable mode together with the notebook/test dependencies, and registers a
Jupyter kernel for the notebook.

If you prefer Conda or Mamba instead:

```bash
./scripts/bootstrap.sh
```

This creates a local Conda environment in `./.conda`, installs the project in
editable mode, and registers a Jupyter kernel for the notebook.

### 2. Activate the environment

```bash
source .venv/bin/activate
```

Or run tools without activation:

```bash
./.venv/bin/pytest
./.venv/bin/jupyter lab notebooks/rnn_polynomial_approximation.ipynb
```

### 3. Alternative manual setup

If you prefer not to use the bootstrap script, a plain `venv` + `pip` workflow
is enough:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[notebook,test]"
```

This installs the local package plus everything needed for the notebook and the
test suite.

If you also want to register a local Jupyter kernel manually, install the
Jupyter extra too:

```bash
pip install -e ".[notebook,jupyter,test]"
python -m ipykernel install --user --name polyrnn-approximation-polynomials
```

If you prefer Conda, the environment file is still self-contained:

```bash
conda env create -p ./.conda -f environment.yml
conda activate ./.conda
```

Because `environment.yml` includes `-e .`, the local package is installed
automatically during environment creation.

### 4. Run the tests

```bash
pytest
```

### 5. Open the notebook

```bash
jupyter lab notebooks/rnn_polynomial_approximation.ipynb
```

## Public API overview

The package exports the following groups of objects.

### Core model

- `ElmanRNN`
- `HiddenStateOperator`
- `OutputMapping`
- `RNNSize`
- `RNNWeights`
- `SpreadInputOperator`

### Structural transformations

- `parallel_rnn_from_list`
- `concat_rnn`
- `multiconcat_rnn`
- `multiconcat_rnn_tree`
- `linear_map_rnn_input`
- `linear_map_rnn_output`
- `clipping_rnn`

### Constructive RNNs from the paper

- `identity_rnn`
- `squaring_rnn`
- `squaring_rnn_weights`
- `square_rnn_from_theorem`
- `multiplication_rnn_from_theorem`
- `square_and_identity_rnn`
- `polymap_rnn`
- `powers_hidden_operator`
- `powers_output_mapping`
- `final_powers_rnn`
- `polynomial_rnn`
- `zero_rnn_from_definition`

### Polynomial map helpers

- `poly_map`
- `poly_map_compose`
- `poly_map_compose_closed_form`

## Notebook usage

The main notebook is:

- `notebooks/rnn_polynomial_approximation.ipynb`

Its helper module is:

- `notebooks/rnn_polynomial_approximation_support.py`

The notebook is set up for exploratory work:

- the first code cell bootstraps imports and precision defaults,
- the square experiment cell can run in standard NumPy mode or
  high-precision `mpmath` mode,
- the polynomial experiment cell can also run in both modes,
- experiment cells expose their intermediate variables back to the notebook
  namespace via `globals().update(...)` so you can inspect the outputs.

## Development notes

- The default numerical experiments use `numpy.float64`.
- The notebook helper can switch to `mpmath` when double precision becomes too
  coarse for long runs.
- The tests are intentionally concrete: many of them compare exact matrices and
  exact tensor shapes from the paper’s constructions.
