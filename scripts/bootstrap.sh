#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${1:-"$REPO_ROOT/.conda"}"
KERNEL_NAME="polyrnn-approximation-polynomials"
KERNEL_DISPLAY_NAME="Python (RNN Approximation Polynomials)"

if command -v mamba >/dev/null 2>&1; then
  CONDA_CMD=(mamba)
elif command -v conda >/dev/null 2>&1; then
  CONDA_CMD=(conda)
else
  echo "Error: neither 'conda' nor 'mamba' is available on PATH." >&2
  exit 1
fi

echo "Creating environment at: $ENV_PREFIX"
"${CONDA_CMD[@]}" env create --prefix "$ENV_PREFIX" --file "$REPO_ROOT/environment.yml"

echo "Registering a Jupyter kernel for the environment"
"${CONDA_CMD[@]}" run --prefix "$ENV_PREFIX" python -m ipykernel install --user --name "$KERNEL_NAME" --display-name "$KERNEL_DISPLAY_NAME"

cat <<EOF

Bootstrap complete.

Activate the environment with:
  conda activate "$ENV_PREFIX"

Or run commands without activation:
  ${CONDA_CMD[0]} run --prefix "$ENV_PREFIX" pytest
  ${CONDA_CMD[0]} run --prefix "$ENV_PREFIX" jupyter lab "$REPO_ROOT/notebooks/rnn_polynomial_approximation.ipynb"
EOF
