#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${1:-"$REPO_ROOT/.venv"}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
KERNEL_NAME="polyrnn-approximation-polynomials"
KERNEL_DISPLAY_NAME="Python (RNN Approximation Polynomials)"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Error: '$PYTHON_BIN' is not available on PATH." >&2
  exit 1
fi

echo "Creating virtual environment at: $ENV_DIR"
"$PYTHON_BIN" -m venv "$ENV_DIR"

echo "Upgrading pip in the virtual environment"
"$ENV_DIR/bin/python" -m pip install --upgrade pip

echo "Installing project, notebook, and Jupyter dependencies"
"$ENV_DIR/bin/pip" install -e "$REPO_ROOT[notebook,jupyter,test]"

echo "Registering a Jupyter kernel for the virtual environment"
"$ENV_DIR/bin/python" -m ipykernel install --user --name "$KERNEL_NAME" --display-name "$KERNEL_DISPLAY_NAME"

cat <<EOF

Bootstrap complete.

Activate the environment with:
  source "$ENV_DIR/bin/activate"

Or run commands without activation:
  "$ENV_DIR/bin/pytest"
  "$ENV_DIR/bin/jupyter" lab "$REPO_ROOT/notebooks/rnn_polynomial_approximation.ipynb"
EOF
