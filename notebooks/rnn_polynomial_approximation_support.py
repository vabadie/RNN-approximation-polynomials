"""Support utilities for the notebook `rnn_polynomial_approximation.ipynb`.

This module keeps the notebook cells short and readable by moving the heavier
setup and plotting logic into plain Python functions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import sys


def paper_plot_rcparams() -> dict[str, Any]:
    """Matplotlib style tuned to match `src/polyrnn/figs/fig_rnn_poly.pdf`."""
    return {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "legend.frameon": True,
    }


def _find_repo_root() -> Path:
    """Locate the `RNN-approximation-polynomials` directory from a notebook context."""
    here = Path(__file__).resolve()
    candidates = []
    for base in (Path.cwd().resolve(), here.parent, here.parent.parent):
        candidates.extend([base, *base.parents])

    seen = set()
    for root in candidates:
        if root in seen:
            continue
        seen.add(root)
        if (root / "polyrnn_impl").exists():
            return root
        nested = root / "RNN-approximation-polynomials" / "polyrnn_impl"
        if nested.exists():
            return nested.parent

    raise RuntimeError(
        "Could not find 'polyrnn_impl'. Launch Jupyter from RNN-2026 or RNN-approximation-polynomials."
    )


def _ensure_repo_on_syspath() -> Path:
    """Make the repository importable even if notebook cells are run out of order."""
    repo_root = _find_repo_root()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


def bootstrap_rnn_polynomial_approximation() -> dict[str, Any]:
    """Return the objects that the notebook expects in its global namespace."""
    import mpmath as mp
    import numpy as np

    repo_root = _ensure_repo_on_syspath()
    np.set_printoptions(precision=4, suppress=True)

    print("Python:", sys.executable)
    print("Repo root for imports:", repo_root)

    from polyrnn_impl import (
        ElmanRNN,
        HiddenStateOperator,
        OutputMapping,
        SpreadInputOperator,
        final_powers_rnn,
        multiplication_rnn_from_theorem,
        poly_map,
        poly_map_compose,
        poly_map_compose_closed_form,
        polymap_rnn,
        polynomial_rnn,
        square_and_identity_rnn,
        square_rnn_from_theorem,
    )

    return {
        "Path": Path,
        "sys": sys,
        "mp": mp,
        "np": np,
        "repo_root": repo_root,
        "paper_plot_rcparams": paper_plot_rcparams,
        "backend": "numpy",
        "HiddenStateOperator": HiddenStateOperator,
        "OutputMapping": OutputMapping,
        "ElmanRNN": ElmanRNN,
        "SpreadInputOperator": SpreadInputOperator,
        "square_rnn_from_theorem": square_rnn_from_theorem,
        "multiplication_rnn_from_theorem": multiplication_rnn_from_theorem,
        "square_and_identity_rnn": square_and_identity_rnn,
        "polymap_rnn": polymap_rnn,
        "poly_map": poly_map,
        "poly_map_compose": poly_map_compose,
        "poly_map_compose_closed_form": poly_map_compose_closed_form,
        "final_powers_rnn": final_powers_rnn,
        "polynomial_rnn": polynomial_rnn,
    }


def run_square_approximation_experiment(
    *,
    full_precision: bool = False,
    mp_dps: int | None = None,
    n_points: int = 100,
    x_min: float = -1.0,
    x_max: float = 1.0,
    T: int = 10,
    t_values: list[int] | tuple[int, ...] = (1, 2, 5, 10),
    D_square: float = 1.0,
    plot_target_curve: bool = True,
    plot_error_across_iterations: bool = True,
    eps_plot: float | None = None,
) -> dict[str, Any]:
    """Run and plot the square-approximation experiment used in the notebook."""
    import matplotlib.pyplot as plt
    import mpmath as mp
    import numpy as np

    _ensure_repo_on_syspath()
    from polyrnn_impl import SpreadInputOperator, square_rnn_from_theorem

    mode = "mpmath" if full_precision else "numpy"
    t_values = [int(t) for t in t_values]
    if not t_values:
        raise ValueError("t_values must contain at least one index.")
    if any((t < 0 or t > T) for t in t_values):
        raise ValueError(f"All t in t_values must be in [0, {T}], got {t_values}.")
    if eps_plot is None:
        eps_plot = 1e-60 if mode == "mpmath" else 1e-30

    square_rnn = square_rnn_from_theorem(dtype=np.float64, D=D_square)
    spread = None

    if mode == "numpy":
        spread = SpreadInputOperator()
        x_batch = np.linspace(x_min, x_max, n_points, dtype=np.float64)[:, None]
        x_batch_seq = spread(x_batch, T=T)
        y_seq = square_rnn(x_batch_seq)
        x_vals = x_batch[:, 0]
        target_vals = x_vals**2
        y_plot_t = {int(t): y_seq[:, t, 0] for t in t_values}
        max_abs_error_by_t = np.max(np.abs(y_seq[:, :, 0] - x_batch**2), axis=0)
    else:
        if mp_dps is None:
            mp_dps = max(50, int(np.ceil(0.8 * T + 25)))
        mp.mp.dps = mp_dps
        print(mp.mp.dps, "decimal digits of precision for mpmath calculations")

        A_h_np = square_rnn.weights.A_h
        A_x_np = square_rnn.weights.A_x
        b_h_np = square_rnn.weights.b_h
        A_o_np = square_rnn.weights.A_o
        b_o_np = square_rnn.weights.b_o

        hidden_dim = A_h_np.shape[0]
        tol_nz = 1e-15
        Ah_rows = []
        for i in range(hidden_dim):
            row = []
            for j, v in enumerate(A_h_np[i]):
                if abs(float(v)) > tol_nz:
                    row.append((j, mp.mpf(str(float(v)))))
            Ah_rows.append(row)

        Ax_col = [mp.mpf(str(float(v))) for v in A_x_np[:, 0]]
        bh = [mp.mpf(str(float(v))) for v in b_h_np]
        Ao = [mp.mpf(str(float(v))) for v in A_o_np[0]]
        bo = mp.mpf(str(float(b_o_np[0])))

        def relu_mp(z: Any) -> Any:
            return z if z > 0 else mp.mpf("0")

        def rnn_rollout_mp(x_mp: Any, n_steps: int) -> list[Any]:
            h = [mp.mpf("0") for _ in range(hidden_dim)]
            ys = []
            for t in range(n_steps + 1):
                x_t = x_mp if t == 0 else mp.mpf("0")
                h_new = [mp.mpf("0") for _ in range(hidden_dim)]
                for i in range(hidden_dim):
                    s = bh[i] + Ax_col[i] * x_t
                    for j, v in Ah_rows[i]:
                        s += v * h[j]
                    h_new[i] = relu_mp(s)
                h = h_new
                y = bo
                for j in range(hidden_dim):
                    y += Ao[j] * h[j]
                ys.append(y)
            return ys

        x_vals = np.linspace(x_min, x_max, n_points)
        target_vals = x_vals**2
        y_plot_t = {int(t): [] for t in t_values}
        max_abs_error_by_t_mp = [mp.mpf("0") for _ in range(T + 1)]

        for xv in x_vals:
            x_mp = mp.mpf(str(float(xv)))
            ys = rnn_rollout_mp(x_mp, T)
            target = x_mp * x_mp
            for t in t_values:
                y_plot_t[int(t)].append(float(ys[int(t)]))
            for t in range(T + 1):
                err = abs(ys[t] - target)
                if err > max_abs_error_by_t_mp[t]:
                    max_abs_error_by_t_mp[t] = err

        max_abs_error_by_t = np.array([float(e) for e in max_abs_error_by_t_mp], dtype=np.float64)
        x_batch = None
        x_batch_seq = None
        y_seq = None

    theoretical_error_by_t = 4.0 ** (-1 - np.arange(T + 1, dtype=np.float64))

    with plt.rc_context(paper_plot_rcparams()):
        n_cols = 2 if plot_error_across_iterations else 1
        fig, axes = plt.subplots(1, n_cols, figsize=(13.0 if plot_error_across_iterations else 7.5, 4.8))
        axes = [axes] if n_cols == 1 else list(axes)
        ax_main = axes[0]

        for t in t_values:
            ax_main.plot(x_vals, y_plot_t[int(t)], linewidth=2, label=f"$t={t}$")
        if plot_target_curve:
            ax_main.plot(x_vals, target_vals, "k--", linewidth=2, label="Target $x^2$")
        ax_main.set_xlabel("x")
        ax_main.set_ylabel("Output")
        ax_main.set_title(f"(a) RNN approximation of $x^2$")
        ax_main.legend()

        if plot_error_across_iterations:
            ax_error = axes[1]
            iterations = list(range(T + 1))
            safe_error = max_abs_error_by_t.copy()
            safe_error[safe_error <= 0] = eps_plot
            ax_error.plot(iterations, safe_error, marker="o", linewidth=2, label="Numerical error")
            ax_error.plot(
                iterations[1:],
                theoretical_error_by_t[1:],
                "--",
                linewidth=2,
                label=r"Theorem 8 bound $(D^2/4)4^{-t}, t \geq 1$",
            )
            ax_error.set_yscale("log", base=2)
            ax_error.set_xlabel("Time step $t$")
            ax_error.set_ylabel("Sup Error")
            ax_error.set_title("(b) Error decay")
            ax_error.legend()

        fig.tight_layout()
        plt.show()

    return {
        "plt": plt,
        "mp": mp,
        "np": np,
        "mode": mode,
        "full_precision": full_precision,
        "mp_dps": mp_dps,
        "n_points": n_points,
        "x_min": x_min,
        "x_max": x_max,
        "T": T,
        "t_values": t_values,
        "D_square": D_square,
        "plot_target_curve": plot_target_curve,
        "plot_error_across_iterations": plot_error_across_iterations,
        "eps_plot": eps_plot,
        "square_rnn": square_rnn,
        "spread": spread,
        "x_batch": x_batch,
        "x_batch_seq": x_batch_seq,
        "y_seq": y_seq,
        "y_plot_t": y_plot_t,
        "x_vals": x_vals,
        "target_vals": target_vals,
        "max_abs_error_by_t": max_abs_error_by_t,
        "theoretical_error_by_t": theoretical_error_by_t,
        "fig": fig,
        "axes": axes,
    }


def _start_upper_bound(degree_poly: int) -> int:
    import numpy as np

    return int(np.ceil(16 * np.log2(degree_poly)))


def run_polynomial_approximation_experiment(
    *,
    full_precision: bool = True,
    coeff_poly: Any = None,
    mp_dps: int | None = None,
    D_poly: float = 1.0,
    n_points_poly: int = 35,
    n_x_err: int = 35,
    x_min_poly: float = -1.0,
    x_max_poly: float = 1.0,
    T_poly: int = 100,
    t_values_poly: Any = None,
    show_main_theorem_bound: bool = True,
    eps_plot: float | None = None,
) -> dict[str, Any]:
    """Run and plot the polynomial-approximation experiment used in the notebook."""
    import matplotlib.pyplot as plt
    import mpmath as mp
    import numpy as np

    _ensure_repo_on_syspath()
    from polyrnn_impl import SpreadInputOperator, polynomial_rnn

    mode = "mpmath" if full_precision else "numpy"

    if coeff_poly is None:
        coeff_poly = np.random.uniform(-1.0, 1.0, size=3)
    coeff_poly = np.asarray(coeff_poly, dtype=np.float64)
    N_poly = len(coeff_poly) - 1

    if t_values_poly is None:
        t_values_poly = np.arange(0, T_poly + 1, max(1, T_poly // 5))
    t_values_poly = np.asarray(t_values_poly, dtype=int)

    if eps_plot is None:
        eps_plot = 1e-60 if mode == "mpmath" else 1e-30

    if mode not in {"numpy", "mpmath"}:
        raise ValueError(f"mode must be numpy or mpmath, got {mode}.")
    if N_poly < 0:
        raise ValueError("coeff_poly must contain at least one coefficient a0.")
    if len(t_values_poly) == 0:
        raise ValueError("t_values_poly must contain at least one index.")
    if any((t < 0 or t > T_poly) for t in t_values_poly):
        raise ValueError(f"All t must be in [0, {T_poly}], got {t_values_poly}.")

    x_plot = np.linspace(x_min_poly, x_max_poly, n_points_poly)
    x_err = np.linspace(x_min_poly, x_max_poly, n_x_err)
    start_target_plot = _start_upper_bound(N_poly) if N_poly >= 2 else 0

    if mode == "numpy":
        rnn_poly = polynomial_rnn(coeff_poly, D=D_poly, dtype=np.float64)
        spread = SpreadInputOperator()

        x_plot_t = x_plot[:, None]
        y_plot_seq = np.squeeze(rnn_poly(spread(x_plot_t, T=T_poly)), axis=-1)
        x_err_t = x_err[:, None]
        y_err_seq = np.squeeze(rnn_poly(spread(x_err_t, T=T_poly)), axis=-1)

        target_plot = np.zeros_like(x_plot)
        for i, a_i in enumerate(coeff_poly):
            target_plot = target_plot + float(a_i) * (x_plot**i)

        target_err_t = np.zeros((len(x_err),), dtype=np.float64)
        x_err_flat = x_err_t[:, 0]
        for i, a_i in enumerate(coeff_poly):
            target_err_t = target_err_t + float(a_i) * (x_err_flat**i)

        y_plot_t = {int(t): y_plot_seq[:, int(t)] for t in t_values_poly}
        max_err = [float(np.max(np.abs(y_err_seq[:, t] - target_err_t))) for t in range(T_poly + 1)]
    else:
        c1 = float(np.sum(np.abs(coeff_poly))) * 16.0 * N_poly * (D_poly ** (2 * N_poly))
        c2 = 1.0 / (4.0 * np.ceil(np.log2(N_poly)))
        mp_dps = int(np.ceil(2 * (2 * c2 * T_poly - np.log2(c1))))
        mp.mp.dps = mp_dps
        print(mp.mp.dps, "decimal digits of precision for mpmath calculations")
        coeff_mp = [mp.mpf(str(float(a))) for a in coeff_poly]

        rnn_poly = polynomial_rnn(coeff_poly, D=D_poly, dtype=np.float64)
        spread = None
        A_h_np = rnn_poly.weights.A_h
        A_x_np = rnn_poly.weights.A_x
        b_h_np = rnn_poly.weights.b_h
        A_o_np = rnn_poly.weights.A_o
        b_o_np = rnn_poly.weights.b_o

        m = A_h_np.shape[0]
        tol_nz = 1e-15
        Ah_rows = []
        for i in range(m):
            row = []
            for j, v in enumerate(A_h_np[i]):
                if abs(float(v)) > tol_nz:
                    row.append((j, mp.mpf(str(float(v)))))
            Ah_rows.append(row)

        Ax_col = [mp.mpf(str(float(v))) for v in A_x_np[:, 0]]
        bh = [mp.mpf(str(float(v))) for v in b_h_np]
        Ao = [mp.mpf(str(float(v))) for v in A_o_np[0]]
        bo = mp.mpf(str(float(b_o_np[0])))

        def relu_mp(z: Any) -> Any:
            return z if z > 0 else mp.mpf("0")

        def poly_mp(x_mp: Any, coeff: list[Any]) -> Any:
            s = mp.mpf("0")
            pwr = mp.mpf("1")
            for a in coeff:
                s += a * pwr
                pwr *= x_mp
            return s

        def rnn_rollout_mp(x_mp: Any, horizon: int) -> list[Any]:
            h = [mp.mpf("0") for _ in range(m)]
            ys = []
            for t in range(horizon + 1):
                x_t = x_mp if t == 0 else mp.mpf("0")
                h_new = [mp.mpf("0") for _ in range(m)]
                for i in range(m):
                    s = bh[i] + Ax_col[i] * x_t
                    for j, v in Ah_rows[i]:
                        s += v * h[j]
                    h_new[i] = relu_mp(s)
                h = h_new
                y = bo
                for j in range(m):
                    y += Ao[j] * h[j]
                ys.append(y)
            return ys

        y_plot_t = {int(t): [] for t in t_values_poly}
        target_plot = []
        for xv in x_plot:
            x_mp = mp.mpf(str(float(xv)))
            ys = rnn_rollout_mp(x_mp, T_poly)
            for t in t_values_poly:
                y_plot_t[int(t)].append(float(ys[int(t)]))
            target_plot.append(float(poly_mp(x_mp, coeff_mp)))

        max_err = [mp.mpf("0") for _ in range(T_poly + 1)]
        for xv in x_err:
            x_mp = mp.mpf(str(float(xv)))
            ys = rnn_rollout_mp(x_mp, T_poly)
            p_ref = poly_mp(x_mp, coeff_mp)
            for t in range(T_poly + 1):
                e = abs(ys[t] - p_ref)
                if e > max_err[t]:
                    max_err[t] = e
        max_err = [float(e) for e in max_err]

    t_axis = np.arange(T_poly + 1)
    err_np = np.array(max_err, dtype=np.float64)
    y_label = "Sup error"
    y_err = err_np

    bound = None
    bound_plot = None
    c1 = None
    c2 = None
    with plt.rc_context(paper_plot_rcparams()):
        fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8))
        ax_values, ax_error = axes

        for t in t_values_poly:
            ax_values.plot(x_plot, y_plot_t[int(t)], linewidth=2, label=f"$t={int(t)}$")
        ax_values.plot(x_plot, target_plot, "k--", linewidth=2, label="Target $f(x)$")
        ax_values.set_xlabel("x")
        ax_values.set_ylabel("Output")
        ax_values.set_title("(a) RNN approximation of random polynomial $f(x)$")
        ax_values.legend()

        ax_error.plot(t_axis, y_err, "o-", linewidth=2, label=f"Numerical error")
        if show_main_theorem_bound and N_poly >= 2:
            c1 = float(np.sum(np.abs(coeff_poly))) * 16.0 * N_poly * (D_poly ** (2 * N_poly))
            c2 = 1.0 / (4.0 * np.ceil(np.log2(N_poly)))
            bound_plot = c1 * (4.0 ** (-c2 * t_axis))
            ax_error.plot(
                t_axis[start_target_plot:],
                bound_plot[start_target_plot:],
                "--",
                linewidth=2,
                label="Theorem 3 bound: $\|a\|_1 C_1 4^{-C_2t}, t \geq 16 \log (N)$",
            )
        ax_error.set_yscale("log", base=2)
        ax_error.set_xlabel("Time step $t$")
        ax_error.set_ylabel(y_label)
        ax_error.set_title("(b) Error decay")
        ax_error.legend()

        fig.tight_layout()
        plt.show()

    fig_values = fig
    fig_error = fig

    return {
        "plt": plt,
        "mp": mp,
        "np": np,
        "mode": mode,
        "full_precision": full_precision,
        "coeff_poly": coeff_poly,
        "N_poly": N_poly,
        "mp_dps": mp_dps,
        "D_poly": D_poly,
        "n_points_poly": n_points_poly,
        "n_x_err": n_x_err,
        "x_min_poly": x_min_poly,
        "x_max_poly": x_max_poly,
        "T_poly": T_poly,
        "t_values_poly": t_values_poly,
        "show_main_theorem_bound": show_main_theorem_bound,
        "eps_plot": eps_plot,
        "start_target_plot": start_target_plot,
        "x_plot": x_plot,
        "x_err": x_err,
        "rnn_poly": rnn_poly,
        "spread": spread,
        "y_plot_t": y_plot_t,
        "target_plot": target_plot,
        "max_err": max_err,
        "t_axis": t_axis,
        "err_np": err_np,
        "y_err": y_err,
        "y_label": y_label,
        "bound": bound,
        "bound_plot": bound_plot,
        "c1": c1,
        "c2": c2,
        "fig": fig,
        "axes": axes,
        "fig_values": fig_values,
        "fig_error": fig_error,
    }
