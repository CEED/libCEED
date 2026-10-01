#!/usr/bin/env python3

# Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and other CEED contributors.
# All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
#
# SPDX-License-Identifier: BSD-2-Clause
#
# This file is part of CEED:  http://github.com/ceed

"""
Plot CEED benchmark performance scaling curves ([DOFs x CG iter] / [nodes x sec])
across polynomial degrees P and quadrature points Q.

Default axis ranges are automatically computed from the global minimum and maximum
across all filtered benchmark results, ensuring consistent, directly comparable
scales across all output plots.

Example Usage:
    # 1. Batch generate PDF plots with auto-computed global scales across all runs:
    python postprocess_plot.py petsc-bpsraw-*-output.txt

    # 2. Filter for a specific test problem, vector case, and code:
    python postprocess_plot.py petsc-bpsraw-*-output.txt \
        --test "CEED Benchmark Problem 1" \
        --case vector \
        --code libCEED

    # 3. Use logarithmic y-axis and reference iter/s slope lines (scales auto-adjust):
    python postprocess_plot.py *.log --log-y --draw-iter-lines

    # 4. Use exact min/max bounds without boundary margin padding:
    python postprocess_plot.py *.log --padding 0.0

    # 5. Manually override with explicit axis ranges:
    python postprocess_plot.py *.log \
        --x-range 1e2 1e7 \
        --y-range 1e6 5e9 \
        --output-dir ./figures \
        --format png

    # 6. Stream log files over stdin:
    cat petsc-bpsraw-bp1-*-output.txt | python postprocess_plot.py --log-y
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from postprocess_base import read_logs

# Default palette
COLOR_PALETTE: List[str] = [
    "dimgrey", "black", "saddlebrown", "firebrick", "red", "orange",
    "gold", "lightgreen", "green", "cyan", "teal", "blue", "navy",
    "purple", "magenta", "pink"
]


def parse_args(args: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for benchmark plotting."""
    parser = argparse.ArgumentParser(
        description="Generate performance scaling plots from CEED benchmark logs."
    )
    parser.add_argument(
        "files",
        nargs="*",
        type=Path,
        help="Input log files to read (reads stdin if omitted).",
    )
    parser.add_argument(
        "--test",
        type=str,
        default=None,
        help="Filter by specific test name (default: auto-detect first test).",
    )
    parser.add_argument(
        "--case",
        type=str,
        choices=["scalar", "vector"],
        default=None,
        help="Filter by problem case: 'scalar' or 'vector' (default: auto-detect).",
    )
    parser.add_argument(
        "--code",
        type=str,
        default=None,
        help="Filter by framework/code name (default: auto-detect first code).",
    )
    parser.add_argument(
        "--log-y",
        action="store_true",
        help="Use logarithmic scaling on the y-axis.",
    )
    parser.add_argument(
        "--x-range",
        type=float,
        nargs=2,
        default=None,
        metavar=("XMIN", "XMAX"),
        help="Explicit domain limits for x-axis (default: auto-computed from filtered data).",
    )
    parser.add_argument(
        "--y-range",
        type=float,
        nargs=2,
        default=None,
        metavar=("YMIN", "YMAX"),
        help="Explicit range limits for y-axis (default: auto-computed from filtered data).",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.05,
        help="Fractional margin padding added to auto-computed min/max bounds (default: 0.05; 0 for exact).",
    )
    parser.add_argument(
        "--draw-iter-lines",
        action="store_true",
        help="Overlay reference iter/s slope guidelines.",
    )
    parser.add_argument(
        "--ymin-iter-lines",
        type=float,
        default=3e5,
        help="Minimum y-value for iter/s lines (default: 3e5).",
    )
    parser.add_argument(
        "--ymax-iter-lines",
        type=float,
        default=8e8,
        help="Maximum y-value for iter/s lines (default: 8e8).",
    )
    parser.add_argument(
        "--legend-ncol",
        type=int,
        default=None,
        help="Number of columns in the plot legend.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory to save output figures (default: current directory).",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="pdf",
        choices=["pdf", "png", "svg"],
        help="Figure output format (default: pdf).",
    )
    parser.add_argument(
        "--save",
        dest="save_figures",
        action="store_true",
        default=True,
        help="Save generated figures to disk (default: enabled).",
    )
    parser.add_argument(
        "--no-save",
        dest="save_figures",
        action="store_false",
        help="Do not save generated figures to disk.",
    )
    parser.add_argument(
        "--show",
        dest="show_figures",
        action="store_true",
        default=False,
        help="Display figures interactively on screen.",
    )
    return parser.parse_args(args)


def configure_matplotlib(headless: bool = True) -> None:
    """Configure matplotlib backend and general typography."""
    if headless:
        mpl.use("Agg")
    mpl.rcParams["font.sans-serif"] = ["Noto Sans", "Open Sans", "DejaVu Sans", "sans-serif"]
    mpl.rcParams["figure.figsize"] = [10.0, 8.0]


def compute_axis_limits(
    df: pd.DataFrame,
    vdim: int,
    log_y: bool,
    padding: float = 0.05,
) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    """
    Calculate global axis boundaries across all filtered benchmark results.

    Returns:
        (x_limits, y_limits, raw_x_bounds, raw_y_bounds)
    """
    # Safe computation of compute nodes
    num_procs_node = df["num_procs_node"].replace(0, np.nan)
    num_nodes = (df["num_procs"] / num_procs_node).fillna(1.0).to_numpy(dtype=float)

    # Compute DOFs per compute node (x) and DPS per compute node (y)
    x_vals = (df["num_unknowns"].to_numpy(dtype=float) / (num_nodes * vdim))
    y_vals = (df["cg_iteration_dps"].to_numpy(dtype=float) / num_nodes)

    mask = np.isfinite(x_vals) & np.isfinite(y_vals) & (x_vals > 0)
    if log_y:
        mask &= (y_vals > 0)
    else:
        mask &= (y_vals >= 0)

    if not np.any(mask):
        default_x = (1e1, 4e6)
        default_y = (1e5, 1e9) if log_y else (0.0, 2e9)
        return default_x, default_y, default_x, default_y

    raw_x_min = float(np.min(x_vals[mask]))
    raw_x_max = float(np.max(x_vals[mask]))
    raw_y_min = float(np.min(y_vals[mask]))
    raw_y_max = float(np.max(y_vals[mask]))

    # Compute X bounds (log scale)
    if padding > 0.0 and raw_x_max > raw_x_min:
        log_xmin = np.log10(raw_x_min)
        log_xmax = np.log10(raw_x_max)
        span_log_x = log_xmax - log_xmin
        x_min_lim = float(10.0 ** (log_xmin - padding * span_log_x))
        x_max_lim = float(10.0 ** (log_xmax + padding * span_log_x))
    else:
        x_min_lim, x_max_lim = raw_x_min, raw_x_max

    # Compute Y bounds (log or linear scale)
    if log_y:
        if padding > 0.0 and raw_y_max > raw_y_min:
            log_ymin = np.log10(raw_y_min)
            log_ymax = np.log10(raw_y_max)
            span_log_y = log_ymax - log_ymin
            y_min_lim = float(10.0 ** (log_ymin - padding * span_log_y))
            y_max_lim = float(10.0 ** (log_ymax + padding * span_log_y))
        else:
            y_min_lim, y_max_lim = raw_y_min, raw_y_max
    else:
        if padding > 0.0 and raw_y_max > raw_y_min:
            span_y = raw_y_max - raw_y_min
            y_min_lim = max(0.0, float(raw_y_min - padding * span_y))
            y_max_lim = float(raw_y_max + padding * span_y)
        else:
            y_min_lim = 0.0
            y_max_lim = raw_y_max if raw_y_max > 0 else 1.0

    return (
        (x_min_lim, x_max_lim),
        (y_min_lim, y_max_lim),
        (raw_x_min, raw_x_max),
        (raw_y_min, raw_y_max),
    )


def extract_series(
    df: pd.DataFrame,
    degree: int,
    quad_pts: int,
    num_nodes: float,
    vdim: int,
) -> np.ndarray:
    """Extract sorted (dofs_per_node, min_rate, max_rate) series for a given P and Q."""
    subset = df[(df["degree"] == degree) & (df["quadrature_pts"] == quad_pts)]
    if subset.empty:
        return np.empty((0, 3), dtype=float)

    x = subset["num_unknowns"].to_numpy(dtype=float)
    y = (subset["cg_iteration_dps"] / num_nodes).to_numpy(dtype=float)

    records: List[List[float]] = []
    for unique_x in np.unique(x):
        mask = x == unique_x
        records.append([unique_x, float(np.min(y[mask])), float(np.max(y[mask]))])

    records.sort(key=lambda r: r[0])
    return np.asarray(records, dtype=float)


def plot_single_configuration(
    pl_runs: pd.DataFrame,
    backend: str,
    backend_memtype: str,
    num_procs: float,
    num_procs_node: float,
    test_short: str,
    vdim: int,
    code: str,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    args: argparse.Namespace,
    console: Console,
) -> Optional[Path]:
    """Render and optionally export a single benchmark figure."""
    num_nodes = num_procs / num_procs_node if num_procs_node > 0 else 1.0

    fig, ax = plt.subplots()

    sol_p_set = sorted(pl_runs["degree"].drop_duplicates().tolist())
    executable = sorted(pl_runs["executable"].drop_duplicates().tolist())[0]
    cm_size = len(COLOR_PALETTE)

    color_idx = 0
    for sol_p in sol_p_set:
        qpts = (
            pl_runs.loc[pl_runs["degree"] == sol_p, "quadrature_pts"]
            .drop_duplicates()
            .sort_values(ascending=False)
            .reset_index(drop=True)
        )

        if qpts.empty:
            continue

        color = COLOR_PALETTE[color_idx % cm_size]

        # Primary quadrature point set
        d0 = extract_series(pl_runs, sol_p, int(qpts[0]), num_nodes, vdim)
        if len(d0) > 0:
            ax.plot(d0[:, 0], d0[:, 2], "o-", color=color, label=f"p={sol_p}")
            if not np.allclose(d0[:, 1], d0[:, 2]):
                ax.plot(d0[:, 0], d0[:, 1], "o-", color=color)
                ax.fill_between(d0[:, 0], d0[:, 1], d0[:, 2], facecolor=color, alpha=0.2)

        # Secondary quadrature point set (collocated/overintegrated)
        if len(qpts) > 1:
            d1 = extract_series(pl_runs, sol_p, int(qpts[1]), num_nodes, vdim)
            if len(d1) > 0:
                ax.plot(d1[:, 0], d1[:, 2], "s--", color=color, label=f"p={sol_p} (q={qpts[1]})")
                if not np.allclose(d1[:, 1], d1[:, 2]):
                    ax.plot(d1[:, 0], d1[:, 1], "s--", color=color)

        color_idx += 1

    # Optional reference iteration lines
    if args.draw_iter_lines:
        y0, y1 = args.ymin_iter_lines, args.ymax_iter_lines
        y_pts = np.asarray([y0, y1]) if args.log_y else np.exp(np.linspace(np.log(y0), np.log(y1), 100))
        slope1, slope2 = 600.0, 6000.0
        ax.plot(y_pts / slope1, y_pts, "k--", label=f"{slope1 / vdim:g} iter/s")
        ax.plot(y_pts / slope2, y_pts, "k-", label=f"{slope2 / vdim:g} iter/s")

    # Titles and formatting
    node_str = "node" if int(num_nodes) == 1 else "nodes"
    title_text = (
        f"{int(num_nodes)} {node_str} \u00d7 {int(num_procs_node)} ranks, "
        f"{backend}, {backend_memtype}, {test_short}"
    )
    ax.set_title(title_text, fontsize=16)
    ax.set_xscale("log")
    if args.log_y:
        ax.set_yscale("log")

    # Apply globally synchronized scales
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)

    ax.grid(True, which="major", color="gray", linestyle="dotted")
    ax.grid(True, which="minor", color="gray", linestyle="dotted")
    ax.tick_params(labelsize=14)
    ax.yaxis.get_offset_text().set_size(14)
    ax.set_axisbelow(True)
    ax.set_xlabel("DOFs per compute node", fontsize=14)
    ax.set_ylabel("[DOFs \u00d7 CG iterations] / [compute nodes \u00d7 seconds]", fontsize=14)

    legend_ncol = args.legend_ncol if args.legend_ncol is not None else (2 if args.log_y else 1)
    ax.legend(ncol=legend_ncol, loc="best", fontsize=13)

    saved_path: Optional[Path] = None
    if args.save_figures:
        short_backend = backend.replace("/", "")
        test_short_save = test_short.replace(" ", "")
        filename = (
            f"plot_{executable}_{code}_{test_short_save}_{short_backend}_{backend_memtype}_"
            f"N{int(num_nodes):03d}_pn{int(num_procs_node)}.{args.format}"
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        saved_path = args.output_dir / filename
        fig.savefig(saved_path, format=args.format, bbox_inches="tight")
        console.print(f"[bold green]✔[/bold green] Saved plot: [cyan]{saved_path}[/cyan]")

    if not args.show_figures:
        plt.close(fig)

    return saved_path


def main(args: Optional[Sequence[str]] = None) -> int:
    parsed_args = parse_args(args)
    console = Console()

    configure_matplotlib(headless=not parsed_args.show_figures)

    file_inputs = [str(p) for p in parsed_args.files] if parsed_args.files else None
    with console.status("[bold blue]Reading benchmark logs...", spinner="dots"):
        runs = read_logs(file_inputs)

    if runs.empty:
        console.print("[bold red]Error:[/bold red] No runs loaded from input.")
        return 1

    # Filter by test
    tests: List[str] = runs["test"].dropna().unique().tolist()
    if not tests:
        console.print("[bold red]Error:[/bold red] No valid tests found in logs.")
        return 1

    test_to_use = parsed_args.test if parsed_args.test else tests[0]
    sel_runs = runs[runs["test"] == test_to_use].copy()
    if sel_runs.empty:
        console.print(f"[bold red]Error:[/bold red] Test '{test_to_use}' not present in dataset.")
        return 1

    if "CEED Benchmark Problem Points" in test_to_use:
        test_short = test_to_use.strip().split()[0] + " BP" + test_to_use.strip().split()[-1] + "P"
    elif "CEED Benchmark Problem" in test_to_use:
        test_short = test_to_use.strip().split()[0] + " BP" + test_to_use.strip().split()[-1]
    else:
        test_short = test_to_use.strip()

    # Filter by case (scalar vs vector)
    cases: List[str] = sel_runs["case"].dropna().unique().tolist()
    case_to_use = parsed_args.case if parsed_args.case else (cases[0] if cases else "scalar")
    vdim = 1 if case_to_use == "scalar" else 3
    sel_runs = sel_runs[sel_runs["case"] == case_to_use]

    # Filter by code
    codes: List[str] = sel_runs["code"].dropna().unique().tolist()
    code_to_use = parsed_args.code if parsed_args.code else (codes[0] if codes else "libCEED")
    sel_runs = sel_runs[sel_runs["code"] == code_to_use]

    # Compute overall min/max bounds across all matching filtered records
    auto_x_lims, auto_y_lims, raw_x_bounds, raw_y_bounds = compute_axis_limits(
        df=sel_runs,
        vdim=vdim,
        log_y=parsed_args.log_y,
        padding=parsed_args.padding,
    )

    effective_x_range = tuple(parsed_args.x_range) if parsed_args.x_range is not None else auto_x_lims
    effective_y_range = tuple(parsed_args.y_range) if parsed_args.y_range is not None else auto_y_lims

    # Display configuration and computed bounds summary
    summary_table = Table(title="Plotting Target Selection & Dynamic Scaling", header_style="bold cyan")
    summary_table.add_column("Parameter")
    summary_table.add_column("Value / Boundaries", style="green")

    summary_table.add_row("Test", test_to_use)
    summary_table.add_row("Case", f"{case_to_use} (vdim={vdim})")
    summary_table.add_row("Code", code_to_use)
    summary_table.add_row("Total Matched Runs", str(len(sel_runs)))
    summary_table.add_row("Raw Data X-extent", f"[{raw_x_bounds[0]:.2e}, {raw_x_bounds[1]:.2e}] pts/node")
    summary_table.add_row("Raw Data Y-extent", f"[{raw_y_bounds[0]:.2e}, {raw_y_bounds[1]:.2e}] DOFs/s/node")
    summary_table.add_row(
        "Plot X-limits",
        f"[{effective_x_range[0]:.2e}, {effective_x_range[1]:.2e}]"
        + (" (user override)" if parsed_args.x_range is not None else f" (auto, pad={parsed_args.padding})")
    )
    summary_table.add_row(
        "Plot Y-limits",
        f"[{effective_y_range[0]:.2e}, {effective_y_range[1]:.2e}]"
        + (" (user override)" if parsed_args.y_range is not None else f" (auto, pad={parsed_args.padding})")
    )
    console.print(summary_table)

    group_cols = ["backend", "backend_memtype", "num_procs", "num_procs_node"]
    unique_groups = sel_runs[group_cols].drop_duplicates()

    for _, row in unique_groups.iterrows():
        backend = str(row["backend"])
        memtype = str(row["backend_memtype"])
        num_procs = float(row["num_procs"])
        num_procs_node = float(row["num_procs_node"])

        subset = sel_runs[
            (sel_runs["backend"] == backend)
            & (sel_runs["backend_memtype"] == memtype)
            & (sel_runs["num_procs"] == num_procs)
            & (sel_runs["num_procs_node"] == num_procs_node)
        ]

        if subset.empty:
            continue

        plot_single_configuration(
            pl_runs=subset,
            backend=backend,
            backend_memtype=memtype,
            num_procs=num_procs,
            num_procs_node=num_procs_node,
            test_short=test_short,
            vdim=vdim,
            code=code_to_use,
            x_range=effective_x_range,
            y_range=effective_y_range,
            args=parsed_args,
            console=console,
        )

    if parsed_args.show_figures:
        console.print("[bold yellow]Displaying interactive figures...[/bold yellow]")
        plt.show()

    return 0


if __name__ == "__main__":
    sys.exit(main())
