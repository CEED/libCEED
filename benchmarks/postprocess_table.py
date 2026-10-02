#!/usr/bin/env python3

# Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and other CEED contributors.
# All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
#
# SPDX-License-Identifier: BSD-2-Clause
#
# This file is part of CEED:  http://github.com/ceed

"""
Convert CEED benchmark logs into delimited data files (CSV/TSV), terminal preview tables,
or publication-ready LaTeX booktabs tables.

Example Usage:
    # 1. Parse log files, display a styled summary table in the terminal,
    #    and export to the default TSV file (benchmark_data.csv):
    python postprocess_table.py petsc-bpsraw-*-output.txt

    # 2. Export benchmark data to a custom CSV file with comma separation:
    python postprocess_table.py run1.log run2.log -o benchmark_data.csv --sep ","

    # 3. Generate a LaTeX table file with custom caption and cross-reference label:
    python postprocess_table.py *.log \
        --latex-output table.tex \
        --latex-caption "CEED BP1 Scaling on AMD MI300A" \
        --latex-label "tab:ceed_bp1_mi300a"

    # 4. Print LaTeX markup directly to stdout (e.g., for piping or copying)
    #    while suppressing the terminal rich table:
    python postprocess_table.py output.txt --print-latex --no-show-table

    # 5. Stream input directly from stdin and increase terminal preview rows:
    cat petsc-bpsraw-bp1-*-output.txt | python postprocess_table.py --rows 30
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import pandas as pd
from rich.console import Console
from rich.table import Table

from postprocess_base import read_logs


def parse_args(args: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for benchmark table generation."""
    parser = argparse.ArgumentParser(
        description="Convert CEED benchmark logs to CSV/TSV, rich preview, or publication-ready LaTeX tables."
    )
    parser.add_argument(
        "files",
        nargs="*",
        type=Path,
        help="Input log files to read (reads from stdin if omitted).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("benchmark_data.csv"),
        help="Output CSV/TSV file path (default: benchmark_data.csv).",
    )
    parser.add_argument(
        "--sep",
        type=str,
        default=None,
        help="Column separator for exported delimited table (default: based on extension of `-o`, or ',').",
    )
    parser.add_argument(
        "--latex-output",
        type=Path,
        default=None,
        help="Destination path to write a standalone or snippet LaTeX table file (.tex).",
    )
    parser.add_argument(
        "--latex-caption",
        type=str,
        default="CEED Benchmark Performance Results",
        help="Caption for the LaTeX table environment.",
    )
    parser.add_argument(
        "--latex-label",
        type=str,
        default="tab:ceed_benchmarks",
        help="Label for cross-referencing the LaTeX table.",
    )
    parser.add_argument(
        "--print-latex",
        action="store_true",
        help="Print the generated LaTeX code directly to stdout.",
    )
    parser.add_argument(
        "--show-table",
        action="store_true",
        default=True,
        help="Print a rich summary table to the console (default: enabled).",
    )
    parser.add_argument(
        "--no-show-table",
        dest="show_table",
        action="store_false",
        help="Suppress rich terminal summary table.",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=15,
        help="Maximum rows to preview in terminal (default: 15).",
    )
    return parser.parse_args(args)


def render_rich_table(df: pd.DataFrame, max_rows: int = 15) -> Table:
    """Construct a styled rich Table from a pandas DataFrame."""
    table = Table(
        title=f"CEED Benchmark Results Summary (First {min(len(df), max_rows)} of {len(df)} runs)",
        header_style="bold magenta",
        show_lines=False,
    )

    columns_to_show: List[Tuple[str, str, str]] = [
        ("test", "Test / BP", "cyan"),
        ("backend", "Backend", "green"),
        ("case", "Case", "yellow"),
        ("degree", "Degree (P)", "blue"),
        ("quadrature_pts", "Quad Pts (Q)", "blue"),
        ("num_procs", "Total Ranks", "magenta"),
        ("num_elem", "Elements", "white"),
        ("cg_iteration_dps", "CG DOFs/s", "bright_green"),
    ]

    active_cols = [(col, label, style) for col, label, style in columns_to_show if col in df.columns]

    for _, label, style in active_cols:
        table.add_column(label, style=style, overflow="fold")

    for _, row in df.head(max_rows).iterrows():
        row_values: List[str] = []
        for col_name, _, _ in active_cols:
            val = row[col_name]
            if pd.isna(val):
                row_values.append("-")
            elif isinstance(val, float):
                row_values.append(f"{val:.3e}")
            else:
                row_values.append(str(val))
        table.add_row(*row_values)

    return table


def escape_latex(text: str) -> str:
    """Escape characters reserved by LaTeX."""
    replacements = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "&": r"\&",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for char, escaped in replacements.items():
        text = text.replace(char, escaped)
    return text


def format_scientific_latex(val: float, precision: int = 2) -> str:
    """Format floating point numbers into standard LaTeX scientific notation (\\(a \\times 10^{b}\\))."""
    if pd.isna(val) or val == 0:
        return "0"
    base_str = f"{val:.{precision}e}"
    mantissa, exponent = base_str.split("e")
    exp_int = int(exponent)
    if exp_int == 0:
        return f"\\({mantissa}\\)"
    return f"\\({mantissa} \\times 10^{{{exp_int}}}\\)"


def generate_latex_table(
    df: pd.DataFrame,
    caption: str = "CEED Benchmark Performance Results",
    label: str = "tab:ceed_benchmarks",
) -> str:
    """Generate a clean, booktabs-compliant LaTeX table snippet."""
    # Define mapping: (column_name, LaTeX_header, alignment, is_numeric, is_sci)
    column_specs: List[Tuple[str, str, str, bool, bool]] = [
        ("test", "Problem", "l", False, False),
        ("backend", "Backend", "l", False, False),
        ("case", "Case", "c", False, False),
        ("degree", "\\(p\\)", "r", True, False),
        ("quadrature_pts", "\\(q\\)", "r", True, False),
        ("num_procs", "Ranks", "r", True, False),
        ("num_elem", "Elements", "r", True, False),
        ("cg_iteration_dps", "CG DOFs/s", "r", True, True),
    ]

    active_specs = [spec for spec in column_specs if spec[0] in df.columns]
    alignments = "".join([spec[2] for spec in active_specs])
    headers = [spec[1] for spec in active_specs]

    lines: List[str] = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \small",
        f"  \\caption{{{escape_latex(caption)}}}",
        f"  \\label{{{escape_latex(label)}}}",
        f"  \\begin{{tabular}}{{{alignments}}}",
        r"    \toprule",
        "    " + " & ".join(headers) + r" \\",
        r"    \midrule",
    ]

    for _, row in df.iterrows():
        row_cells: List[str] = []
        for col_name, _, _, is_num, is_sci in active_specs:
            val = row[col_name]
            if pd.isna(val):
                row_cells.append("--")
            elif is_sci and isinstance(val, (int, float)):
                row_cells.append(format_scientific_latex(float(val)))
            elif is_num and isinstance(val, (int, float)):
                row_cells.append(f"{int(val):,}" if float(val).is_integer() else f"{val:.2f}")
            else:
                row_cells.append(escape_latex(str(val)))
        lines.append("    " + " & ".join(row_cells) + r" \\")

    lines.extend([
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def export_dataframe(df: pd.DataFrame, output_path: Path, sep: str, console: Console) -> None:
    """Export benchmark DataFrame to CSV/TSV on disk[cite: 1]."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, sep=sep, index=False)
    console.print(
        f"[bold green]✔[/bold green] Wrote [bold]{len(df)}[/bold] records to "
        f"[cyan]{output_path}[/cyan] (delimiter: [italic]{repr(sep)}[/italic])"
    )


def export_latex(
    latex_str: str,
    output_path: Path,
    console: Console,
) -> None:
    """Save the generated LaTeX table to a .tex file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(latex_str, encoding="utf-8")
    console.print(
        f"[bold green]✔[/bold green] LaTeX table saved to [cyan]{output_path}[/cyan]"
    )


def main(args: Optional[Sequence[str]] = None) -> int:
    parsed_args = parse_args(args)
    console = Console()

    file_inputs = [str(p) for p in parsed_args.files] if parsed_args.files else None

    with console.status("[bold blue]Parsing benchmark logs...", spinner="dots"):
        runs = read_logs(file_inputs)  # [cite: 1]

    if runs.empty:
        console.print("[bold red]Error:[/bold red] No benchmark records found in input source.")
        return 1

    # Rich terminal table output
    if parsed_args.show_table:
        table = render_rich_table(runs, max_rows=parsed_args.rows)
        console.print(table)

    # Standard tabular export (CSV / TSV)
    if parsed_args.sep is None:
        if parsed_args.output.suffix == ".csv":
            parsed_args.sep = ","
        elif parsed_args.output.suffix == ".tsv":
            parsed_args.sep = "\t"
        else:
            parsed_args.sep = ","
    export_dataframe(runs, parsed_args.output, parsed_args.sep, console)

    # Generate LaTeX table if requested via file output or direct print
    if parsed_args.latex_output or parsed_args.print_latex:
        latex_table = generate_latex_table(
            df=runs,
            caption=parsed_args.latex_caption,
            label=parsed_args.latex_label,
        )

        if parsed_args.latex_output:
            export_latex(latex_table, parsed_args.latex_output, console)

        if parsed_args.print_latex:
            print("\n" + latex_table + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
