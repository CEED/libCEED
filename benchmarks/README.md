# libCEED: Benchmarks

This directory contains benchmark problems for performance evaluation of libCEED backends.

## Running the Benchmarks

Example:
```sh
benchmark.sh -c /cpu/self -r petsc-bpsraw.sh -b bp1 -n 16 -p 16
```
where the option `-c <specs-list>` specifies a list of libCEED specs to benchmark, `-b <bp-list>` specifies a list of CEED benchmark problems to run, `-n 16` is the total number of processors and `-p 16` is the number of processors per node.

Multiple backends, benchmark problems, and processor configurations can be benchmarked with:
```sh
benchmark.sh -c "/cpu/self/ref/serial /cpu/self/ref/blocked" -r petsc-bpsraw.sh -b "bp1 bp3" -n "16 32 64" -p "16 32 64"
```

The results from the benchmarks are written to files named `*-output.txt`.

For a short help message, use the option `-h`.

When running the tests `petsc-bpsraw.sh`, the following variables can be set on the command line:
* `max_dofs_node=<number>`, e.g. `max_dofs_node=1000000` - this sets the upper bound of the problem sizes, per compute node; the default value is $3 \times 2^{20}$.
* `max_p=<number>`, e.g. `max_p=12` - this sets the highest degree for which the tests will be run (the lowest degree is 1); the default value is 8.

## Post-Processing the Results

Post-processing tools parse output logs to generate comparative performance tables and scaling plots. Both scripts support multiple input files, stdin streams, and configurable command-line options via `argparse`.

### Prerequisites

Install the required Python dependencies:
```sh
pip install matplotlib pandas rich numpy
```

---

### Tabular Summaries (`postprocess_table.py`)

`postprocess_table.py` parses benchmark logs into structured data. It displays a formatted terminal summary, exports delimited files (TSV/CSV), and generates publication-ready LaTeX tables (`booktabs`).

```sh
# Display rich summary table and save to default TSV (benchmark_data.csv)
python postprocess_table.py petsc-bpsraw-bp1-*-output.txt

# Export to a custom CSV file
python postprocess_table.py *.log -o benchmark_summary.csv --sep ","

# Generate a LaTeX booktabs table file
python postprocess_table.py *.log \
    --latex-output table.tex \
    --latex-caption "CEED Benchmark Performance" \
    --latex-label "tab:ceed_benchmarks"

# Print LaTeX markup directly to stdout (e.g., for clipboard copy)
python postprocess_table.py *.log --print-latex --no-show-table

# Pipe log output directly from stdin
cat petsc-bpsraw-bp1-*-output.txt | python postprocess_table.py --rows 25
```

#### Key Arguments for `postprocess_table.py`
| Argument | Description | Default |
| :--- | :--- | :--- |
| `files` | Input log file paths (reads from stdin if omitted) | `stdin` |
| `-o`, `--output` | Destination path for delimited output | `benchmark_data.csv` |
| `--sep` | Field delimiter string | `\t` |
| `--latex-output` | Output `.tex` file destination | `None` |
| `--latex-caption` | Caption string for LaTeX table environment | `"CEED Benchmark Performance Results"` |
| `--latex-label` | Cross-reference label for LaTeX table | `"tab:ceed_benchmarks"` |
| `--print-latex` | Print LaTeX source code to standard output | `False` |
| `--rows` | Maximum rows to preview in the terminal | `15` |
| `--no-show-table` | Suppress the rich terminal preview | `False` |

---

### Scaling Plots (`postprocess_plot.py`)

`postprocess_plot.py` creates $[\text{DOFs} \times \text{CG iterations}] / [\text{nodes} \times \text{seconds}]$ vs. problem size scaling curves grouped by backend and rank configuration. Plot boundaries default to the global minimum and maximum across all filtered records, ensuring identical, cross-comparable axes across all generated figures.

```sh
# Generate and save PDF figures using auto-computed global scales
python postprocess_plot.py petsc-bpsraw-*-output.txt

# Filter by specific benchmark problem and case
python postprocess_plot.py *.log --test "CEED Benchmark Problem 1" --case scalar

# Use logarithmic y-axis and overlay reference iteration/second slopes
python postprocess_plot.py *.log --log-y --draw-iter-lines

# Set exact unpadded boundaries (disable margin padding)
python postprocess_plot.py *.log --padding 0.0

# Override with explicit domain and range boundaries
python postprocess_plot.py *.log \
    --x-range 1e2 1e7 \
    --y-range 1e6 2e9 \
    --output-dir ./figures \
    --format png

# Interactive preview without saving files
python postprocess_plot.py *.log --show --no-save
```

#### Key Arguments for `postprocess_plot.py`
| Argument | Description | Default |
| :--- | :--- | :--- |
| `files` | Input log file paths (reads from stdin if omitted) | `stdin` |
| `--test` | Filter logs by test/BP name | First test in log |
| `--case` | Filter by problem type (`scalar` or `vector`) | Auto-detected |
| `--code` | Filter by framework/code name | First code in log |
| `--log-y` | Plot y-axis on logarithmic scale | Linear |
| `--x-range XMIN XMAX` | Explicit domain boundaries for points per node | Auto (data min/max) |
| `--y-range YMIN YMAX` | Explicit range boundaries for DOFs/s per node | Auto (data min/max) |
| `--padding` | Fractional margin added to auto-computed bounds | `0.05` ($5\%$) |
| `--draw-iter-lines` | Overlay reference iteration rate guidelines | `False` |
| `--output-dir` | Directory where generated figures are saved | Current directory |
| `--format` | Graphic output format (`pdf`, `png`, `svg`) | `pdf` |
| `--show` | Display figures interactively in a GUI window | `False` |
| `--no-save` | Skip saving figures to disk | `False` |
