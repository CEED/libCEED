# libCEED Benchmarks

This directory contains benchmark problems for performance evaluation of libCEED
backends.

## Running the Benchmarks

Example:
```sh
benchmark.sh -c /cpu/self -r petsc-bp1.sh -n 16 -p 16
```
where the option `-c <specs-list>` specifies a list of libCEED specs to
benchmark, `-n 16` is the total number of processors and `-p 16` is the number
of processors per node.

Multiple backends and multiple processor configurations can be benchmarked with:
```sh
benchmark.sh -c "/cpu/self /cpu/self/blocked" -r petsc-bp1.sh -n "16 32 64" -p "16 32 64"
```

The results from the benchmarks are written to files named `*-output.txt`.

For a short help message, use the option `-h`.

The following variables can be set on the command line:
* `max_dofs_node=<number>`, e.g. `max_dofs_node=1000000` - this sets the upper
  bound of the problem sizes, per compute node; the default value is 3*2^20.
* `max_p=<number>`, e.g. `max_p=12` - this sets the highest degree for which the
  tests will be run (the lowest degree is 1); the default value is 8.

## Post-processing the results

After generating the results, use the `postprocess-plot-2.py` script (which
requires the python package matplotlib) or the `postprocess-table.py` script,
e.g.:
```sh
python postprocess-plot-2.py petsc-bp1-output.txt
```
The plot ranges and some other options can be adjusted by editing the values
in the beginning of the script `postprocess-plot-2.py`.

Note that the `postprocess-*.py` scripts can read multiple files at a time just
by listing them on the command line and also read the standard input if no files
were specified on the command line.
