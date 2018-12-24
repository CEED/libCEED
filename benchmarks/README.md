# CEED Benchmarks

This directory contains bake-off/benchmark problems for performance
evaluation of high-order kernels on HPC architectures developed in
the ECP co-design [Center for Efficient Exascale Discretizations
(CEED)](http://ceed.exascaleproject.org).

For more details on the CEED benchmarks see http://ceed.exascaleproject.org/bps/

## Running the Benchmarks

Example:
```sh
benchmark.sh -r petsc-bp1.sh -n 16 -p 16
```
where `-n 16` is the total number of processors and `-p 16` is the number of
processors per node.

Multiple processor configuration can be run with:
```sh
benchmark.sh -r petsc-bp1.sh -n "16 32 64" -p "16 32 64"
```

The following variables can be set on the command line:
* `ceed=<libceed-device-spec>`, e.g. `ceed=/cpu/self/ref`; the default value is
  `/cpu/self`.
* `max_dofs_node=<number>`, e.g. `max_dofs_node=1000000` - this sets the upper
  bound of the problem sizes, per compute node; the default value is 3*2^20.
* `max_p=<number>`, e.g. `max_p=12` - this sets the highest degree for which the
  tests will be run (the lowest degree is 1); the default value is 8.

## Post-processing the results

First, save the output of the run to a file:
```sh
benchmark.sh -r petsc-bp1.sh -n 16 -p 16 > petsc-bp1-output.txt
```
and then use the `postprocess-plot-2.py` script (which requires the python
package matplotlib) or the `postprocess-table.py` script, e.g.:
```sh
python postprocess-plot-2.py petsc-bp1-output.txt
```
The plot ranges and some other options can be adjusted by editing the values
in the beginning of the script `postprocess-plot-2.py`.

Note that the `postprocess-*.py` scripts can read multiple files at a time just
by listing them on the command line and also read the standard input if no files
were specified on the command line.
