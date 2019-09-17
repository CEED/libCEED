## libCEED + PETSc Examples

### CEED bakeoff problems - bps

This code solves the CEED bakeoff problems on a structured grid generated and
referenced using only low-level communication primitives.

To build, run `make bps`.

To run, `./bps -ceed [ceed-resource] -problem bp[1-6] -degree [degree]`.

In addition to the common arguments, the following arguments may be set:

- `-local`             - Target number of locally owned DoFs per process
- `-degree`            - Polynomial degree of tensor product basis

### CEED bakeoff problems with DMPlex - bpsdmplex

This code solves the CEED bakeoff problems on a unstructured grid using DMPlex.
This example requires a PETSc version later than 3.11.3.

To build, run `make bpsdmplex`.

To run, `./bpsdmplex -ceed [ceed-resource] -problem bp[1-6] -petscspace_degree [degree]`.

In addition to the common arguments, the following arguments may be set:

- `-mesh`              - Read mesh from file
- `-cells`             - Number of cells per dimension
- `-petscspace_degree` - Polynomial degree of tensor product basis (Mandatory)
- `-enforce_bc`        - Enforce essential BCs

### Command line arguments

The following arguments can be specified for both examples:

- `-ceed`              - CEED resource specifier
- `-problem`           - CEED benchmark problem to solve
- `-qextra`            - Number of extra quadrature points
- `-test`              - Testing mode (do not print unless error is large)
- `-benchmark`         - Benchmarking mode (prints benchmark statistics)
