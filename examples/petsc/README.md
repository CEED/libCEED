## libCEED + PETSc Examples

### CEED bakeoff problems - bps

This code solves the CEED bakeoff problems on a structured grid generated and
referenced using only low-level communication primitives.

To build, run `make bpsraw`

To run, `./bpsraw -ceed [ceed-resource] -problem bp[1-6] -degree [degree]`

In addition to the common arguments, the following arguments may be set:

- `-local`             - Target number of locally owned DoFs per process

### CEED bakeoff problems with DMPlex - bps

This code solves the CEED bakeoff problems on a unstructured grid using DMPlex.
This example requires a PETSc version later than 3.11.3.

To build, run `make bps`

To run, `./bps -ceed [ceed-resource] -problem bp[1-6] -degree [degree]`

In addition to the common arguments, the following arguments may be set:

- `-mesh`              - Read mesh from file
- `-cells`             - Number of cells per dimension

### CEED bakeoff problems with DMPlex and PCMG - multigrid

This code solves the CEED bakeoff problems on a unstructured grid using DMPlex
with p-multigrid implemented in PCMG. This example requires a PETSc version later than 3.11.3.

To build, run `make multigrid`

To run, `./multigrid -ceed [ceed-resource] -problem bp[1-6] -degree [degree]`

In addition to the common arguments, the following arguments may be set:

- `-mesh`              - Read mesh from file
- `-cells`             - Number of cells per dimension

### Command line arguments

The following arguments can be specified for all of the above examples:

- `-ceed`              - CEED resource specifier
- `-problem`           - CEED benchmark problem to solve
- `-degree`            - Polynomial degree of tensor product basis
- `-qextra`            - Number of extra quadrature points
- `-test`              - Testing mode (do not print unless error is large)
- `-benchmark`         - Benchmarking mode (prints benchmark statistics)

### libCEED examples to compute surface area using DMPlex - areacube and areasphere

These examples use the mass matrix to compute the surface area of a cube or a discrete cubed-sphere, defined via DMPlex.

To build, run `make area`

To run, `./area -problem cube -ceed [ceed-resource] -petscspace_degree [degree]`

or

`./area -problem sphere -ceed [ceed-resource] -petscspace_degree [degree]`

### Command line arguments

The following arguments can be specified for these examples:

- `-ceed`              - CEED resource specifier
- `-problem`           - Problem to solve, either 'cube' or 'sphere'
- `-petscspace_degree` - Polynomial degree of tensor product basis
- `-qextra`            - Number of extra quadrature points
- `-test`              - Testing mode (do not print unless error is large)
- `-mesh`              - Read mesh from file

