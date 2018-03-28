## Running Nek5000 examples

### Prerequisites

Nek5000 must be [installed](https://nek5000.mcs.anl.gov/getstarted/) to
run these examples.  It is assumed to exist at `../../../Nek5000` (a
sibling to the libCEED directory) or at a path defined in the
environment variable `NEK5K_DIR`.  For example, you could set

    export NEK5K_DIR=/scratch/Nek5000

if that is where it is located.

### Generate meshes (boxes)

You can generate box geometries using `generate-boxes.sh` script, with
usage

```sh
   ./generate_boxes log_2(<min_elem>) log_2(<max_elem>)."
```

Example:
```sh
./generate-boxes 2 4
```
This will generate three boxes with 4(=2^2), 8 and 16(=2^4) elements inside
`boxes/b*` directories.

This script depends on the Nek5000 tools: `genbox`, `genmap`, and
`reatore2`.  They can be built using

    ( cd $NEK5K_DIR/tools && ./maketools genbox genmap reatore2 )

(see also the [Nek5000 documentation](https://nek5000.mcs.anl.gov/getstarted/)).


### Make Nek5000 examples

You can make Nek5000 examples by invoking `make-nek-examples.sh` script.
```sh
./make-nek-examples.sh
```

### Run Nek5000 examples

You can run Nek5000 examples by invoking `run-nek-examples.sh` script.
Syntax for the command is
```sh
  ./run-nek-example <example_name> <#mpi_ranks> <rea_name> <rea_and_map_path>"
```

Example:
```
  ./run-nek-example ex1 4 b3 boxes/b3
```
