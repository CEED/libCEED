## Nek5000 + libCEED examples

### Prerequisites

Nek5000 must be [installed](https://nek5000.mcs.anl.gov/getstarted/) to run
these examples.  It is assumed to exist at `../../../Nek5000` (a sibling to the
libCEED directory) or at a path defined in the environment variable `NEK5K_DIR`.
For example, you could set
```sh
    export NEK5K_DIR=/scratch/Nek5000
```
if that is where it is located.

### Generate meshes (boxes)

You can generate box geometries using `generate-boxes.sh` script:
```sh
  ./generate-boxes.sh log_2(<min_elem>) log_2(<max_elem>)."
```
For example:
```sh
  ./generate-boxes.sh 2 4
```
will generate three boxes with 4(=2^2), 8 and 16(=2^4) elements inside the
`boxes/b*` directories.

The `generate-boxes.sh` script depends on the Nek5000 tools: `genbox`, `genmap`,
and `reatore2`. They can be built using
```sh
   ( cd $NEK5K_DIR/tools && ./maketools genbox genmap reatore2 )
```
See also the [Nek5000 documentation](https://nek5000.mcs.anl.gov/getstarted/).

### Building the Nek5000 examples

You can build the Nek5000 libCEED examples by invoking `make-nek-examples.sh` script.
```sh
  ./make-nek-examples.sh
```

### Running Nek5000 examples

You can run the Nek5000 libCEED examples by invoking `run-nek-examples.sh`
script. The syntax is:
```sh
  ./run-nek-example.sh -c <ceed_backend> -e <example_name> \
                                            -n <mpi_ranks> -b <box_geometry>
```
The different options that can be used for the script are listed below:
```
options:
   -h|-help     Print this usage information and exit
   -c|-ceed     Ceed backend to be used for the run (optional, default: /cpu/self)
   -e|-example  Example name (optional, default: bp1)
   -n|-np       Specify number of MPI ranks for the run (optional, default: 4)
   -b|-box      Specify the box geometry to be found in ./boxes/ directory (Mandatory)
```
The only mandatory argument is `-b` or `-box` which sets the box geometry to be
used. This geometry should be found in `./boxes` directory.

For example, you can run bp1 as follows:
```sh
  ./run-nek-example.sh -ceed /cpu/self -e bp1 -n 4 -b 3
```
which is the same as running:
```sh
  ./run-nek-example.sh -b 3
```
