## Nek5000 + libCEED Examples

### Prerequisites

Nek5000 v18.0 or greater must be [installed](https://nek5000.mcs.anl.gov/getstarted/) to run
these examples.  It is assumed to exist at `../../../Nek5000` (a sibling to the
libCEED directory) or at a path defined in the environment variable `NEK5K_DIR`.
For example, you could set
```sh
    export NEK5K_DIR=/scratch/Nek5000
```
if that is where it is located.

The Nek5000 examples depend on the Nek5000 tools: `genbox`, `genmap`,
and `reatore2`. They can be built using
```sh
   ( cd $NEK5K_DIR/tools && ./maketools genbox genmap reatore2 )
```
See also the [Nek5000 documentation](https://nek5000.mcs.anl.gov/getstarted/).

### Building the Nek5000 examples

You can build the Nek5000 libCEED examples with the command `make bps`.

You can also build the Nek5000 libCEED examples by invoking `nek-examples.sh` script.
```sh
  ./nek-examples.sh -m
```

By default, the examples are built with MPI. To build the examples without MPI,
set the environment variable `MPI=0`.

Note: Nek5000 examples must be built sequentially. Due to the Nek5000 build
process, multiple examples cannot be built in parallel. At present, there is
only one Nek5000 example file to build, which handles both CEED BP 1 and
CEED BP 3.

### Running Nek5000 examples

You can run the Nek5000 libCEED examples by invoking `nek-examples.sh`
script. The syntax is:
```sh
  ./nek-examples.sh -c <ceed_backend> -e <example_name> \
                   -n <mpi_ranks> -b <box_geometry>
```
The different options that can be used for the script are listed below:
```
options:
   -h|-help     Print this usage information and exit
   -c|-ceed     Ceed backend to be used for the run (optional, default: /cpu/self)
   -e|-example  Example name (optional, default: bp1)
   -n|-np       Specify number of MPI ranks for the run (optional, default: 1)
   -t|-test     Run in test mode (not on by default)
   -b|-box      Box case in boxes sub-directory found along with this script (default: 2x2x2)
   -clean       clean the examples directory
   -m|-make     Make the examples
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
