## Running Nek5000 examples

### Generate meshes (boxes)

You can generate box geometries using `generate-boxes.sh` script. Syntax
for using the script is the following.
```sh
   generate_boxes log_2(<min_elem>) log_2(<max_elem>)."
```

Example:
```sh
./generate-boxes 2 4
```
This will generate three boxes with 4(=2^2), 8 and 16(=2^4) elements inside
`boxes/b*` directories.

### Make Nek5000 examples

You can make Nek5000 examples by invoking `make-nek-examples.sh` script.
```sh
./make-nek-examples.sh
```

You can set the path to Nek5000 installation directory and the example
names you want to be built by editing this script.

### Run Nek5000 examples

You can run Nek5000 examples by invoking `make-nek-examples.sh` script.
Syntax for the command is 
```sh
  ./run-nek-example <example_name> <#mpi_ranks> <rea_name> <rea_and_map_path>"
```

Example:
```
  echo "Example ./run-nek-example ex1 4 b10 ./boxes/b10"
```
