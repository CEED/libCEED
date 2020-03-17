## libCEED + PETSc Examples
This code is able to solve boundary value linear elasticity, hyperelasticity at small strain and
hyperelasticity at finite strain (large deformation) using libCEED and PETSc. The hyperlesticity
at finite strain formulation is Total Lagrangian. This code was tested in valgrind with the
following options: `--track-origins=yes` `--leak-check=full` `--show-leak-kinds=all`

**Boundary:**

Setting boundary is mesh dependent in every FEM problem. As a result, the examples we have provided here depend on the  mesh files in `meshes\` folder. However, this code is capable of importing any structured or structured ExodusII (.exo) mesh file. In such cases, the user is responsible for providing boundary functions in `setup.h`. Currently, the following boundary functions have been implemented in `setup.h` for our purpose: `BCMMS(...)`, `BCBend2_ss(...)` and `BCBend1_ss(...)`. The function signature for these boundary functions in `setup.h` must be used as required by PETSc. We have used Trelis/Cubit software to generate meshes. The journal file (`.jou`) is provided in the `meshes\` directory. We have employed the `sideset` feature from Trelis/Cubit software to select different regions of the geometry to insert boundary values in the solution vector corresponding to those regions. The `sideset` feature is the appropriate choice to handle *essential* (Dirichlet) boundary values as this code runs with high-order polynomials and all points on a face (`sideset`) need to be considered. The specific `sideset` numbers in the image below have been employed in our boundary functions. Note that `nodeset` feature from Trelis/Cubit software is considered inappropriate for the purposes of choosing boundary regions in the mesh. This is due internal workings of PETSc. Everything else about the code is general.

![Image of one finger of a glove](pictures/gloveFinger.png)

**General Notes about mesh file naming conventions in `meshes\` folder:**

As an example, consider `cyl-hole_672e_4ss_us.exo`, `cyl-hole_672e_2ss_us.exo` and `cyl-hole_672e_1ss_us.exo` file names:\
   `_4ss` refers to the left, right, inner and outer *walls* of the image above.\
   `_2ss` refers to the left and right *walls* of the image above.\
   `_1ss` refers to the left *wall* of the image above.\
   `_672e` in the mesh file name means 672 elements.\
   `_us` means `unstructured mesh`.\
   `_s` means `structured mesh`.


### CEED/PETSc Linear Elasticity problem

To build, run `make`

To run:\
 `./elasticity -mesh [.exo file]  -degree [degree] -nu [nu] -E [E] -boundary [boundary] -forcing [forcing]`\
 or\
  `mpirun -n [n] ./elasticity -mesh [.exo file]  -degree [degree] -nu [nu] -E [E] -boundary [boundary] -forcing [forcing]`

`mms` stands for Method of Manufactured Solutions. In our case `mms` is based on the following contrived solution:

`u[0] = exp(2x)sin(3y)cos(4z)`\
`u[1] = exp(3y)sin(4z)cos(2x)`\
`u[2] = exp(4z)sin(2x)cos(3y)`

**Note 1:** For the `mms` to work correctly, you must use: \
            mesh files with `_4ss_` in their name from `meshes\` directory.\
            `-boundary mms` and `-forcing mms` options.

Example:\
 `./elasticity -mesh ./meshes/cyl-hole_672e_4ss_us.exo -degree 2 -nu .3 -E 1e6 -boundary mms -forcing mms`

**Note 2:** Two other boundary and forcing functions may be used with this mesh files provided in `meshes\`:

**1)** left side of the `cyl-hol` object (one finger of a glove) is attached to a wall (hand):\
       mesh files with `_1ss` must be used.\
       `-boundary wall_none` must be used.\
       forcing function on that could be `none` (no force) or `constant` (constant force in `-y` direction. u[1] = -1)

Example:\
 `./elasticity -mesh ./meshes/cyl-hole_672e_1ss_us.exo -degree 2 -nu .3 -E 1e6 -boundary wall_none -forcing constant`

**2)** left side of the `cyl-hol` object (one finger of a glove) is attached to a wall (hand) **and** the right side of it has a dead wight hanging off of it:\
   mesh files with `_2ss` must be used.\
   `-boundary wall_weight` must be used.\
   forcing function on that could be `none` (no force) or `constant` (constant force in `-y` direction. u[1] = -1)

Example:\
 `./elasticity -mesh ./meshes/cyl-hole_672e_2ss_us.exo -degree 2 -nu .3 -E 1e6 -boundary wall_weight -forcing constant`

### CEED/PETSc Hyperelasticity at small strain problem

To build, run `make`

To run, `./elasticity -mesh [.exo file]  -degree [degree] -nu [nu] -E [E] -problem [hyperSS] -boundary [boundary] -forcing [forcing]`

Example: `./elasticity -mesh ./meshes/cyl-hole_672e_2ss_us.exo -degree 2 -nu .3 -E 1e6 -problem hyperSS
-boundary wall -forcing none`

See figure `\pictures\gloveFinger.png`.

details will be updated when ready.

### CEED/PETSc Hyperelasticity at finite strain problem

To build, run `make`

To run, `./elasticity -mesh [.exo file]  -degree [degree] -nu [nu] -E [E] -problem [hyperFS] -boundary [boundary] -forcing [forcing]`

See figure `\pictures\gloveFinger.png`.

details will be updated when ready.
