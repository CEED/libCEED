## libCEED: Solid Mechanics Example

This page provides a description of the solid mechanics example for the libCEED library, based on PETSc.

This code is able to solve boundary value linear elasticity, hyperelasticity at small strain and
hyperelasticity at finite strain (large deformation) using libCEED and PETSc. The hyperelasticity
at finite strain formulation is Total Lagrangian.

Build by using

`make`

and run with

`./elasticity -mesh [.exo file] -degree [degree] -nu [nu] -E [E] -boundary [boundary] -forcing [forcing] -problem [problem type] -multigrid [multigrid type]`

Available runtime options are:

|  Option                  | Meaning                                                                                         |
| :----------------------- | :-----------------------------------------------------------------------------------------------|
| `-ceed`                  | CEED resource specifier                                                                         |
| `-ceed_fine`             | CEED resource specifier for fine grid (P >= 5)                                                  |
| `-test`                  | Run in test mode                                                                                |
| `-degree`                | Polynomial degree of tensor product basis                                                       |
| `-mesh`                  | Filepath to Exodus-II file                                                                      |
| `-problem`               | Problem to solve (`linElas`, `hyperSS`, or `hyperFS`)                                           |
| `-forcing`               | Forcing term option (`none`, `constant`, or `mms`)                                              |
| `-bc_zero`               | List of boundary face IDs to apply zero Dirichlet BCs                                           |
| `-bc_clamp`              | List of boundary face IDs to apply incremental -y Dirichlet BCs                                 |
| `-bc_clamp_max`          | Maximum value to displace clamped boundary                                                      |
| `-num_steps`             | Number of pseudo-time steps for continuation method                                             |
| `-view_soln`             | Output solution at each pseudo-time step for viewing                                            |
| `-E`                     | Young's modulus                                                                                 |
| `-nu`                    | Poisson's ratio                                                                                 |
| `-units_meter`           | 1 meter in scaled length units                                                                  |
| `-units_second`          | 1 second in scaled time units                                                                   |
| `-units_kilogram`        | 1 kilogram in scaled mass units                                                                 |

## Boundary

Setting boundaries is mesh dependent in every FEM problem. As a result, the examples we have provided here depend on the  mesh files in `meshes\` folder. However, this code is capable of importing any structured or structured ExodusII (.exo) mesh file. In such cases, the user is responsible for providing boundary functions in `setup.h`. Currently, the following boundary functions have been implemented in `setup.h` for our purpose: `BCMMS(...)`, `BCBend2_ss(...)` and `BCBend1_ss(...)`. The function signature for these boundary functions in `elasticity.h` must be used as required by PETSc. We have used Trelis/Cubit software to generate meshes. The journal file (`.jou`) is provided in the `meshes\` directory. We have employed the `sideset` feature from Trelis/Cubit software to select different regions of the geometry to insert boundary values in the solution vector corresponding to those regions. The `sideset` feature is the appropriate choice to handle *essential* (Dirichlet) boundary values as this code runs with high-order polynomials and all points on a face (`sideset`) need to be considered. The specific `sideset` numbers in the image below have been employed in our boundary functions. Note that `nodeset` feature from Trelis/Cubit software is considered inappropriate for the purposes of choosing boundary regions in the mesh. This is due internal workings of PETSc. Everything else about the code is general.

### Forcing Functions

Two forcing functions may be used for any problem, `none` (no force) or `constant` (constant force in `-y` direction. u[1] = -1).
At least one boundary condition must be set for these forcing functions.
The boundaries can be constrained to zero or a unit decrement in the `-y` direction.

Example:\
 `./elasticity -mesh ./meshes/cylinder8_672e_4ss_us.exo -degree 2 -E 1e6 -nu .3 -bc_zero 999 -bc_clamp 998 -forcing constant`

To test the linear elasticity code, the forcing term can be set instead to `mms`.

Example:\
 `./elasticity -mesh ./meshes/cylinder8_672e_4ss_us.exo -degree 2 -E 1e6 -nu .3 -forcing mms`
