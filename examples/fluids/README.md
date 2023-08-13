## libCEED: Navier-Stokes Example

This page provides a description of the Navier-Stokes example for the libCEED library, based on PETSc.
PETSc v3.17 or a development version of PETSc at commit 0e95d842 or later is required.

The Navier-Stokes problem solves the compressible Navier-Stokes equations in three dimensions using an explicit time integration.
The state variables are mass density, momentum density, and energy density.

The main Navier-Stokes solver for libCEED is defined in [`navierstokes.c`](navierstokes.c) with different problem definitions according to the application of interest.

Build by using:

`make`

and run with:

```
./navierstokes -ceed [ceed] -problem [problem type] -degree [degree]
```

## Runtime options

% inclusion-fluids-marker

The Navier-Stokes mini-app is controlled via command-line options.
The following options are common among all problem types:

:::{list-table} Common Runtime Options
:header-rows: 1

* - Option
  - Description
  - Default value

* - `-ceed`
  - CEED resource specifier
  - `/cpu/self/opt/blocked`

* - `-test_type`
  - Run in test mode and specify whether solution (`solver`) or turbulent statistics (`turb_spanstats`) output should be verified
  - `none`

* - `-compare_final_state_atol`
  - Test absolute tolerance
  - `1E-11`

* - `-compare_final_state_filename`
  - Test filename
  -

* - `-problem`
  - Problem to solve (`advection`, `advection2d`, `density_current`, or `euler_vortex`)
  - `density_current`

* - `-implicit`
  - Use implicit time integartor formulation
  -

* - `-degree`
  - Polynomial degree of tensor product basis (must be >= 1)
  - `1`

* - `-q_extra`
  - Number of extra quadrature points
  - `0`

* - `-ts_monitor_solution`
  - PETSc output format, such as `cgns:output-%d.cgns` (requires PETSc `--download-cgns`)
  -

* - `-ts_monitor_solution_interval`
  - Number of time steps between visualization output frames.
  - `1`

* - `-viewer_cgns_batch_size`
  - Number of frames written per CGNS file if the CGNS file name includes a format specifier (`%d`).
  - `20`

* - `-checkpoint_interval`
  - Number of steps between writing binary checkpoints. `0` has no output, `-1` outputs final state only
  - `10`

* - `-checkpoint_vtk`
  - Checkpoints include VTK (`*.vtu`) files for visualization. Consider `-ts_monitor_solution`instead.
  - `false`

* - `-viz_refine`
  - Use regular refinement for VTK visualization
  - `0`

* - `-output_dir`
  - Output directory for binary checkpoints and VTK files (if enabled).
  - `.`

* - `-output_add_stepnum2bin`
  - Whether to add step numbers to output binary files
  - `false`

* - `-continue`
  - Continue from previous solution (input is step number of previous solution)
  - `0`

* - `-continue_filename`
  - Path to solution binary file from which to continue from
  - `[output_dir]/ns-solution.bin`

* - `-continue_time_filename`
  - Path to time stamp binary file (only for legacy checkpoints)
  - `[output_dir]/ns-time.bin`

* - `-bc_wall`
  - Use wall boundary conditions on this list of faces
  -

* - `-wall_comps`
  - An array of constrained component numbers for wall BCs
  -

* - `-bc_slip_x`
  - Use slip boundary conditions, for the x component, on this list of faces
  -

* - `-bc_slip_y`
  - Use slip boundary conditions, for the y component, on this list of faces
  -

* - `-bc_slip_z`
  - Use slip boundary conditions, for the z component, on this list of faces
  -

* - `-bc_inflow`
  - Use inflow boundary conditions on this list of faces
  -

* - `-bc_outflow`
  - Use outflow boundary conditions on this list of faces
  -

* - `-bc_freestream`
  - Use freestream boundary conditions on this list of faces
  -

* - `-ts_monitor_turbulence_spanstats_collect_interval`
  - Number of timesteps between statistics collection
  - `1`

* - `-ts_monitor_turbulence_spanstats_viewer`
  - Sets the PetscViewer for the statistics file writing, such as `cgns:output-%d.cgns` (requires PETSc `--download-cgns`). Also turns the statistics collection on.
  -

* - `-ts_monitor_turbulence_spanstats_viewer_interval`
  - Number of timesteps between statistics file writing (`-1` means only at end of run)
  - `-1`

* - `-ts_monitor_turbulence_spanstats_viewer_cgns_batch_size`
  - Number of frames written per CGNS file if the CGNS file name includes a format specifier (`%d`).
  - `20`

* - `-ts_monitor_wall_force`
  - Viewer for the force on each no-slip wall, e.g., `ascii:force.csv:ascii_csv` to write a CSV file.
  -

* - `-snes_view`
  - View PETSc `SNES` nonlinear solver configuration
  -

* - `-log_view`
  - View PETSc performance log
  -

* - `-help`
  - View comprehensive information about run-time options
  -
:::

For the case of a square/cubic mesh, the list of face indices to be used with `-bc_wall`, `bc_inflow`, `bc_outflow`, `bc_freestream`  and/or `-bc_slip_x`, `-bc_slip_y`, and `-bc_slip_z` are:

:::{list-table} 2D Face ID Labels
:header-rows: 1
* - PETSc Face Name
  - Cartesian direction
  - Face ID

* - faceMarkerBottom
  - -z
  - 1

* - faceMarkerRight
  - +x
  - 2

* - faceMarkerTop
  - +z
  - 3

* - faceMarkerLeft
  - -x
  - 4
:::

:::{list-table} 3D Face ID Labels
:header-rows: 1
* - PETSc Face Name
  - Cartesian direction
  - Face ID

* - faceMarkerBottom
  - -z
  - 1

* - faceMarkerTop
  - +z
  - 2

* - faceMarkerFront
  - -y
  - 3

* - faceMarkerBack
  - +y
  - 4

* - faceMarkerRight
  - +x
  - 5

* - faceMarkerLeft
  - -x
  - 6
:::

### Boundary conditions

Boundary conditions for compressible viscous flows are notoriously tricky. Here we offer some recommendations

#### Inflow

If in a region where the flow velocity is known (e.g., away from viscous walls), use `bc_freestream`, which solves a Riemann problem and can handle inflow and outflow (simultaneously and dynamically).
It is stable and the least reflective boundary condition for acoustics.

If near a viscous wall, you may want a specified inflow profile.
Use `bc_inflow` and see {ref}`example-blasius` and discussion of synthetic turbulence generation for ways to analytically generate developed inflow profiles.
These conditions may be either weak or strong, with the latter specifying velocity and temperature as essential boundary conditions and evaluating a boundary integral for the mass flux.
The strong approach gives sharper resolution of velocity structures.
We have described the primitive variable formulation here; the conservative variants are similar, but not equivalent.

#### Outflow

If you know the complete exterior state, `bc_freestream` is the least reflective boundary condition, but is disruptive to viscous flow structures.
If thermal anomalies must exit the domain, the Riemann solver must resolve the contact wave to avoid reflections.
The default Riemann solver, HLLC, is sufficient in this regard while the simpler HLL converts thermal structures exiting the domain into grid-scale reflecting acoustics.

If acoustic reflections are not a concern and/or the flow is impacted by walls or interior structures that you wish to resolve to near the boundary, choose `bc_outflow`. This condition (with default `outflow_type: riemann`) is stable for both inflow and outflow, so can be used in areas that have recirculation and lateral boundaries in which the flow fluctuates.

The simpler `bc_outflow` variant, `outflow_type: pressure`, requires that the flow be a strict outflow (or the problem becomes ill-posed and the solver will diverge).
In our experience, `riemann` is slightly less reflective but produces similar flows in cases of strict outflow.
The `pressure` variant is retained to facilitate comparison with other codes, such as PHASTA-C, but we recommend `riemann` for general use.

#### Periodicity

PETSc provides two ways to specify periodicity:

1. Topological periodicity, in which the donor and receiver dofs are the same, obtained using:

```yaml
dm_plex:
  shape: box
  box_faces: 10,12,4
  box_bd: none,none,periodic
```

The coordinates for such cases are stored as a new field with special cell-based indexing to enable wrapping through the boundary.
This choice of coordinates prevents evaluating boundary integrals that cross the periodicity, such as for the outflow Riemann problem in the presence of spanwise periodicity.

2. Isoperiodicity, in which the donor and receiver dofs are distinct in local vectors. This is obtained using `zbox`, as in:

```yaml
dm_plex:
  shape: zbox
  box_faces: 10,12,4
  box_bd: none,none,periodic
```

Isoperiodicity enables standard boundary integrals, and is recommended for general use.
At the time of this writing, it only supports one direction of periodicity.
The `zbox` method uses [Z-ordering](https://en.wikipedia.org/wiki/Z-order_curve) to construct the mesh in parallel and provide an adequate initial partition, which makes it higher performance and avoids needing a partitioning package.

### Advection

For testing purposes, there is a reduced mode for pure advection, which holds density $\rho$ and momentum density $\rho \bm u$ constant while advecting "total energy density" $E$.
These are available in 2D and 3D.

#### 2D advection

For the 2D advection problem, the following additional command-line options are available:

:::{list-table} Advection2D Runtime Options
:header-rows: 1

* - Option
  - Description
  - Default value
  - Unit

* - `-rc`
  - Characteristic radius of thermal bubble
  - `1000`
  - `m`

* - `-units_meter`
  - 1 meter in scaled length units
  - `1E-2`
  -

* - `-units_second`
  - 1 second in scaled time units
  - `1E-2`
  -

* - `-units_kilogram`
  - 1 kilogram in scaled mass units
  - `1E-6`
  -

* - `-strong_form`
  - Strong (1) or weak/integrated by parts (0) residual
  - `0`
  -

* - `-stab`
  - Stabilization method (`none`, `su`, or `supg`)
  - `none`
  -

* - `-CtauS`
  - Scale coefficient for stabilization tau (nondimensional)
  - `0`
  -

* - `-wind_type`
  - Wind type in Advection (`rotation` or `translation`)
  - `rotation`
  -

* - `-wind_translation`
  - Constant wind vector when `-wind_type translation`
  - `1,0,0`
  -

* - `-E_wind`
  - Total energy of inflow wind when `-wind_type translation`
  - `1E6`
  - `J`
:::

An example of the `rotation` mode can be run with:

```
./navierstokes -problem advection2d -dm_plex_box_faces 20,20 -dm_plex_box_lower 0,0 -dm_plex_box_upper 1000,1000 -bc_wall 1,2,3,4 -wall_comps 4 -wind_type rotation -implicit -stab supg
```

and the `translation` mode with:

```
./navierstokes -problem advection2d -dm_plex_box_faces 20,20 -dm_plex_box_lower 0,0 -dm_plex_box_upper 1000,1000 -units_meter 1e-4 -wind_type translation -wind_translation 1,-.5 -bc_inflow 1,2,3,4
```
Note the lengths in `-dm_plex_box_upper` are given in meters, and will be nondimensionalized according to `-units_meter`.

#### 3D advection

For the 3D advection problem, the following additional command-line options are available:

:::{list-table} Advection3D Runtime Options
:header-rows: 1

* - Option
  - Description
  - Default value
  - Unit

* - `-rc`
  - Characteristic radius of thermal bubble
  - `1000`
  - `m`

* - `-units_meter`
  - 1 meter in scaled length units
  - `1E-2`
  -

* - `-units_second`
  - 1 second in scaled time units
  - `1E-2`
  -

* - `-units_kilogram`
  - 1 kilogram in scaled mass units
  - `1E-6`
  -

* - `-strong_form`
  - Strong (1) or weak/integrated by parts (0) residual
  - `0`
  -

* - `-stab`
  - Stabilization method (`none`, `su`, or `supg`)
  - `none`
  -

* - `-CtauS`
  - Scale coefficient for stabilization tau (nondimensional)
  - `0`
  -

* - `-wind_type`
  - Wind type in Advection (`rotation` or `translation`)
  - `rotation`
  -

* - `-wind_translation`
  - Constant wind vector when `-wind_type translation`
  - `1,0,0`
  -

* - `-E_wind`
  - Total energy of inflow wind when `-wind_type translation`
  - `1E6`
  - `J`

* - `-bubble_type`
  - `sphere` (3D) or `cylinder` (2D)
  - `sphere`
  -

* - `-bubble_continuity`
  - `smooth`, `back_sharp`, or `thick`
  - `smooth`
  -
:::

An example of the `rotation` mode can be run with:

```
./navierstokes -problem advection -dm_plex_box_faces 10,10,10 -dm_plex_dim 3 -dm_plex_box_lower 0,0,0 -dm_plex_box_upper 8000,8000,8000 -bc_wall 1,2,3,4,5,6 -wall_comps 4 -wind_type rotation -implicit -stab su
```

and the `translation` mode with:

```
./navierstokes -problem advection -dm_plex_box_faces 10,10,10 -dm_plex_dim 3 -dm_plex_box_lower 0,0,0 -dm_plex_box_upper 8000,8000,8000 -wind_type translation -wind_translation .5,-1,0 -bc_inflow 1,2,3,4,5,6
```

### Inviscid Ideal Gas

#### Isentropic Euler vortex

For the Isentropic Vortex problem, the following additional command-line options are available:

:::{list-table} Isentropic Vortex Runtime Options
:header-rows: 1

* - Option
  - Description
  - Default value
  - Unit

* - `-center`
  - Location of vortex center
  - `(lx,ly,lz)/2`
  - `(m,m,m)`

* - `-units_meter`
  - 1 meter in scaled length units
  - `1E-2`
  -

* - `-units_second`
  - 1 second in scaled time units
  - `1E-2`
  -

* - `-mean_velocity`
  - Background velocity vector
  - `(1,1,0)`
  -

* - `-vortex_strength`
  - Strength of vortex < 10
  - `5`
  -

* - `-c_tau`
  - Stabilization constant
  - `0.5`
  -
:::

This problem can be run with:

```
./navierstokes -problem euler_vortex -dm_plex_box_faces 20,20,1 -dm_plex_box_lower 0,0,0 -dm_plex_box_upper 1000,1000,50 -dm_plex_dim 3 -bc_inflow 4,6 -bc_outflow 3,5 -bc_slip_z 1,2 -mean_velocity .5,-.8,0.
```

#### Sod shock tube

For the Shock Tube problem, the following additional command-line options are available:

:::{list-table} Shock Tube Runtime Options
:header-rows: 1

* - Option
  - Description
  - Default value
  - Unit

* - `-units_meter`
  - 1 meter in scaled length units
  - `1E-2`
  -

* - `-units_second`
  - 1 second in scaled time units
  - `1E-2`
  -

* - `-yzb`
  - Use YZB discontinuity capturing
  - `none`
  -

* - `-stab`
  - Stabilization method (`none`, `su`, or `supg`)
  - `none`
  -
:::

This problem can be run with:

```
./navierstokes -problem shocktube -yzb -stab su -bc_slip_z 3,4 -bc_slip_y 1,2 -bc_wall 5,6 -dm_plex_dim 3 -dm_plex_box_lower 0,0,0 -dm_plex_box_upper 1000,100,100 -dm_plex_box_faces 200,1,1 -units_second 0.1
```

### Newtonian viscosity, Ideal Gas

For the Density Current, Channel, and Blasius problems, the following common command-line options are available:

:::{list-table} Newtonian Ideal Gas problems Runtime Options
:header-rows: 1

* - Option
  - Description
  - Default value
  - Unit

* - `-units_meter`
  - 1 meter in scaled length units
  - `1`
  -

* - `-units_second`
  - 1 second in scaled time units
  - `1`
  -

* - `-units_kilogram`
  - 1 kilogram in scaled mass units
  - `1`
  -

* - `-units_Kelvin`
  - 1 Kelvin in scaled temperature units
  - `1`
  -

* - `-stab`
  - Stabilization method (`none`, `su`, or `supg`)
  - `none`
  -

* - `-c_tau`
  - Stabilization constant, $c_\tau$
  - `0.5`
  -

* - `-Ctau_t`
  - Stabilization time constant, $C_t$
  - `1.0`
  -

* - `-Ctau_v`
  - Stabilization viscous constant, $C_v$
  - `36, 60, 128 for degree = 1, 2, 3`
  -

* - `-Ctau_C`
  - Stabilization continuity constant, $C_c$
  - `1.0`
  -

* - `-Ctau_M`
  - Stabilization momentum constant, $C_m$
  - `1.0`
  -

* - `-Ctau_E`
  - Stabilization energy constant, $C_E$
  - `1.0`
  -

* - `-cv`
  - Heat capacity at constant volume
  - `717`
  - `J/(kg K)`

* - `-cp`
  - Heat capacity at constant pressure
  - `1004`
  - `J/(kg K)`

* - `-g`
  - Gravitational acceleration
  - `9.81`
  - `m/s^2`

* - `-lambda`
  - Stokes hypothesis second viscosity coefficient
  - `-2/3`
  -

* - `-mu`
  - Shear dynamic viscosity coefficient
  - `75`
  -  `Pa s`

* - `-k`
  - Thermal conductivity
  - `0.02638`
  - `W/(m K)`

* - `-newtonian_unit_tests`
  - Developer option to test properties
  - `false`
  - boolean

* - `-state_var`
  - State variables to solve solution with. `conservative` ($\rho, \rho \bm{u}, \rho e$), `primitive` ($P, \bm{u}, T$), or `entropy` ($\frac{\gamma - s}{\gamma - 1} - \frac{\rho}{P} (e - c_v T),\ \frac{\rho}{P} \bm{u},\ -\frac{\rho}{P}$) where  $s = \ln(P\rho^{-\gamma})$
  - `conservative`
  - string

* - `-idl_decay_time`
  - Characteristic timescale of the pressure deviance decay. The timestep is good starting point
  - `-1` (disabled)
  - `s`

* - `-idl_start`
  - Start of IDL in the x direction
  - `0`
  - `m`

* - `-idl_length`
  - Length of IDL in the positive x direction
  - `0`
  - `m`

* - `-sgs_model_type`
  - Type of subgrid stress model to use. Currently only `data_driven` is available
  - `none`
  - string

* - `-sgs_model_dd_leakyrelu_alpha`
  - Slope parameter for Leaky ReLU activation function. `0` corresponds to normal ReLU
  - 0
  -

* - `-sgs_model_dd_parameter_dir`
  - Path to directory with data-driven model parameters (weights, biases, etc.)
  - `./dd_sgs_parameters`
  - string

* - `-diff_filter_monitor`
  - Enable differential filter TSMonitor
  - `false`
  - boolean

* - `-diff_filter_grid_based_width`
  - Use filter width based on the grid size
  - `false`
  - boolean

* - `-diff_filter_width_scaling`
  - Anisotropic scaling for filter width in wall-aligned coordinates (snz)
  - `1,1,1`
  - `m`

* - `-diff_filter_kernel_scaling`
  - Scaling to make differential kernel size equivalent to other filter kernels
  - `0.1`
  - `m^2`

* - `-diff_filter_wall_damping_function`
  - Damping function to use at the wall for anisotropic filtering (`none`, `van_driest`)
  - `none`
  - string

* - `-diff_filter_wall_damping_constant`
  - Constant for the wall-damping function. $A^+$ for `van_driest` damping function.
  - 25
  -

* - `-diff_filter_friction_length`
  - Friction length associated with the flow, $\delta_\nu$. Used in wall-damping functions
  - 0
  - `m`

:::

#### Gaussian Wave

The Gaussian wave problem has the following command-line options in addition to the Newtonian Ideal Gas options:

:::{list-table} Gaussian Wave Runtime Options
:header-rows: 1

* - Option
  - Description
  - Default value
  - Unit

* - `-freestream_riemann`
  - Riemann solver for boundaries (HLL or HLLC)
  - `hllc`
  -

* - `-freestream_velocity`
  - Freestream velocity vector
  - `0,0,0`
  - `m/s`

* - `-freestream_temperature`
  - Freestream temperature
  - `288`
  - `K`

* - `-freestream_pressure`
  - Freestream pressure
  - `1.01e5`
  - `Pa`

* - `-epicenter`
  - Coordinates of center of perturbation
  - `0,0,0`
  - `m`

* - `-amplitude`
  - Amplitude of the perturbation
  - `0.1`
  -

* - `-width`
  - Width parameter of the perturbation
  - `0.002`
  - `m`

:::

This problem can be run with the `gaussianwave.yaml` file via:

```
./navierstokes -options_file gaussianwave.yaml
```

```{literalinclude} ../../../../../examples/fluids/gaussianwave.yaml
:language: yaml
```

#### Vortex Shedding - Flow past Cylinder

The vortex shedding, flow past cylinder problem has the following command-line options in addition to the Newtonian Ideal Gas options:

:::{list-table} Vortex Shedding Runtime Options
:header-rows: 1

* - Option
  - Description
  - Default value
  - Unit

* - `-freestream_velocity`
  - Freestream velocity vector
  - `0,0,0`
  - `m/s`

* - `-freestream_temperature`
  - Freestream temperature
  - `288`
  - `K`

* - `-freestream_pressure`
  - Freestream pressure
  - `1.01e5`
  - `Pa`

:::

The initial condition is taken from `-reference_temperature` and `-reference_pressure`.
To run this problem, first generate a mesh:

```console
$ make -C examples/fluids/meshes
```

Then run by building the executable and running:

```console
$ make build/fluids-navierstokes
$ mpiexec -n 6 build/fluids-navierstokes -options_file examples/fluids/vortexshedding.yaml -{ts,snes}_monitor_
```

The vortex shedding period is roughly 5.6 and this problem runs until time 100 (2000 time steps).
The above run writes a file named `force.csv` (see `ts_monitor_wall_force` in `vortexshedding.yaml`), which can be postprocessed by running to create a figure showing lift and drag coefficients over time.

```console
$ python examples/fluids/postprocess/vortexshedding.py
```

```{literalinclude} ../../../../../examples/fluids/vortexshedding.yaml
:language: yaml
```

#### Density current

The Density Current problem has the following command-line options in addition to the Newtonian Ideal Gas options:

:::{list-table} Density Current Runtime Options
:header-rows: 1

* - Option
  - Description
  - Default value
  - Unit

* - `-center`
  - Location of bubble center
  - `(lx,ly,lz)/2`
  - `(m,m,m)`

* - `-dc_axis`
  - Axis of density current cylindrical anomaly, or `(0,0,0)` for spherically symmetric
  - `(0,0,0)`
  -

* - `-rc`
  - Characteristic radius of thermal bubble
  - `1000`
  - `m`

* - `-theta0`
  - Reference potential temperature
  - `300`
  - `K`

* - `-thetaC`
  - Perturbation of potential temperature
  - `-15`
  - `K`

* - `-P0`
  - Atmospheric pressure
  - `1E5`
  - `Pa`

* - `-N`
  - Brunt-Vaisala frequency
  - `0.01`
  - `1/s`
:::

This problem can be run with:

```
./navierstokes -problem density_current -dm_plex_box_faces 16,1,8 -degree 1 -dm_plex_box_lower 0,0,0 -dm_plex_box_upper 2000,125,1000 -dm_plex_dim 3 -rc 400. -bc_wall 1,2,5,6 -wall_comps 1,2,3 -bc_slip_y 3,4 -mu 75
```

#### Channel flow

The Channel problem has the following command-line options in addition to the Newtonian Ideal Gas options:

:::{list-table} Channel Runtime Options
:header-rows: 1

* - Option
  - Description
  - Default value
  - Unit

* - `-umax`
  - Maximum/centerline velocity of the flow
  - `10`
  - `m/s`

* - `-theta0`
  - Reference potential temperature
  - `300`
  - `K`

* - `-P0`
  - Atmospheric pressure
  - `1E5`
  - `Pa`

* - `-body_force_scale`
  - Multiplier for body force (`-1` for flow reversal)
  - 1
  -
:::

This problem can be run with the `channel.yaml` file via:

```
./navierstokes -options_file channel.yaml
```
```{literalinclude} ../../../../../examples/fluids/channel.yaml
:language: yaml
```

(example-blasius)=

#### Blasius boundary layer

The Blasius problem has the following command-line options in addition to the Newtonian Ideal Gas options:

:::{list-table} Blasius Runtime Options
:header-rows: 1

* - Option
  - Description
  - Default value
  - Unit

* - `-velocity_infinity`
  - Freestream velocity
  - `40`
  - `m/s`

* - `-temperature_infinity`
  - Freestream temperature
  - `288`
  - `K`

* - `-temperature_wall`
  - Wall temperature
  - `288`
  - `K`

* - `-delta0`
  - Boundary layer height at the inflow
  - `4.2e-3`
  - `m`

* - `-P0`
  - Atmospheric pressure
  - `1.01E5`
  - `Pa`

* - `-platemesh_refine_height`
  - Height at which `-platemesh_Ndelta` number of elements should refined into
  - `5.9E-4`
  - `m`

* - `-platemesh_Ndelta`
  - Number of elements to keep below `-platemesh_refine_height`
  - `45`
  -

* - `-platemesh_growth`
  - Growth rate of the elements in the refinement region
  - `1.08`
  -

* - `-platemesh_top_angle`
  - Downward angle of the top face of the domain. This face serves as an outlet.
  - `5`
  - `degrees`

* - `-stg_use`
  - Whether to use stg for the inflow conditions
  - `false`
  -

* - `-platemesh_y_node_locs_path`
  - Path to file with y node locations. If empty, will use mesh warping instead.
  - `""`
  -

* - `-n_chebyshev`
  - Number of Chebyshev terms
  - `20`
  -

* - `-chebyshev_`
  - Prefix for Chebyshev snes solve
  -
  -

:::

This problem can be run with the `blasius.yaml` file via:

```
./navierstokes -options_file blasius.yaml
```

```{literalinclude} ../../../../../examples/fluids/blasius.yaml
:language: yaml
```

#### STG Inflow for Flat Plate

Using the STG Inflow for the blasius problem adds the following command-line options:

:::{list-table} Blasius Runtime Options
:header-rows: 1

* - Option
  - Description
  - Default value
  - Unit

* - `-stg_inflow_path`
  - Path to the STGInflow file
  - `./STGInflow.dat`
  -

* - `-stg_rand_path`
  - Path to the STGRand file
  - `./STGRand.dat`
  -

* - `-stg_alpha`
  - Growth rate of the wavemodes
  - `1.01`
  -

* - `-stg_u0`
  - Convective velocity, $U_0$
  - `0.0`
  - `m/s`

* - `-stg_mean_only`
  - Only impose the mean velocity (no fluctutations)
  - `false`
  -

* - `-stg_strong`
  - Strongly enforce the STG inflow boundary condition
  - `false`
  -

* - `-stg_fluctuating_IC`
  - "Extrude" the fluctuations through the domain as an initial condition
  - `false`
  -

:::

This problem can be run with the `blasius.yaml` file via:

```
./navierstokes -options_file blasius.yaml -stg_use true
```

Note the added `-stg_use true` flag
This overrides the `stg: use: false` setting in the `blasius.yaml` file, enabling the use of the STG inflow.
