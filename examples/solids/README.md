# libCEED: Solid Mechanics Example

This page provides a description of the solid mechanics example for the
libCEED library, based on PETSc.
PETSc v3.17 or a development version of PETSc at commit 0e95d842 or later is required.

This code solves the steady-state static momentum balance equations using unstructured high-order finite/spectral element spatial discretizations.
In this mini-app, we consider three formulations used in solid mechanics applications: linear elasticity, Neo-Hookean hyperelasticity at small strain, and Neo-Hookean hyperelasticity at finite strain.
All three of these formulations are for compressible materials.

Build by using:

```
make
```

and run with:

```
./elasticity -mesh [.exo file] -degree [degree] -nu [nu] -E [E] [boundary options] -problem [problem type] -forcing [forcing] -ceed [ceed]
```

## Runtime options

% inclusion-solids-marker

The elasticity mini-app is controlled via command-line options, the following of which are mandatory.

:::{list-table} Mandatory Runtime Options
:header-rows: 1
:widths: 3 7

* - Option
  - Description
* - `-mesh [filename]`
  - Path to mesh file in any format supported by PETSc.
* - `-degree [int]`
  - Polynomial degree of the finite element basis
* - `-E [real]`
  - [Young's modulus](https://en.wikipedia.org/wiki/Young%27s_modulus), $E > 0$
* - `-nu [real]`
  - [Poisson's ratio](https://en.wikipedia.org/wiki/Poisson%27s_ratio), $\nu < 0.5$
* - `-bc_clamp [int list]`
  - List of face sets on which to displace by `-bc_clamp_[facenumber]_translate [x,y,z]`
    and/or `bc_clamp_[facenumber]_rotate [rx,ry,rz,c_0,c_1]`. Note: The default
    for a clamped face is zero displacement. All displacement is with respect to
    the initial configuration.
* - `-bc_traction [int list]`
  - List of face sets on which to set traction boundary conditions with the
    traction vector `-bc_traction_[facenumber] [tx,ty,tz]`
:::

:::{note}
This solver can use any mesh format that PETSc's `DMPlex` can read (Exodus, Gmsh, Med, etc.).
Our tests have primarily been using Exodus meshes created using [CUBIT]; sample meshes used for the example runs suggested here can be found in [this repository].
Note that many mesh formats require PETSc to be configured appropriately; e.g., `--download-exodusii` for Exodus support.
:::

Consider the specific example of the mesh seen below:

```{image} https://github.com/jeremylt/ceedSampleMeshes/raw/master/cylinderDiagram.png
```

With the sidesets defined in the figure, we provide here an example of a minimal set of command line options:

```
./elasticity -mesh [.exo file] -degree 4 -E 1e6 -nu 0.3 -bc_clamp 998,999 -bc_clamp_998_translate 0,-0.5,1
```

In this example, we set the left boundary, face set $999$, to zero displacement and the right boundary, face set $998$, to displace $0$ in the $x$ direction, $-0.5$ in the $y$, and $1$ in the $z$.

As an alternative to specifying a mesh with {code}`-mesh`, the user may use a DMPlex box mesh by specifying {code}`-dm_plex_box_faces [int list]`, {code}`-dm_plex_box_upper [real list]`, and {code}`-dm_plex_box_lower [real list]`.

As an alternative example exploiting {code}`-dm_plex_box_faces`, we consider a {code}`4 x 4 x 4` mesh where essential (Drichlet) boundary condition is placed on all sides. Sides 1 through 6 are rotated around $x$-axis:

```
./elasticity -problem FSInitial-NH1 -E 1 -nu 0.3 -num_steps 40 -snes_linesearch_type cp -dm_plex_box_faces 4,4,4 -bc_clamp 1,2,3,4,5,6 -bc_clamp_1_rotate 0,0,1,0,.3 -bc_clamp_2_rotate 0,0,1,0,.3 -bc_clamp_3_rotate 0,0,1,0,.3 -bc_clamp_4_rotate 0,0,1,0,.3 -bc_clamp_5_rotate 0,0,1,0,.3 -bc_clamp_6_rotate 0,0,1,0,.3
```

:::{note}
If the coordinates for a particular side of a mesh are zero along the axis of rotation, it may appear that particular side is clamped zero.
:::

On each boundary node, the rotation magnitude is computed: {code}`theta = (c_0 + c_1 * cx) * loadIncrement` where {code}`cx = kx * x + ky * y + kz * z`, with {code}`kx`, {code}`ky`, {code}`kz` are normalized values.

The command line options just shown are the minimum requirements to run the mini-app, but additional options may also be set as follows

:::{list-table} Additional Runtime Options
:header-rows: 1

* - Option
  - Description
  - Default value

* - `-ceed`
  - CEED resource specifier
  - `/cpu/self`

* - `-qextra`
  - Number of extra quadrature points
  - `0`

* - `-test`
  - Run in test mode
  -

* - `-problem`
  - Problem to solve (`Linear`, `SS-NH`, `FSInitial-NH1`, etc.)
  - `Linear`

* - `-forcing`
  -  Forcing term option (`none`, `constant`, or `mms`)
  - `none`

* - `-forcing_vec`
  -  Forcing vector
  - `0,-1,0`

* - `-multigrid`
  - Multigrid coarsening to use (`logarithmic`, `uniform` or `none`)
  - `logarithmic`

* - `-nu_smoother [real]`
  - Poisson's ratio for multigrid smoothers, $\nu < 0.5$
  -

* - `-num_steps`
  - Number of load increments for continuation method
  - `1` if `Linear` else `10`

* - `-view_soln`
  - Output solution at each load increment for viewing
  -

* - `-view_final_soln`
  - Output solution at final load increment for viewing
  -

* - `-snes_view`
  - View PETSc `SNES` nonlinear solver configuration
  -

* - `-log_view`
  - View PETSc performance log
  -

* - `-output_dir`
  - Output directory
  - `.`

* - `-help`
  - View comprehensive information about run-time options
  -
:::

To verify the convergence of the linear elasticity formulation on a given mesh with the method of manufactured solutions, run:

```
./elasticity -mesh [mesh] -degree [degree] -nu [nu] -E [E] -forcing mms
```

This option attempts to recover a known solution from an analytically computed forcing term.

### On algebraic solvers

This mini-app is configured to use the following Newton-Krylov-Multigrid method by default.

- Newton-type methods for the nonlinear solve, with the hyperelasticity models globalized using load increments.
- Preconditioned conjugate gradients to solve the symmetric positive definite linear systems arising at each Newton step.
- Preconditioning via $p$-version multigrid coarsening to linear elements, with algebraic multigrid (PETSc's `GAMG`) for the coarse solve.
  The default smoother uses degree 3 Chebyshev with Jacobi preconditioning.
  (Lower degree is often faster, albeit less robust; try {code}`-outer_mg_levels_ksp_max_it 2`, for example.)
  Application of the linear operators for all levels with degree $p > 1$ is performed matrix-free using analytic Newton linearization, while the lowest order $p = 1$ operators are assembled explicitly (using coloring at present).

Many related solvers can be implemented by composing PETSc command-line options.

### Nondimensionalization

Quantities such as the Young's modulus vary over many orders of magnitude, and thus can lead to poorly scaled equations.
One can nondimensionalize the model by choosing an alternate system of units, such that displacements and residuals are of reasonable scales.

:::{list-table} (Non)dimensionalization options
:header-rows: 1

* - Option
  - Description
  - Default value

* - :code:`-units_meter`
  - 1 meter in scaled length units
  - :code:`1`

* - :code:`-units_second`
  - 1 second in scaled time units
  - :code:`1`

* - :code:`-units_kilogram`
  - 1 kilogram in scaled mass units
  - :code:`1`
:::

For example, consider a problem involving metals subject to gravity.

:::{list-table} Characteristic units for metals
:header-rows: 1

* - Quantity
  - Typical value in SI units

* - Displacement, $\bm u$
  - $1 \,\mathrm{cm} = 10^{-2} \,\mathrm m$

* - Young's modulus, $E$
  - $10^{11} \,\mathrm{Pa} = 10^{11} \,\mathrm{kg}\, \mathrm{m}^{-1}\, \mathrm s^{-2}$

* - Body force (gravity) on volume, $\int \rho \bm g$
  - $5 \cdot 10^4 \,\mathrm{kg}\, \mathrm m^{-2} \, \mathrm s^{-2} \cdot (\text{volume} \, \mathrm m^3)$
:::

One can choose units of displacement independently (e.g., {code}`-units_meter 100` to measure displacement in centimeters), but $E$ and $\int \rho \bm g$ have the same dependence on mass and time, so cannot both be made of order 1.
This reflects the fact that both quantities are not equally significant for a given displacement size; the relative significance of gravity increases as the domain size grows.

### Diagnostic Quantities

Diagnostic quantities for viewing are provided when the command line options for visualization output, {code}`-view_soln` or {code}`-view_final_soln` are used.
The diagnostic quantities include displacement in the $x$ direction, displacement in the $y$ direction, displacement in the $z$ direction, pressure, $\operatorname{trace} \bm{E}$, $\operatorname{trace} \bm{E}^2$, $\lvert J \rvert$, and strain energy density.
The table below summarizes the formulations of each of these quantities for each problem type.

:::{list-table} Diagnostic quantities
   :header-rows: 1

   * - Quantity
     - Linear Elasticity
     - Hyperelasticity, Small Strain
     - Hyperelasticity, Finite Strain

   * - Pressure
     - $\lambda \operatorname{trace} \bm{\epsilon}$
     - $\lambda \log \operatorname{trace} \bm{\epsilon}$
     - $\lambda \log J$

   * - Volumetric Strain
     - $\operatorname{trace} \bm{\epsilon}$
     - $\operatorname{trace} \bm{\epsilon}$
     - $\operatorname{trace} \bm{E}$

   * - $\operatorname{trace} \bm{E}^2$
     - $\operatorname{trace} \bm{\epsilon}^2$
     - $\operatorname{trace} \bm{\epsilon}^2$
     - $\operatorname{trace} \bm{E}^2$

   * - $\lvert J \rvert$
     - $1 + \operatorname{trace} \bm{\epsilon}$
     - $1 + \operatorname{trace} \bm{\epsilon}$
     - $\lvert J \rvert$

   * - Strain Energy Density
     - $\frac{\lambda}{2} (\operatorname{trace} \bm{\epsilon})^2 + \mu \bm{\epsilon} : \bm{\epsilon}$
     - $\lambda (1 + \operatorname{trace} \bm{\epsilon}) (\log(1 + \operatorname{trace} \bm{\epsilon} ) - 1) + \mu \bm{\epsilon} : \bm{\epsilon}$
     - $\frac{\lambda}{2}(\log J)^2 + \mu \operatorname{trace} \bm{E} - \mu \log J$
:::

[cubit]: https://cubit.sandia.gov/
[this repository]: https://github.com/jeremylt/ceedSampleMeshes
