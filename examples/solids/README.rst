libCEED: Solid Mechanics Example
--------------------------------

This page provides a description of the solid mechanics example for the
libCEED library, based on PETSc.

This code solves the steady-state static momentum balance equations using unstructured high-order finite/spectral element spatial discretizations.
In this mini-app, we consider three formulations used in solid mechanics applications: linear elasticity, Neo-Hookean hyperelasticity at small strain, and Neo-Hookean hyperelasticity at finite strain.

Build by using::

   make

and run with::

   ./elasticity -mesh [.exo file] -degree [degree] -nu [nu] -E [E] [boundary options] -problem [problem type] -forcing [forcing] -ceed [ceed]

Runtime options
---------------

.. inclusion-marker-do-not-remove

The elasticity min-app is controlled via command-line options, the following of which are mandatory.

.. list-table:: Mandatory Runtime Options
   :header-rows: 1

   * - Option
     - Description

   * - :code:`-mesh [filename]`
     - Path to mesh file in any format supported by PETSc.

   * - :code:`-degree [int]`
     - Polynomial degree of the finite element basis

   * - :code:`-E [real]`
     - `Young's modulus <https://en.wikipedia.org/wiki/Young%27s_modulus>`_, :math:`E > 0`

   * - :code:`-nu [real]`
     - `Poisson's ratio <https://en.wikipedia.org/wiki/Poisson%27s_ratio>`_, :math:`\nu < 0.5`

   * - :code:`-bc_zero [int list]`
     - List of faces sets on which to enforce zero displacement

   * - :code:`-bc_clamp [int list]`
     - List of face sets on which to displace by :code:`-bc_clamp_max` in the :math:`y` direction

(One can set only one of :code:`-bc_zero` or :code:`-bc_clamp`, but the result will likely not be interesting.)
The following is an example of a minimal set of command line options::

   ./elasticity -mesh [.exo file] -degree 4 -E 1e6 -nu 0.3 -bc_zero 999 -bc_clamp 998

.. note::

   This solver can use any mesh format that PETSc's ``DMPlex`` can read (Exodus, Gmsh, Med, etc.).
   Our tests have primarily been using Exodus meshes created using CUBIT_; sample meshes used for the example runs suggested here can be found in `this repository`_.

.. _CUBIT: https://cubit.sandia.gov/
.. _this repository: https://github.com/jeremylt/ceedSampleMeshes

These command line options are the minimum requirements for the mini-app, but additional options may also be set.

.. list-table:: Additional Runtime Options
   :header-rows: 1

   * - Option
     - Description
     - Default value

   * - :code:`-ceed`
     - CEED resource specifier
     - :code:`/cpu/self`

   * - :code:`-ceed_fine`
     - CEED resource specifier for multigrid fine grid
     - :code:`/cpu/self`

   * - :code:`-test`
     - Run in test mode
     -

   * - :code:`-problem`
     - Problem to solve (:code:`linElas`, :code:`hyperSS` or :code:`hyperFS`)
     - :code:`linElas`

   * - :code:`-forcing`
     -  Forcing term option (:code:`none`, :code:`constant`, or :code:`mms`)
     - :code:`none`

   * - :code:`-bc_clamp_max`
     - Maximum value to displace clamped boundary
     - :code:`-1.0`

   * - :code:`-num_steps`
     - Number of load increments for continuation method
     - :code:`1` if :code:`linElas` else :code:`10`

   * - :code:`-view_soln`
     - Output solution at each load increment for viewing
     -

   * - :code:`-snes_view`
     - View PETSc :code:`SNES` nonlinear solver configuration
     -

   * - :code:`-log_view`
     - View PETSc performance log
     -

   * - :code:`-help`
     - View comprehensive information about run-time options
     -

To verify the convergence of the linear elasticity formulation on a given mesh with the method of manufactured solutions, run::

   ./elasticity -mesh [mesh] -degree [degree] -nu [nu] -E [E] -forcing mms

This option attempts to recover a known solution from an analytically computed forcing term.

On algebraic solvers
^^^^^^^^^^^^^^^^^^^^
This mini-app is configured to use the following Newton-Krylov-Multigrid method by default.

* Newton-type methods for the nonlinear solve, with the hyperelasticity models globalized using load increments.
* Preconditioned conjugate gradients to solve the symmetric positive definite linear systems arising at each Newton step.
* Preconditioning via :math:`p`-version multigrid coarsening to linear elements, with algebraic multigrid (PETSc's ``GAMG``) for the coarse solve.
  The default smoother uses degree 3 Chebyshev with Jacobi preconditioning.
  (Lower degree is often faster, albeit less robust; try :code:`-outer_mg_levels_ksp_max_it 2`, for example.)
  Application of the linear operators for all levels with degree :math:`p > 1` is performed matrix-free using analytic Newton linearization, while the lowest order :math:`p = 1` operators are assembled explicitly (using coloring at present).

Many related solvers can be implemented by composing PETSc command-line options.

Nondimensionalization
^^^^^^^^^^^^^^^^^^^^^

Quantities such as the Young's modulus vary over many orders of magnitude, and thus can lead to poorly scaled equations.
One can nondimensionalize the model by choosing an alternate system of units, such that displacements and residuals are of reasonable scales.

.. list-table:: (Non)dimensionalization options
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

For example, consider a problem involving metals subject to gravity.

.. list-table:: Characteristic units for metals
   :header-rows: 1

   * - Quantity
     - Typical value in SI units

   * - Displacement, :math:`\bm u`
     - :math:`1 \,\mathrm{cm} = 10^{-2} \,\mathrm m`

   * - Young's modulus, :math:`E`
     - :math:`100 \,\mathrm{GPa} = 10^{11} \,\mathrm{kg}\, \mathrm{m}^{-1}\, \mathrm s^{-2}`

   * - Body force (gravity) on volume, :math:`\int \rho \bm g`
     - :math:`5 \cdot 10^4 \,\mathrm{kg}\, \mathrm m^{-2} \, \mathrm s^{-2} \cdot (\text{volume} \, \mathrm m^3)`

One can choose units of displacement independently (e.g., :code:`-units_meter 100` to measure displacement in centimeters), but :math:`E` and :math:`\int \rho \bm g` have the same dependence on mass and time, so cannot both be made of order 1.
This reflects the fact that both quantities are not equally significant for a given displacement size; the relative significance of gravity increases as the domain size grows.
