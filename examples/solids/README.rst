libCEED: Solid Mechanics Example
--------------------------------

This page provides a description of the solid mechanics example for the
libCEED library, based on PETSc.

This code solves the steady-state static momentum balance equations using unstructured high-order finite/spectral element spatial discretizations.
In this mini-app, we consider three formulations used in solid mechanics applications: linear elasticity, Neo-Hookean hyperelasticity at small strain, and Neo-Hookean hyperelasticity at finite strain.

Build by using

``make``

and run with

``./elasticity -mesh [.exo file] -degree [degree] -nu [nu] -E [E] [boundary options] -problem [problem type] -forcing [forcing] -ceed [ceed]``

Runtime options
---------------

.. inclusion-marker-do-not-remove

There are five mandatory command line options and a variety of additional command line options for this mini-app.

.. list-table:: Mandatory Runtime Options
   :header-rows: 1

   * - Option
     - Description

   * - :code:`-mesh`
     - File path to ExodusII mesh file

   * - :code:`-degree`
     - Polynomial degree of the finite element basis

   * - :code:`-E`
     - Young's modulus and Poisson's ratio

   * - :code:`-nu`
     - Poisson's ratio

   * - :code:`-bc_zero`
     - A face set with :code:`-bc_zero` will remain fixed at zero displacement

   * - :code:`-bc_clamp`
     - A face set with :code:`-bc_clamp` will be displaced by :code:`-bc_clamp_max` in the y direction

The following is an example of a minimal set of command line options::

   ./elasticity -mesh ./meshes/cylinder8_672e_4ss_us.exo -degree 4 -E 1e6 -nu 0.3 -bc_zero 999 -bc_clamp 998

.. note::

   Sample meshes can be found here_.

.. _here: https://github.com/jeremylt/ceedSampleMeshes

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
     - -1.0

   * - :code:`-num_steps`
     - Number of pseudo-time steps for continuation method
     - :code:`1` for :code:`linElas`, otherwise :code:`10`

   * - :code:`-view_soln`
     - Output solution at each pseudo-time step for viewing
     -

   * - :code:`-units_meter`
     - 1 meter in scaled length units
     - :code:`1`

   * - :code:`-units_second`
     - 1 second in scaled time units
     - :code:`1`

   * - :code:`-units_kilogram`
     - 1 kilogram in scaled mass units
     - :code:`1`

To verify the convergence of the linear elasticity formulation on a given mesh with the method of manufactured solutions, run::

   ./elasticity -mesh [mesh] -degree [degree] -nu [nu] -E [E] -forcing mms

This option attempts to recover a known solution from an analytically computed forcing term.

