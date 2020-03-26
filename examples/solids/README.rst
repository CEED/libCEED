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

There are five required command line options and a variety of additional command line options for this mini-app.
The four required command line options are :code:`-mesh`, :code:`-degree`, :code:`-E`, and :code:`-nu`. Additionally, at least one boundary condition must be set, using :code:`-bc_zero` or :code:`-bc_clamp`.

To set the ExodusII mesh file, use the :code:`-mesh` option and the file path to the mesh file.

To set the polynomial degree of the finite element basis, use the :code:`-degree` option.

To set the materiel parameters, use the :code:`-E` and :code:`-nu` options, for Young's modulus and Poisson's ratio, respectively.

To set the boundary conditions, use :code:`-bc_zero` or :code:`-bc_clamp` followed by a comma separated list of the constrained faces.

A face set with :code:`-bc_zero` will remain fixed at zero displacement and a face set with :code:`-bc_clamp` will be displaced by :code:`-bc_clamp_max` in the y direction, or -1 in the y direction if this value is not set.

The following is an example of a minimal set of command line options::

   ./elasticity -mesh ./meshes/meshes/cylinder8_672e_4ss_us.exo -degree 4 -E 1e6 -nu 0.3 -bc_zero 999 -bc_clamp 998

These command line options are the minimum requirements for the mini-app, but additional options may also be set.
For example, the problem formulation (:code:`-problem`), forcing term (:code:`-forcing`), libCEED backend resource (:code:`-ceed`) can be specified::

   ./elasticity -mesh ./meshes/meshes/cylinder8_672e_4ss_us.exo -degree 4 -E 1e6 -nu 0.3 -bc_zero 999 -bc_clamp 998 -problem hyperFS -forcing none -ceed /cpu/self/opt/blocked

Available runtime options are:

+-----------------------+-------------------------------------------------------------------+
| Option                | Meaning                                                           |
+=======================+===================================================================+
| ``-ceed``             | CEED resource specifier                                           |
+-----------------------+-------------------------------------------------------------------+
| ``-ceed_fine``        | CEED resource specifier for fine grid (P >= 5)                    |
+-----------------------+-------------------------------------------------------------------+
| ``-test``             | Run in test mode                                                  |
+-----------------------+-------------------------------------------------------------------+
| ``-degree``           | Polynomial degree of tensor product basis                         |
+-----------------------+-------------------------------------------------------------------+
| ``-mesh``             | Filepath to Exodus-II file                                        |
+-----------------------+-------------------------------------------------------------------+
| ``-problem``          | Problem to solve (``linElas``, ``hyperSS``, or ``hyperFS``)       |
+-----------------------+-------------------------------------------------------------------+
| ``-forcing``          | Forcing term option (``none``, ``constant``, or ``mms``)          |
+-----------------------+-------------------------------------------------------------------+
| ``-bc_zero``          | List of boundary face IDs to apply zero Dirichlet BCs             |
+-----------------------+-------------------------------------------------------------------+
| ``-bc_clamp``         | List of boundary face IDs to apply incremental -y Dirichlet BCs   |
+-----------------------+-------------------------------------------------------------------+
| ``-bc_clamp_max``     | Maximum value to displace clamped boundary                        |
+-----------------------+-------------------------------------------------------------------+
| ``-num_steps``        | Number of pseudo-time steps for continuation method               |
+-----------------------+-------------------------------------------------------------------+
| ``-view_soln``        | Output solution at each pseudo-time step for viewing              |
+-----------------------+-------------------------------------------------------------------+
| ``-E``                | Young's modulus                                                   |
+-----------------------+-------------------------------------------------------------------+
| ``-nu``               | Poisson's ratio                                                   |
+-----------------------+-------------------------------------------------------------------+
| ``-units_meter``      | 1 meter in scaled length units                                    |
+-----------------------+-------------------------------------------------------------------+
| ``-units_second``     | 1 second in scaled time units                                     |
+-----------------------+-------------------------------------------------------------------+
| ``-units_kilogram``   | 1 kilogram in scaled mass units                                   |
+-----------------------+-------------------------------------------------------------------+

