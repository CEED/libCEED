Standalone libCEED
======================================

The following two examples have no dependencies, and are designed to be self-contained.
For additional examples that use external discretization libraries (MFEM, PETSc, Nek5000
etc.) see the subdirectories in :file:`examples/`.


.. _ex1-volume:

Ex1-Volume
--------------------------------------

This example is located in the subdirectory :file:`examples/ceed`. It illustrates a
simple usage of libCEED to compute the volume of a given body using a matrix-free
application of the mass operator. Arbitrary mesh and solution orders in 1D, 2D and 3D
are supported from the same code.

This example shows how to compute line/surface/volume integrals of a 1D, 2D, or 3D
domain :math:`\Omega` respectively, by applying the mass operator to a vector of
:math:`\mathbf{1}`\s. It computes:

.. math::
   I = \int_{\Omega} \mathbf{1} \, dV .
   :label: eq-ex1-volume

Using the same notation as in :ref:`Theoretical Framework`, we write here the vector
:math:`u(\mathbf{x})\equiv \mathbf{1}` in the Galerkin approximation,
and find the volume of :math:`\Omega` as

.. math::
   :label: volume-sum

   \sum_e \int_{\Omega_e} v(x) \cdot \mathbf{1} \, dV

with :math:`v(x) \in \mathcal{V}_p = \{ v \in H^{1}(\Omega_e) \,|\, v \in P_p(\boldsymbol{I}), e=1,\ldots,N_e \}`,
the test functions.


.. _ex2-surface:

Ex2-Surface
--------------------------------------

This example is located in the subdirectory :file:`examples/ceed`. It computes the
surface area of a given body using matrix-free application of a diffusion operator.
Arbitrary mesh and solution orders in 1D, 2D and 3D are supported from the same code.

Similarly to :ref:`Ex1-Volume`, it computes:

.. math::
   I = \int_{\partial \Omega} \mathbf{1} \, dS .
   :label: eq-ex2-surface

but this time by applying the divergence theorem using a Laplacian.
In particular, we select :math:`u(\bm x) = x_0 + x_1 + x_2`, for which :math:`\nabla u = [1, 1, 1]^T`, and thus :math:`\nabla u \cdot \hat{\bm n} = 1`.

Given Laplace's equation,

.. math::
   -\nabla \cdot \nabla u = 0, \textrm{ for  } \mathbf{x} \in \Omega

multiply by a test function :math:`v` and integrate by parts to obtain

.. math::
    \int_\Omega \nabla v \cdot \nabla u \, dV - \int_{\partial \Omega} v \nabla u \cdot \hat{\bm n}\, dS = 0 .

Since we have chosen :math:`u` such that :math:`\nabla u \cdot \hat{\bm n} = 1`, the boundary integrand is :math:`v 1 \equiv v`. Hence, similar to :math:numref:`volume-sum`, we can evaluate the surface integral by applying the volumetric Laplacian as follows

.. math::
   \int_\Omega \nabla v \cdot \nabla u \, dV \approx \sum_e \int_{\partial \Omega_e} v(x) \cdot \mathbf{1} \, dS .
