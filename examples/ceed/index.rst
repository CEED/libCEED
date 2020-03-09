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
   I = \int_{\Omega} \mathbf{1} \, d V\, .
   :label: eq-ex1-volume

Using the same notation as in :ref:`Theoretical Framework`, we write here the vector
:math:`u(\mathbf{x})\equiv \mathbf{1}` in the Galerkin approximation,
and find the volume of :math:`\Omega` as

.. math::
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
   I = \int_{\partial \Omega} \mathbf{1} \, d S\, .
   :label: eq-ex2-surface

but this time by solving a Laplace's equation for a harmonic function
:math:`u(\mathbf{x})`. We write the Laplace's equation

.. math::
   \nabla \cdot \nabla u = 0, \textrm{ for  } \mathbf{x} \in \Omega.
   :label: eq-laplace

We can rewrite this via the bilinear form

.. math::
   B(u,v) = L(v)

where :math:`v` is the test function, and for which :math:`L(v)=0` in this case. We
obtain

.. math::
   B(u,v)  = \int_\Omega v \nabla \cdot \nabla u \, d V =   \int_{\partial \Omega} v \nabla u \cdot \mathbf{n}\, d S - \int_\Omega \nabla v \cdot \nabla u \, d V  = 0\, ,

where we have used integration by parts.

:math:`B(u,v) = 0` because we have chosen :math:`u(\mathbf{x})` to be harmonic, so we
can write

.. math::
   \int_{\partial \Omega} v \nabla u \cdot \mathbf{n}\, d S = \int_\Omega \nabla v \cdot \nabla u \, d V
   :label: eq-laplace-by-parts

and use the :ref:`CeedOperator` for Laplace's operator to compute the right-hand side of
equation :math:numref:`eq-laplace-by-parts`. This way, the left-hand side of equation
:math:numref:`eq-laplace-by-parts` (which gives :math:numref:`eq-ex2-surface` because
we have chosen :math:`u(\mathbf{x}) = (x + y + z)` such that  :math:`\nabla u \cdot \mathbf{n} = 1`)
is readily found.
