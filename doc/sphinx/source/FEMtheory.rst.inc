.. _Theoretical Framework:

Theoretical Framework
---------------------

In finite element formulations, the weak form of a Partial Differential Equation
(PDE) is evaluated on a subdomain :math:`\Omega_e` (element) and the local results
are composed into a larger system of equations that models the entire problem on
the global domain :math:`\Omega`. In particular, when high-order finite elements or
spectral elements are used, the resulting sparse matrix representation of the global
operator is computationally expensive, with respect to both the memory transfer and
floating point operations needed for its evaluation. libCEED provides an interface
for matrix-free operator description that enables efficient evaluation on a variety
of computational device types (selectable at run time). We present here the notation
and the mathematical formulation adopted in libCEED.

We start by considering the discrete residual :math:`F(u)=0` formulation
in weak form. We first define the :math:`L^2` inner product

.. math::
   \langle u, v \rangle = \int_\Omega u v d \bm{x},

where :math:`d \bm{x} \in \mathbb{R}^d \supset \Omega`.

We want to find :math:`u` in a suitable space :math:`V_D`,
such that

.. math::
   \langle v, f(u) \rangle = \int_\Omega v \cdot f_0 (u, \nabla u) + \nabla v : f_1 (u, \nabla u) = 0
   :label: residual

for all :math:`v` in the corresponding homogeneous space :math:`V_0`, where :math:`f_0`
and :math:`f_1` contain all possible sources in the problem. We notice here that
:math:`f_0` represents all terms in :math:numref:`residual` which multiply the test
function :math:`v` and :math:`f_1` all terms which multiply its gradient :math:`\nabla v`.
For an n-component problems in :math:`d` dimensions, :math:`f_0 \in \mathbb{R}^n` and
:math:`f_1 \in \mathbb{R}^{nd}`.

.. note::
   The notation :math:`\nabla \bm v \!:\! f_1` represents contraction over both
   fields and spatial dimensions while a single dot represents contraction in just one,
   which should be clear from context, e.g., :math:`v \cdot f_0` contracts only over
   fields.

.. note::
   In the code, the function that represents the weak form at quadrature
   points is called the :ref:`CeedQFunction`. In the :ref:`Examples` provided with the
   library (in the :file:`examples/` directory), we store the term :math:`f_0` directly
   into `v`, and the term :math:`f_1` directly into `dv` (which stands for
   :math:`\nabla v`). If equation :math:numref:`residual` only presents a term of the
   type :math:`f_0`, the :ref:`CeedQFunction` will only have one output argument,
   namely `v`. If equation :math:numref:`residual` also presents a term of the type
   :math:`f_1`, then the :ref:`CeedQFunction` will have two output arguments, namely,
   `v` and `dv`.


