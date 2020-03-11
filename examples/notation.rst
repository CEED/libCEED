Common notation
========================================

For most of our examples, the spatial discretization
uses high-order finite elements/spectral elements, namely, the high-order Lagrange
polynomials defined over :math:`P` non-uniformly spaced nodes, the
Gauss-Legendre-Lobatto (GLL) points, and quadrature points :math:`\{q_i\}_{i=1}^Q`, with
corresponding weights :math:`\{w_i\}_{i=1}^Q` (typically the ones given by Gauss
or Gauss-Lobatto quadratures, that are built in the library).

We discretize the domain, :math:`\Omega \subset \mathbb{R}^d` (with :math:`d=1,2,3`,
typically) by letting :math:`\Omega = \bigcup_{e=1}^{N_e}\Omega_e`, with :math:`N_e`
disjoint elements. For most examples we use unstructured meshes for which the elements
are hexahedra (although this is not a requirement in libCEED).

The physical coordinates are denoted by :math:`\mathbf{x}=(x,y,z)\in\Omega_e`,
while the reference coordinates are represented as
:math:`\boldsymbol{X}=(X,Y,Z) \equiv (X_1,X_2,X_3) \in\mathbf{I}=[-1,1]^3`
(for :math:`d=3`).
