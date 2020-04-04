.. _bps:

CEED Bakeoff Problems
========================================

.. include:: ./README.rst
   :start-after: bps-inclusion-marker
   :end-before: bps-exclusion-marker


.. _Mass Operator:

Mass Operator
----------------------------------------

The Mass Operator used in BP1 and BP2 is defined via the :math:`L^2` projection
problem, posed as a weak form on a Hilbert space :math:`V^p \subset H^1`, i.e.,
find :math:`u \in V^p` such that for all :math:`v \in V^p`

.. math::
   :label: eq-general-weak-form

   \langle v,u \rangle = \langle v,f \rangle ,

where :math:`\langle v,u\rangle` and :math:`\langle v,f\rangle` express the continuous
bilinear and linear forms, respectively, defined on :math:`V^p`, and, for sufficiently
regular :math:`u`, :math:`v`, and :math:`f`, we have:

.. math::
   \begin{aligned}
   \langle v,u \rangle &:= \int_{\Omega} \, v \, u \, dV ,\\
   \langle v,f \rangle &:= \int_{\Omega} \, v \, f \, dV .
   \end{aligned}

Following the standard finite/spectral element approach, we formally
expand all functions in terms of basis functions, such as

.. math::
   :label: eq-nodal-values

   \begin{aligned}
   u(\bm x) &= \sum_{j=1}^n u_j \, \phi_j(\bm x) ,\\
   v(\bm x) &= \sum_{i=1}^n v_i \, \phi_i(\bm x) .
   \end{aligned}

The coefficients :math:`\{u_j\}` and :math:`\{v_i\}` are the nodal values of :math:`u`
and :math:`v`, respectively. Inserting the expressions :math:numref:`eq-nodal-values`
into :math:numref:`eq-general-weak-form`, we obtain the inner-products

.. math::
   :label: eq-inner-prods

   \langle v,u \rangle = \bm v^T M \bm u , \qquad  \langle v,f\rangle =  \bm v^T \bm b \,.

Here, we have introduced the mass matrix, :math:`M`, and the right-hand side,
:math:`\bm b`,

.. math::
   M_{ij} :=  (\phi_i,\phi_j), \;\; \qquad b_{i} :=  \langle \phi_i, f \rangle,

each defined for index sets :math:`i,j \; \in \; \{1,\dots,n\}`.


.. _Laplace Operator:

Laplace's Operator
----------------------------------------

The Laplace's operator used in BP3-BP6 is defined via the following variational
formulation, i.e., find :math:`u \in V^p` such that for all :math:`v \in V^p`

.. math::
   a(v,u) = \langle v,f \rangle , \,

where now :math:`a (v,u)` expresses the continuous bilinear form defined on
:math:`V^p` for sufficiently regular :math:`u`, :math:`v`, and :math:`f`, that is:

.. math::
   \begin{aligned}
   a(v,u) &:= \int_{\Omega}\nabla v \, \cdot \, \nabla u \, dV ,\\
   \langle v,f \rangle &:= \int_{\Omega} \, v \, f \, dV .
   \end{aligned}

After substituting the same formulations provided in :math:numref:`eq-nodal-values`,
we obtain

.. math::
   a(v,u) = \bm v^T K \bm u ,

in which we have introduced the stiffness (diffusion) matrix, :math:`K`, defined as

.. math::
   K_{ij} = a(\phi_i,\phi_j),

for index sets :math:`i,j \; \in \; \{1,\dots,n\}`.
