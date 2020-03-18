.. _example-petsc-elasticity:

Solid mechanics elasticity mini-app
========================================

This example is located in the subdirectory :file:`examples/solids`. It solves
the steady-state static balance momentum equations using unstructured high-order
finite element/spectral element spatial discretizations. Moreover, the solid
mechanics elasticity example has been developed using PETSc, so that the
pointwise physics (defined at quadrature points) is separated from the
parallelization and meshing concerns.


.. _problem-linear-elasticity:

----------------------------------------

The strong form of the static balance linear momentum at small strain for the three-dimensional linear elasticity problem is given by :cite:`hughes2012finite`:

.. math::
   :label: lin-elas
   
   \nabla \cdot \boldsymbol{\sigma} + \boldsymbol{g} = \boldsymbol{0} 


where :math:`\boldsymbol{\sigma}`, :math:`\boldsymbol{\epsilon}`
and :math:`\boldsymbol{g}` are stress, strain and forcing functions
respectively. Integrating by parts on the divergence term, we arrive at the weak form the of equation :math:numref:`lin-elas`:

.. math::

   \int_{\Omega}{ \nabla \boldsymbol{v} \colon \boldsymbol{\sigma}} dV - \int_{d\Omega}{\boldsymbol{v} \cdot \left(\boldsymbol{\sigma}_t \cdot \hat{\boldsymbol{n}}\right)} dS + \int_{\Omega}{\boldsymbol{v} \cdot \boldsymbol{g}} dV = 0

where :math:`\boldsymbol{\sigma}_t \cdot \hat{\boldsymbol{n}} dS` is typically
replaced with a boundary condition.

The stress-strain relationship (constitutive law) is given by:

.. math::

   \boldsymbol{\sigma} = \boldsymbol{S} \boldsymbol{\epsilon}

where 

.. math::

   \boldsymbol{\epsilon} = \dfrac{1}{2}\left(\boldsymbol{\nabla} \boldsymbol{u} + \boldsymbol{\nabla} \boldsymbol{u}^T \right)

and

.. math::

   \boldsymbol{S} = \dfrac{E}{(1+\nu)(1-2\nu)}
   \left[
     \begin{array}{cccccc} 
        1-\nu & \nu & \nu & & & \\
          \nu & 1 - \nu & \nu & & & \\
          \nu & \nu &  1 - \nu & & & \\
          & & & \dfrac{1 - 2\nu}{2} & & \\    
         & & & &\dfrac{1 - 2\nu}{2} & \\
         & & & & & \dfrac{1 - 2\nu}{2} \\   
     \end {array}
   \right] 

where :math:`E` is the Young’s modulus and :math:`\nu` is the Poisson’s ratio.

.. _problem-hyper-small-strain:

Hyperelasticity at Small Strain
----------------------------------------

The strong form of the static balance linear momentum at small strain for the three-dimensional neo-Hookean Hyperelasticity problem is given by
:cite:`hughes2012finite`:

.. math::

   \nabla \cdot \boldsymbol{\sigma} + \boldsymbol{g} = \boldsymbol{0} 

Integrating by parts on the divergence term, we arrive at the weak form:

.. math::

   \int_{\Omega}{ \nabla \boldsymbol{v} \colon \boldsymbol{\sigma}} dV - \int_{d\Omega}{\boldsymbol{v} \cdot \left(\boldsymbol{\sigma}_t \cdot \hat{\boldsymbol{n}}\right)} dS + \int_{\Omega}{\boldsymbol{v} \cdot \boldsymbol{g}} dV = 0

where :math:`\boldsymbol{\sigma}_t \cdot \hat{\boldsymbol{n}} dS` is typically replaced with a boundary condition.

The small strain version of a Neo-Hookean hyperelasticity material is given as
follows:

.. math::
   :label: clss
   
   \boldsymbol{\sigma} = \lambda \ln(1 + \boldsymbol{\epsilon_v)} \boldsymbol{I}_3 + 2\mu \boldsymbol{\epsilon}

where :math:`\boldsymbol{\sigma}`, :math:`\boldsymbol{\epsilon}`, are stress and
strain respectively. Strain is defined by

.. math::

   \boldsymbol{\epsilon} = \dfrac{1}{2}\left(\boldsymbol{\nabla} \boldsymbol{u} + \boldsymbol{\nabla} \boldsymbol{u}^T \right)

and :math:`\mu` and  :math:`\lambda` are the Lamé parameters.
:math:`\boldsymbol{\epsilon}_v` is known as *Volumetric Strain*:

.. math::

   \boldsymbol{\epsilon}_v = \boldsymbol{\epsilon}_{11} + \boldsymbol{\epsilon}_{22} + \boldsymbol{\epsilon}_{33} 

Also :math:`\boldsymbol{I}_3` is a :math:`3 \times 3` identity matrix.

Equation :math:numref:`clss` in indicial notation is given by:

.. math::
   \sigma_{ij} = \lambda ln(1 + \epsilon_v)\delta_{ij} + 2\mu\epsilon_{ij}

where its derivative in indicial notation is:

.. math::
   :label: derss

   \dfrac{\partial{\sigma_{ij}}}{\partial{\epsilon_{kl}}} = \bar{\lambda}\delta_{ij}\delta_{kl} + 2\mu \delta_{ik} \delta_{jl}

with,

.. math::

   \bar{\lambda} = \dfrac{\lambda}{1+\epsilon_v}

Equation :math:numref:`derss` can be written in matrix form as follows:

.. math::
   :label: mdss

   \left[
     \begin{array}{c} 
       d\sigma_{11} \\
       d\sigma_{22} \\
       d\sigma_{33} \\
       d\sigma_{12} \\
       d\sigma_{13} \\
       d\sigma_{23}       
    \end {array}
   \right]  = 
   \left[
     \begin{array}{cccccc} 
       2\mu +\bar{\lambda} & \bar{\lambda} & \bar{\lambda} & & & \\
        \bar{\lambda} & 2\mu +\bar{\lambda} & \bar{\lambda} & & & \\
        \bar{\lambda} & \bar{\lambda} & 2\mu +\bar{\lambda} & & & \\
        & & & \mu & & \\    
        & & & &\mu & \\
        & & & & & \mu \\   
     \end {array}
   \right] 
   \left[
     \begin{array}{c} 
       d\epsilon_{11} \\
       d\epsilon_{22} \\
       d\epsilon_{33} \\
       d\epsilon_{12} \\
       d\epsilon_{13} \\
       d\epsilon_{23}       
     \end {array}
   \right]
   

.. _problem-hyperelasticity-finite-strain:

Hyperelasticity at Finite Strain
----------------------------------------

In the *total Lagrangian* approach for the neo-Hookean Hyperelasticity
probelm, the discrete equations are formulated with respect to the reference
configuration. The independent variables are :math:`X' and :math:`t'. The
dependent variable is the displacement :math:`u(X,t)`. The notation for
elasticity at finite strain is inspired by :cite:`holzapfel2000nonlinear` to
distinguish between the current and reference configurations.
**Capital letters** refer to **reference** and *small letters* refer to
*current* configurations.


The strong form of the static balance of linear-momentum at
*Finite Strain* (Total Lagrangian) is given by:

.. math::
   :label: sblFinS

   \nabla_X \cdot \boldsymbol{P} + \rho_0 \boldsymbol{g} = \boldsymbol{0}
 
where :math:`_X` in :math:`\nabla_X` indicates the reference configuration in
the finite strain regime. :math:`\boldsymbol{P}` and :math:`\boldsymbol{g}` are
:math:`2^{nd}` *Piola-Kirchhoff stress* and the prescribed forcing
function respectively. :math:`\rho_0` is known as the *reference* mass
density.

The constitutive law of the material is given by:

.. math::
   :label: 1st2nd
   
   \boldsymbol{P} = \boldsymbol{F} \cdot \boldsymbol{S}

where,

.. math::

   \boldsymbol{S} = \mu \boldsymbol{1} + \left[\lambda \ln(J) - \mu \right] \boldsymbol{C}^{-1}

:math:`\boldsymbol{P}` and :math:`\boldsymbol{S}` are the first and second
Piola-Kirchhoff stresses, respectfully; :math:`\mu` and :math:`\lambda` are the
Lamé parameters; :math:`\boldsymbol{C} = \boldsymbol{F}^T \cdot \boldsymbol{F}`
is right Cauchy-Green tensor, and :math:`\boldsymbol{F}` is the deformation
gradient in reference configuration :math:`(\boldsymbol{1} + \nabla \boldsymbol{u})`;
and :math:`J = det(\boldsymbol{F})` is the Jacobian of deformation.

It is crucial to distinguish between the current and reference element in the Total Lagrangian Finite Strain regime. Therefore, we switch to the indicial notation:

.. math::

    \int_{B}{\boldsymbol{v} \cdot \left(\nabla_X \cdot \boldsymbol{P} + \rho \boldsymbol{g}\right)} JdV = \boldsymbol{0}

and in indicial notation we have,

.. math::

   \int_{B_0}{v_i \left(\dfrac{\partial{P_{iI}}}{\partial{X_I}} + \rho_0 g_i \right)} dV = 0

By Integration by part we arrive at the weak form:

.. math::

   \int_{B_0}{\dfrac{\partial{v_i P_{iI}}}{\partial{X_I}}}dV =
   - \int_{B_0}{v_i \rho_0 g_i}dV
   - \int_{\Gamma_0^t}{v_i t_i}dA

where :math:`t_i` is a prescribed boundary written in terms of reference
configuration.

The constitutive law in indicial notation is given by:

.. math::

   P_{iI} = F_{iB}S_{BI} 

Therfore, its material derivative is given by

.. math::
   :label: mtfs

   \dfrac{\partial P_{iI}}{\partial F_{aA}} = \delta_{ai}S_{AI} + \left[\lambda F_{Aa}^{-1} F_{Ii}^{-1} 
   -\left( \lambda ln(J) - \mu\right)\left(F_{Ai}^{-1} F_{Ia}^{-1} + \delta_{ai} C^{-1}_{AI}  \right)   \right]
