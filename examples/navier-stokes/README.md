# libCEED: Navier-Stokes Example

This page provides a description of the Navier-Stokes example for the libCEED library, based on PETSc.

The Navier-Stokes problem solves the compressible Navier-Stokes equations in three dimensions using an
explicit time integration. The state variables are mass density, momentum density, and energy density.

The main Navier-Stokes solver for libCEED is defined in `/examples/navier-stokes/navierstokes.c`
with different problem definitions according to the application of interest.

## Advection

This problem solves the convection (advection) equation for the total (scalar) energy density,
transported by the (vector) velocity field.

This is 3D advection given in two formulations based upon the weak form.

State Variables:

   *q = ( rho, U<sub>1</sub>, U<sub>2</sub>, U<sub>3</sub>, E )*
    
   *rho* - Mass Density
    
   *U<sub>i</sub>*  - Momentum Density    ,   *U<sub>i</sub> = rho ui*
    
   *E*   - Total Energy Density,   *E  = rho Cv T + rho (u u) / 2 + rho g z*
    
Advection Equation:

   *dE/dt + div( E _u_ ) = 0*

### Initial Conditions

Mass Density:
    Constant mass density of 1.0

Momentum Density:
    Rotational field in x,y with no momentum in z

Energy Density:
    Maximum of 1. x0 decreasing linearly to 0. as radial distance increases
    to 1/8, then 0. everywhere else

### Boundary Conditions

Mass Density:
    0.0 flux

Momentum Density:
    0.0

Energy Density:
    0.0 flux

## Density Current

This problem solves the full compressible Navier-Stokes equations, using
operator composition and design of coupled solvers in the context of atmospheric modeling.

The 3D compressible Navier-Stokes equations are formulated in conservation form with state
variables of density, momentum density, and total energy density.

State Variables:

   *q = ( rho, U<sub>1</sub>, U<sub>2</sub>, U<sub>3</sub>, E )*

   *rho* - Mass Density

   *U<sub>i</sub>*  - Momentum Density   ,  *U<sub>i</sub> = rho u<sub>i</sub>*

   *E*   - Total Energy Density,  *E  = rho c<sub>v</sub> T + rho (u u) / 2 + rho g z*

Navier-Stokes Equations:

   *drho/dt + div( U )                               = 0*

   *dU/dt   + div( rho (u x u) + P I<sub>3</sub> ) + rho g khat = div( F<sub>u</sub> )*

   *dE/dt   + div( (E + P) u )                       = div( F<sub>e</sub> )*

Viscous Stress:

   *F<sub>u</sub> = mu (grad( u ) + grad( u )^T + lambda div ( u ) I<sub>3</sub>)*

Thermal Stress:

   *F<sub>e</sub> = u F<sub>u</sub> + k grad( T )*

Equation of State:

   *P = (gamma - 1) (E - rho (u u) / 2 - rho g z)*

Temperature:

   *T = (E / rho - (u u) / 2 - g z) / c<sub>v</sub>*

Constants:

   *lambda = - 2 / 3*,  From Stokes hypothesis

   *mu*              ,  Dynamic viscosity
 
   *k*               ,  Thermal conductivity
 
   *c<sub>v</sub>*              ,  Specific heat, constant volume
 
   *c<sub>p</sub>*              ,  Specific heat, constant pressure
 
   *g*               ,  Gravity
 
   *gamma  = c<sub>p</sub> / c<sub>v</sub>*,  Specific heat ratio

### Initial Conditions

Potential Temperature:

   *theta = thetabar + deltatheta*
 
   *thetabar   = theta0 exp( N * * 2 z / g )*
 
   *deltatheta =
        r <= rc : theta0(1 + cos(pi r)) / 2
        r > rc : 0*
 
   *r        = sqrt( (x - xc) * * 2 + (y - yc) * * 2 + (z - zc) * * 2 )*
    with *(xc,yc,zc)* center of domain

Exner Pressure:

   *Pi = Pibar + deltaPi*

   *Pibar      = g * * 2 (exp( - N * * 2 z / g ) - 1) / (cp theta0 N * * 2)*
 
   *deltaPi    = 0* (hydrostatic balance)

Velocity/Momentum Density:

   *U<sub>i</sub> = u<sub>i</sub> = 0*

Conversion to Conserved Variables:

   *rho = P0 Pi**(c<sub>v</sub>/R<sub>d</sub>) / (R<sub>d</sub> theta)*

   *E   = rho (c<sub>v</sub> theta Pi + (u u)/2 + g z)*

Constants:

   *theta0*          ,  Potential temperature constant
   
   *thetaC*          ,  Potential temperature perturbation
   
   *P0*              ,  Pressure at the surface
   
   *N*               ,  Brunt-Vaisala frequency
   
   *c<sub>v</sub>*              ,  Specific heat, constant volume
   
   *c<sub>p</sub>*              ,  Specific heat, constant pressure
   
   *R<sub>d</sub>*     = c<sub>p</sub> - c<sub>v</sub>,  Specific heat difference
   
   *g*               ,  Gravity
   
   *r<sub>c</sub>*              ,  Characteristic radius of thermal bubble
   
   *l<sub>x</sub>*              ,  Characteristic length scale of domain in x
   
   *l<sub>y</sub>*              ,  Characteristic length scale of domain in y
   
   *l<sub>z</sub>*              ,  Characteristic length scale of domain in z
   

### Boundary Conditions

Mass Density:
    0.0 flux

Momentum Density:
    0.0

Energy Density:
    0.0 flux

## Time Discretization

For all different problems, the time integration is performed with an explicit formulation, therefore
it can be subject to numerical instability, if run for large times or with large time steps.

## Space Discretization

The geometric factors and coordinate transformations required for the integration of the weak form
are described in the file `/examples/navier-stokes/common.h`

