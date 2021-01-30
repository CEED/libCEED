## libCEED: Navier-Stokes Example

This page provides a description of the Navier-Stokes example for the libCEED library, based on PETSc.

The Navier-Stokes problem solves the compressible Navier-Stokes equations in three dimensions using an
explicit time integration. The state variables are mass density, momentum density, and energy density.

The main Navier-Stokes solver for libCEED is defined in [`navierstokes.c`](navierstokes.c)
with different problem definitions according to the application of interest.

Build by using

`make`

and run with

`./navierstokes`

Available runtime options are:

|  Option                               | Meaning                                                                                         |
| :-------------------------------------| :-----------------------------------------------------------------------------------------------|
| `-ceed`                               | CEED resource specifier                                                                         |
| `-test`                               | Run in test mode                                                                                |
| `-problem`                            | Problem to solve (`advection`, `advection2d`, `density_current`, or `euler_vortex`)             |
| `-problem_advection_wind`             | Wind type in Advection (`rotation` or `translation`)                                            |
| `-problem_advection_wind_translation` | Constant wind vector when `-problem_advection_wind translation`                                 |
| `-problem_euler_mean_velocity`        | Constant mean velocity vector in `euler_vortex`                                                 |
| `-vortex_strength`                    | Strength of vortex in `euler_vortex`                                                            |
| `-stab`                               | Stabilization method                                                                            |
| `-implicit`                           | Use implicit time integartor formulation                                                        |
| `-bc_wall`                            | Use wall boundary conditions on this list of faces                                              |
| `-bc_slip_x`                          | Use slip boundary conditions, for the x component, on this list of faces                        |
| `-bc_slip_y`                          | Use slip boundary conditions, for the y component, on this list of faces                        |
| `-bc_slip_z`                          | Use slip boundary conditions, for the z component, on this list of faces                        |
| `-viz_refine`                         | Use regular refinement for visualization                                                        |
| `-degree`                             | Polynomial degree of tensor product basis (must be >= 1)                                        |
| `-units_meter`                        | 1 meter in scaled length units                                                                  |
| `-units_second`                       | 1 second in scaled time units                                                                   |
| `-units_kilogram`                     | 1 kilogram in scaled mass units                                                                 |
| `-units_Kelvin`                       | 1 Kelvin in scaled temperature units                                                            |
| `-theta0`                             | Reference potential temperature                                                                 |
| `-thetaC`                             | Perturbation of potential temperature                                                           |
| `-P0`                                 | Atmospheric pressure                                                                            |
| `-E_wind`                             | Total energy of inflow wind                                                                     |
| `-N`                                  | Brunt-Vaisala frequency                                                                         |
| `-cv`                                 | Heat capacity at constant volume                                                                |
| `-cp`                                 | Heat capacity at constant pressure                                                              |
| `-g`                                  | Gravitational acceleration                                                                      |
| `-lambda`                             | Stokes hypothesis second viscosity coefficient                                                  |
| `-mu`                                 | Shear dynamic viscosity coefficient                                                             |
| `-k`                                  | Thermal conductivity                                                                            |
| `-CtauS`                              | Scale coefficient for stabilization tau (nondimensional)                                        |
| `-strong_form`                        | Strong (1) or weak/integrated by parts (0) advection residual                                   |
| `-lx`                                 | Length scale in x direction                                                                     |
| `-ly`                                 | Length scale in y direction                                                                     |
| `-lz`                                 | Length scale in z direction                                                                     |
| `-rc`                                 | Characteristic radius of thermal bubble                                                         |
| `-resx`                               | Resolution in x                                                                                 |
| `-resy`                               | Resolution in y                                                                                 |
| `-resz`                               | Resolution in z                                                                                 |
| `-center`                             | Location of bubble center                                                                       |
| `-dc_axis`                            | Axis of density current cylindrical anomaly, or {0,0,0} for spherically symmetric               |
| `-output_freq`                        | Frequency of output, in number of steps                                                         |
| `-continue`                           | Continue from previous solution                                                                 |
| `-degree`                             | Polynomial degree of tensor product basis                                                       |
| `-qextra`                             | Number of extra quadrature points                                                               |
| `-qextra_boundary`                    | Number of extra quadrature points on in/outflow faces                                           |
| `-output_dir`                         | Output directory                                                                                |

For the case of a square/cubic mesh, the list of face indices to be used with `-bc_wall` and/or `-bc_slip_x`,
`-bc_slip_y`, and `-bc_slip_z` are:

* 2D:
  - faceMarkerBottom = 1;
  - faceMarkerRight  = 2;
  - faceMarkerTop    = 3;
  - faceMarkerLeft   = 4;
* 3D:
  - faceMarkerBottom = 1;
  - faceMarkerTop    = 2;
  - faceMarkerFront  = 3;
  - faceMarkerBack   = 4;
  - faceMarkerRight  = 5;
  - faceMarkerLeft   = 6;

### Advection

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

#### Initial Conditions

Mass Density:
    Constant mass density of 1.0

Momentum Density:
    Rotational field in x,y with no momentum in z

Energy Density:
    Maximum of 1. x0 decreasing linearly to 0. as radial distance increases
    to 1/8, then 0. everywhere else

#### Boundary Conditions

This problem is solved for two test cases with different BCs.

##### Rotation

Mass Density:
    0.0 flux

Momentum Density:
    0.0

Energy Density:
    0.0 flux

##### Translation

Mass Density:
    0.0 flux

Momentum Density:
    0.0

Energy Density:

Inflow BCs:
   *E = E</sub>wind</sub>*

Outflow BCs:
   *E = E</sub>boundary</sub>*

Both In/Outflow BCs for E are applied weakly.


### Euler Traveling Vortex

This problem solves the 3D Euler equations for vortex evolution provided
in On the Order of Accuracy and Numerical Performance of Two Classes of
Finite Volume WENO Schemes, Zhang, Zhang, and Shu (2011).

State Variables:

   *q = ( rho, U<sub>1</sub>, U<sub>2</sub>, U<sub>3</sub>, E )*

   *rho* - Mass Density

   *U<sub>i</sub>*  - Momentum Density   ,  *U<sub>i</sub> = rho u<sub>i</sub>*

   *E*   - Total Energy Density,  *E  = P / (gamma - 1) + rho (u u) / 2*

Euler Equations:

   *drho/dt + div( U )                               = 0*

   *dU/dt   + div( rho (u x u) + P I<sub>3</sub> )   = 0*

   *dE/dt   + div( (E + P) u )                       = 0*

Constants:

   *c<sub>v</sub>*              ,  Specific heat, constant volume

   *c<sub>p</sub>*              ,  Specific heat, constant pressure

   *gamma  = c<sub>p</sub> / c<sub>v</sub>*,  Specific heat ratio

   *epsilon*                    ,  Vortex Strength

#### Initial Conditions

Temperature:

   *T   = 1 - (gamma - 1) epsilon^2 exp(1 - r^2) / (8 gamma pi^2)*

Entropy:

   *S = 1* , Constant entropy

Density:

   *rho = (T/S)^(1 / (gamma - 1))*

Pressure:

   *P = rho T*

Velocity:

   *u<sub>i</sub>  = 1 + epsilon exp((1 - r^2)/2) [yc - y, x - xc, 0] / (2 pi)*

   *r        = sqrt( (x - xc)^2 + (y - yc)^2 )*
    with *(xc,yc)* center of the xy-plane in the domain

#### Boundary Conditions

For this problem, in/outflow BCs are implemented where the validity of the weak
form of the governing equations is extended to the outflow.
For the inflow fluxes, prescribed T_inlet and P_inlet are converted to
conservative variables and applied weakly.

### Density Current

This problem solves the full compressible Navier-Stokes equations, using
operator composition and design of coupled solvers in the context of atmospheric
modeling. This problem uses the formulation given in Semi-Implicit Formulations
of the Navier-Stokes Equations: Application to Nonhydrostatic Atmospheric Modeling,
Giraldo, Restelli, and Lauter (2010).

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

#### Initial Conditions

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


#### Boundary Conditions

Mass Density:
    0.0 flux

Momentum Density:
    0.0

Energy Density:
    0.0 flux

### Time Discretization

For all different problems, the time integration is performed with an explicit
or implicit formulation.

### Space Discretization

The geometric factors and coordinate transformations required for the integration of the weak form
for the interior domain and for the boundaries are described in the files [`common.h`](common.h)
and [`setup-boundary.h`](setup-boundary.h), respectively.
