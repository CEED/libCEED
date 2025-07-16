## libCEED Basic Examples

Two examples are provided that rely only upon libCEED without any external libraries.

### Example 1: ex1-volume

This example uses the mass matrix to compute the length, area, or volume of a region, depending upon runtime parameters.

### Example 2: ex2-surface

This example uses the diffusion matrix to compute the surface area of a region, in 1D, 2D or 3D, depending upon runtime parameters.

### Example 3: ex3-volume

This example uses the mass matrix to compute the length, area, or volume of a region, depending upon runtime parameters.
Unlike ex1, this example also adds the diffusion matrix to add a zero contribution to this calculation while demonstrating the ability of libCEED to handle multiple basis evaluation modes on the same input and output vectors.
