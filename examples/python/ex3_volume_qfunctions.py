# Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
# All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
#
# SPDX-License-Identifier: BSD-2-Clause
#
# This file is part of CEED:  http://github.com/ceed

# QFunctions for the ex3-volume example

def build_mass_diff(ctx, q, inputs, outputs):
    """Build quadrature data for a mass + diffusion operator.
    
    At every quadrature point, compute w/det(J).adj(J).adj(J)^T and store
    the symmetric part of the result.
    
    Args:
        ctx: QFunction context with 'dim' and 'space_dim' fields
        q: Number of quadrature points
        inputs: QFunction inputs (Jacobians, quadrature weights)
        outputs: QFunction outputs (quadrature data)
    """
    # in[0] is Jacobians with shape [dim, dim, Q]
    # in[1] is quadrature weights, size (Q)
    J = inputs[0]
    w = inputs[1]
    q_data = outputs[0]
    dim = ctx["dim"]
    
    if dim == 1:
        for i in range(q):
            # Mass
            q_data[0][i] = w[i] * J[0][0][i]
            # Diffusion
            q_data[1][i] = w[i] / J[0][0][i]
    
    elif dim == 2:
        for i in range(q):
            # J: 0 2   q_data: 0 2   adj(J):  J22 -J12
            #    1 3           2 1           -J10  J00
            J00 = J[0][0][i]
            J10 = J[0][1][i]
            J01 = J[1][0][i]
            J11 = J[1][1][i]
            qw = w[i] / (J00 * J11 - J10 * J01)
            
            # Mass
            q_data[0][i] = w[i] * (J00 * J11 - J10 * J01)
            # Diffusion
            q_data[1][i] = qw * (J01 * J01 + J11 * J11)
            q_data[2][i] = qw * (J00 * J00 + J10 * J10)
            q_data[3][i] = -qw * (J00 * J01 + J10 * J11)
    
    elif dim == 3:
        for i in range(q):
            # Compute the adjoint
            A = [[0 for _ in range(3)] for _ in range(3)]
            
            for j in range(3):
                for k in range(3):
                    # Equivalent code with J as a VLA and no mod operations:
                    # A[k][j] = J[j+1][k+1]*J[j+2][k+2] - J[j+1][k+2]*J[j+2][k+1]
                    A[k][j] = (
                        J[(j+1)%3][(k+1)%3][i] * J[(j+2)%3][(k+2)%3][i] -
                        J[(j+1)%3][(k+2)%3][i] * J[(j+2)%3][(k+1)%3][i]
                    )
            
            # Compute quadrature weight / det(J)
            det_j = J[0][0][i] * A[0][0] + J[0][1][i] * A[0][1] + J[0][2][i] * A[0][2]
            qw = w[i] / det_j
            
            # Mass
            q_data[0][i] = w[i] * det_j
            
            # Diffusion - stored in Voigt convention
            # 1 6 5
            # 6 2 4
            # 5 4 3
            q_data[1][i] = qw * (A[0][0] * A[0][0] + A[0][1] * A[0][1] + A[0][2] * A[0][2])
            q_data[2][i] = qw * (A[1][0] * A[1][0] + A[1][1] * A[1][1] + A[1][2] * A[1][2])
            q_data[3][i] = qw * (A[2][0] * A[2][0] + A[2][1] * A[2][1] + A[2][2] * A[2][2])
            q_data[4][i] = qw * (A[1][0] * A[2][0] + A[1][1] * A[2][1] + A[1][2] * A[2][2])
            q_data[5][i] = qw * (A[0][0] * A[2][0] + A[0][1] * A[2][1] + A[0][2] * A[2][2])
            q_data[6][i] = qw * (A[0][0] * A[1][0] + A[0][1] * A[1][1] + A[0][2] * A[1][2])


def build_mass_diff_single(ctx, q, inputs, outputs):
    """Build quadrature data for a mass + diffusion operator using a single component.
    
    This QFunction is similar to build_mass_diff, but it works with a single component
    (x-coordinate) instead of all components. It assumes that the mesh is a Cartesian
    grid, so the Jacobians can be computed from the x-coordinates.
    
    Args:
        ctx: QFunction context with 'dim' field
        q: Number of quadrature points
        inputs: QFunction inputs (x-coordinates, quadrature weights)
        outputs: QFunction outputs (quadrature data)
    """
    # in[0] is x-coordinates with shape [1, Q]
    # in[1] is quadrature weights, size (Q)
    x = inputs[0]
    w = inputs[1]
    q_data = outputs[0]
    dim = ctx["dim"]
    
    if dim == 1:
        for i in range(q):
            # Mass
            q_data[0][i] = w[i] * x[0][i]
            # Diffusion
            q_data[1][i] = w[i] / x[0][i]
    
    elif dim == 2:
        for i in range(q):
            # For a Cartesian grid, the Jacobian is diagonal
            # We'll use a simple approximation based on the x-coordinate
            J00 = x[0][i]
            J11 = x[0][i]
            det_j = J00 * J11
            qw = w[i] / det_j
            
            # Mass
            q_data[0][i] = w[i] * det_j
            # Diffusion
            q_data[1][i] = qw * J11 * J11
            q_data[2][i] = qw * J00 * J00
            q_data[3][i] = 0.0
    
    elif dim == 3:
        for i in range(q):
            # For a Cartesian grid, the Jacobian is diagonal
            # We'll use a simple approximation based on the x-coordinate
            J00 = x[0][i]
            J11 = x[0][i]
            J22 = x[0][i]
            det_j = J00 * J11 * J22
            qw = w[i] / det_j
            
            # Mass
            q_data[0][i] = w[i] * det_j
            # Diffusion - stored in Voigt convention
            # 1 6 5
            # 6 2 4
            # 5 4 3
            q_data[1][i] = qw * J00 * J00
            q_data[2][i] = qw * J11 * J11
            q_data[3][i] = qw * J22 * J22
            q_data[4][i] = 0.0
            q_data[5][i] = 0.0
            q_data[6][i] = 0.0


def apply_mass_diff(ctx, q, inputs, outputs):
    """Apply mass + diffusion operator.
    
    Apply the action of the operator at quadrature points.
    
    Args:
        ctx: QFunction context with 'dim' field
        q: Number of quadrature points
        inputs: QFunction inputs (solution values, gradients, quadrature data)
        outputs: QFunction outputs (result values, gradients)
    """
    u = inputs[0]       # Solution values
    ug = inputs[1]      # Solution gradients
    q_data = inputs[2]  # Quadrature data
    v = outputs[0]      # Result values
    vg = outputs[1]     # Result gradients
    dim = ctx["dim"]
    
    if dim == 1:
        for i in range(q):
            # Mass
            v[0][i] = q_data[0][i] * u[0][i]
            # Diffusion
            vg[0][i] = q_data[1][i] * ug[0][i]
    
    elif dim == 2:
        for i in range(q):
            # Mass
            v[0][i] = q_data[0][i] * u[0][i]
            # Diffusion
            vg[0][i] = q_data[1][i] * ug[0][i] + q_data[3][i] * ug[1][i]
            vg[1][i] = q_data[3][i] * ug[0][i] + q_data[2][i] * ug[1][i]
    
    elif dim == 3:
        for i in range(q):
            # Mass
            v[0][i] = q_data[0][i] * u[0][i]
            # Diffusion
            vg[0][i] = q_data[1][i] * ug[0][i] + q_data[6][i] * ug[1][i] + q_data[5][i] * ug[2][i]
            vg[1][i] = q_data[6][i] * ug[0][i] + q_data[2][i] * ug[1][i] + q_data[4][i] * ug[2][i]
            vg[2][i] = q_data[5][i] * ug[0][i] + q_data[4][i] * ug[1][i] + q_data[3][i] * ug[2][i]