# Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and other CEED contributors
# All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
#
# SPDX-License-Identifier: BSD-2-Clause
#
# This file is part of CEED:  http://github.com/ceed

import sys
import os
from sysconfig import get_config_var
import argparse
import math
import numpy as np
import libceed
import ctypes


def parse_arguments():
    """Parse command line arguments for surface area computation

    Returns:
        Namespace: Parsed arguments with fields:
            ceed: CEED resource specifier
            dim: Problem dimension (1-3)
            mesh_degree: Mesh polynomial degree
            solution_degree: Solution polynomial degree
            num_qpts: Number of quadrature points
            problem_size: Approximate problem size
            test: Test mode flag
            quiet: Suppress output flag
            gallery: Use gallery QFunctions flag
    """
    parser = argparse.ArgumentParser(description="libCEED surface area example")
    parser.add_argument("-c", "--ceed", default="/cpu/self",
                        help="libCEED resource specifier (default: /cpu/self)")
    parser.add_argument("-d", "--dim", type=int, default=3,
                        help="Problem dimension (1-3) (default: 3)")
    parser.add_argument("-m", "--mesh-degree", type=int, default=4,
                        help="Mesh polynomial degree (default: 4)")
    parser.add_argument("-p", "--solution-degree", type=int, default=4,
                        help="Solution polynomial degree (default: 4)")
    parser.add_argument("-q", "--quadrature-points", type=int, default=6,
                        help="Number of quadrature points (default: 6)")
    parser.add_argument("-s", "--problem-size", type=int, default=-1,
                        help="Approximate problem size (default: ~256k)")
    parser.add_argument("-t", "--test", action="store_true",
                        help="Test mode (reduced problem size)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress output")
    parser.add_argument("-g", "--gallery", action="store_true",
                        help="Use gallery QFunctions")

    args = parser.parse_args()
    if args.dim not in [1, 2, 3]:
        parser.error("Dimension must be 1, 2, or 3")
    return args


def get_cartesian_mesh_size(dim, degree, prob_size):
    """Determine Cartesian mesh size for given problem size

    Args:
        dim: Spatial dimension (1-3)
        degree: Polynomial degree
        prob_size: Target problem size

    Returns:
        list: Number of elements in each dimension
    """
    # Calculate number of elements needed
    num_elem = prob_size // (degree ** dim)

    # Find smallest power of 2 >= num_elem
    s = 0
    while num_elem > 1:
        num_elem = num_elem / 2
        s += 1

    # Distribute across dimensions
    r = s % dim
    num_xyz = []
    for d in range(dim):
        sd = s // dim
        if r > 0:
            sd += 1
            r -= 1
        num_xyz.append(1 << sd)
    return num_xyz


def build_cartesian_restriction(ceed, dim, num_xyz, degree, num_comp, num_q_comp, num_qpts, create_qdata=False):
    """Build element restriction for Cartesian grid

    Args:
        ceed: libCEED context
        dim: Spatial dimension
        num_xyz: Elements per dimension
        degree: Polynomial degree
        num_comp: Number of components
        num_q_comp: Number of quadrature data components
        num_qpts: Quadrature points per dimension
        build_qdata: Flag to build restriction for quadrature data

    Returns:
        tuple: (elem_restriction, size, q_data_restriction, num_elem, elem_qpts)
    """
    p = degree + 1  # Nodes per element per dimension
    num_nodes = p ** dim
    elem_qpts = num_qpts ** dim

    # Calculate grid parameters
    nd = []
    num_elem = 1
    scalar_size = 1
    for d in range(dim):
        num_elem *= num_xyz[d]
        nd.append(num_xyz[d] * (p - 1) + 1)  # Nodes per dimension
        scalar_size *= nd[d]

    size = scalar_size * num_comp

    # Create element connectivity
    elem_nodes = np.zeros(num_elem * num_nodes, dtype=np.int32)
    for e in range(num_elem):
        # Get element coordinates
        e_xyz = [0] * dim
        re = e
        for d in range(dim):
            e_xyz[d] = re % num_xyz[d]
            re //= num_xyz[d]

        # Calculate global node numbers
        for n in range(num_nodes):
            g_node = 0
            g_stride = 1
            r_node = n
            for d in range(dim):
                g_node += (e_xyz[d] * (p - 1) + r_node % p) * g_stride
                g_stride *= nd[d]
                r_node //= p
            elem_nodes[e * num_nodes + n] = g_node

    # Create restrictions
    elem_restriction = ceed.ElemRestriction(
        num_elem, num_nodes, num_comp, scalar_size, size, elem_nodes)

    q_data_restriction = None
    if create_qdata:
        strides = np.array([1, elem_qpts, elem_qpts * num_q_comp], dtype=np.int32)
        q_data_restriction = ceed.StridedElemRestriction(
            num_elem, elem_qpts, num_q_comp, num_elem * elem_qpts * num_q_comp, strides)

    return elem_restriction, size, q_data_restriction, num_elem, elem_qpts


def set_cartesian_mesh_coords(ceed, dim, num_xyz, mesh_degree, mesh_coords):
    """Create Cartesian mesh coordinates

    Args:
        ceed: libCEED context
        dim: Spatial dimension
        num_xyz: Elements per dimension
        mesh_degree: Mesh polynomial degree
        mesh_coords: CeedVector to hold mesh coordinates

    Returns:
        Vector: Mesh coordinates
    """
    p = mesh_degree + 1
    nd = []
    scalar_size = 1
    for d in range(dim):
        nd.append(num_xyz[d] * (p - 1) + 1)
        scalar_size *= nd[d]

    # Get Lobatto nodes (quadrature points)
    nodes, _ = ceed.lobatto_quadrature(p)
    nodes = 0.5 + 0.5 * nodes  # Map from [-1,1] to [0,1]

    # Create coordinates
    coords = np.zeros(scalar_size * dim)
    for gs_node in range(scalar_size):
        r_node = gs_node
        for d in range(dim):
            d_1d = r_node % nd[d]
            elem_id = d_1d // (p - 1)
            node_id = d_1d % (p - 1)
            coords[gs_node + scalar_size * d] = (elem_id + nodes[node_id]) / num_xyz[d]
            r_node //= nd[d]

    mesh_coords.set_array(coords, cmode=libceed.COPY_VALUES)
    return mesh_coords


def transform_mesh_coords(dim, mesh_size, mesh_coords, use_sin=True):
    """Transform mesh coordinates and return exact surface area

    Args:
        dim: Spatial dimension
        mesh_size: Total mesh vector size
        mesh_coords: Mesh coordinates vector
        use_sin: Use sinusoidal transformation

    Returns:
        float: Tuple with exact volume and surface area for transformed mesh
    """
    exact_volume = {1: 1.0, 2: 3. / 4. * np.pi, 3: 3. / 4. * np.pi}[dim]
    exact_area = {1: 2.0, 2: 4.0, 3: 6.0}[dim]

    # Apply transformation to coordinates
    num_nodes = mesh_size // dim
    with mesh_coords.array_write() as coords:
        if dim == 1:
            for i in range(num_nodes):
                x = coords[i] - 0.5
                coords[i] = 0.5 + (1.0 / np.sqrt(3.0)) * np.sin((2.0 / 3.0) * np.pi * x)
        else:
            if use_sin:
                for i in range(num_nodes):
                    u = 1. + coords[i]
                    v = np.pi / 2. * coords[i + num_nodes]
                    coords[i] = u * np.cos(v)
                    coords[i + num_nodes] = u * np.sin(v)
            else:
                for i in range(num_nodes):
                    x = coords[i] - 0.5
                    coords[i] = 0.5 + (1.0 / np.sqrt(3.0)) * np.sin((2.0 / 3.0) * np.pi * x)

    return (exact_volume, exact_area)


def find_qfs_so(name, path):
    """Find the QFunctions shared library.
    Returns:
        Filepath to shared library object
    """
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)


def load_qfs_so():
    """Load the QFunctions shared library.
    Returns:
        Loaded shared library object
    """
    file_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "build")
    qfs_so = find_qfs_so(
        "libceed_c_qfunctions" + get_config_var("EXT_SUFFIX"),
        file_dir)

    # Load library
    return ctypes.cdll.LoadLibrary(qfs_so)
