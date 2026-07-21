#!/usr/bin/env python3
# Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and other CEED contributors.
# All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
#
# SPDX-License-Identifier: BSD-2-Clause
#
# This file is part of CEED:  http://github.com/ceed
#
# libCEED example solving a CEED benchmark problem (BP) with PETSc
#
# This is a Python port of examples/petsc/bpsraw.c and uses the same QFunctions,
# from examples/petsc/qfunctions/bps.
#
# Sample runs:
#
#     python bpsraw.py
#     python bpsraw.py -problem bp1 -degree 3
#     mpiexec -n 4 python bpsraw.py -problem bp1 -degree 3 -ceed /gpu/cuda

import argparse
import math
import os
import sys

import numpy as np
import petsc4py
from mpi4py import MPI

import libceed
import ex_common as common

# BP types, matching the bp_types[] enum used by bpsraw.c
BP_TYPES = ("bp1", "bp2", "bp3", "bp4", "bp5", "bp6")

# Per-BP settings, matching the bp_options[] table in bpsraw.c
BP_OPTIONS = {
    "bp1": {
        "num_comp_u": 1,
        "q_data_size": 1,
        "q_extra": 1,
        "source": "bp1.h",
        "setup_geo": "SetupMassGeo",
        "setup_rhs": "SetupMassRhs",
        "apply": "Mass",
        "error": "Error",
        "in_mode": libceed.EVAL_INTERP,
        "out_mode": libceed.EVAL_INTERP,
        "q_mode": libceed.GAUSS,
    },
}


def int_pair(value):
    """Parse a pair of comma separated integers, as PetscOptionsIntArray does

    Args:
        value: String of two comma separated integers

    Returns:
        list: The two parsed integers
    """
    entries = [int(entry) for entry in value.split(",")]
    if len(entries) != 2:
        raise argparse.ArgumentTypeError("expected two comma separated "
                                         f"integers, got '{value}'")
    return entries


def parse_arguments(argv=None):
    """Parse the options owned by this example

    Options that are not recognized here are left for PETSc.

    Args:
        argv: Argument list, defaults to sys.argv[1:]

    Returns:
        tuple: (parsed arguments, remaining arguments for PETSc)
    """
    parser = argparse.ArgumentParser(
        description="CEED BPs in PETSc", add_help=False)
    parser.add_argument("-problem", default="bp1", choices=BP_TYPES,
                        help="CEED benchmark problem to solve")
    parser.add_argument("-degree", type=int, default=1,
                        help="Polynomial degree of tensor product basis")
    parser.add_argument("-q_extra", type=int, default=None,
                        help="Number of extra quadrature points")
    parser.add_argument("-ceed", default="/cpu/self",
                        help="CEED resource specifier")
    parser.add_argument("-local", type=int, default=1000,
                        help="Target number of locally owned nodes per process")
    parser.add_argument("-test", action="store_true",
                        help="Testing mode (do not print unless error is large)")
    parser.add_argument("-benchmark", action="store_true",
                        help="Benchmarking mode (prints benchmark statistics)")
    parser.add_argument("-write_solution", action="store_true",
                        help="Write solution for visualization")
    parser.add_argument("-ksp_max_it_clip", type=int_pair, default=[5, 20],
                        help="Min and max number of iterations to use during "
                             "benchmarking")
    return parser.parse_known_args(argv)


# Initialize PETSc with only the arguments it owns, so that this example's own options are not reported back as
# unused. When imported, such as by the test suite, the caller's arguments are not ours to forward.
_args, _petsc_argv = parse_arguments(
    None if __name__ == "__main__" else [])
petsc4py.init([sys.argv[0]] + _petsc_argv)
from petsc4py import PETSc  # noqa: E402

# petsc4py exposes the converged reasons as integer attributes, not an enum
KSP_CONVERGED_REASONS = {value: name for name, value
                         in vars(PETSc.KSP.ConvergedReason).items()
                         if isinstance(value, int)}


def split3(size, reverse=False):
    """Split an integer into three nearly equal factors

    Args:
        size: Integer to factor
        reverse: Assign the larger factors to the last dimensions first

    Returns:
        list: Three factors whose product is size
    """
    p = [0, 0, 0]
    size_left = size
    for d in range(3):
        part = int(math.ceil(size_left ** (1.0 / (3 - d))))
        while part * (size_left // part) != size_left:
            part += 1
        idx = 2 - d if reverse else d
        p[idx] = part
        size_left //= part
    return p


def global_nodes(p, i_rank, degree, mesh_elem):
    """Number of nodes owned by the given process in each dimension

    Args:
        p: Process grid dimensions
        i_rank: Coordinates of the process in the grid
        degree: Polynomial degree
        mesh_elem: Local elements per dimension

    Returns:
        list: Owned nodes in each of the three dimensions
    """
    return [degree * mesh_elem[d] + (1 if i_rank[d] == p[d] - 1 else 0)
            for d in range(3)]


def global_start(p, i_rank, degree, mesh_elem):
    """Index of the first node owned by the given process

    Args:
        p: Process grid dimensions
        i_rank: Coordinates of the process in the grid
        degree: Polynomial degree
        mesh_elem: Local elements per dimension

    Returns:
        int: Global index of the first owned node, or -1 if not found
    """
    start = 0
    for i in range(p[0]):
        for j in range(p[1]):
            for k in range(p[2]):
                if [i, j, k] == list(i_rank):
                    return start
                m_nodes = global_nodes(p, (i, j, k), degree, mesh_elem)
                start += m_nodes[0] * m_nodes[1] * m_nodes[2]
    return -1


def create_restriction(ceed, mesh_elem, P, num_comp):
    """Create an element restriction for a tensor product element

    Args:
        ceed: libCEED context
        mesh_elem: Local elements per dimension
        P: Nodes per dimension per element (degree + 1)
        num_comp: Number of components

    Returns:
        ElemRestriction: Element restriction for the local mesh
    """
    num_elem = mesh_elem[0] * mesh_elem[1] * mesh_elem[2]
    m_nodes = [mesh_elem[d] * (P - 1) + 1 for d in range(3)]

    # Get indices; offsets are CeedInt, which is 32 bit
    idx = np.empty(num_elem * P * P * P, dtype=np.int32)
    idx_p = 0
    for i in range(mesh_elem[0]):
        for j in range(mesh_elem[1]):
            for k in range(mesh_elem[2]):
                for ii in range(P):
                    node_i = i * (P - 1) + ii
                    for jj in range(P):
                        node_j = j * (P - 1) + jj
                        for kk in range(P):
                            node_k = k * (P - 1) + kk
                            node = (node_i * m_nodes[1] + node_j) * m_nodes[2] + node_k
                            idx[idx_p + (ii + P * (jj + P * kk))] = num_comp * node
                idx_p += P * P * P

    l_size = m_nodes[0] * m_nodes[1] * m_nodes[2] * num_comp
    return ceed.ElemRestriction(num_elem, P * P * P, num_comp, 1, l_size, idx, cmode=libceed.USE_POINTER)


class DeviceArray:
    """Expose a raw CUDA device pointer through the CUDA array interface

    libCEED reads the device pointer of an array from this interface, so this is how a PETSc device array is
    handed to libCEED without a copy.
    """

    def __init__(self, pointer, size, dtype):
        """Wrap a raw device pointer in the CUDA array interface

        Args:
            pointer: Raw CUDA device pointer
            size: Number of elements
            dtype: NumPy dtype of the elements
        """
        self.__cuda_array_interface__ = {
            "shape": (size,),
            "typestr": np.dtype(dtype).str,
            "data": (int(pointer), False),
            "version": 2,
        }


class CeedMatCtx:
    """Context for the PETSc Mat that applies the libCEED operator

    This plays the role of the MatShell and MatMult_Mass in bpsraw.c.
    """

    def __init__(self, ceed, op_apply, l_to_g, X_loc):
        """Set up the vectors and scatter the operator applies with

        Args:
            ceed: libCEED context
            op_apply: libCEED operator applied in mult()
            l_to_g: Local-to-global scatter
            X_loc: Local vector, duplicated for input and output staging
        """
        self.ceed = ceed
        self.op_apply = op_apply
        self.l_to_g = l_to_g
        self.X_loc = X_loc
        self.Y_loc = X_loc.duplicate()

        # These only ever wrap the arrays of the PETSc Vecs above, see mult()
        nloc = X_loc.getSize()
        self.x_ceed = ceed.Vector(nloc)
        self.y_ceed = ceed.Vector(nloc)

        # C: MemTypeP2C(), the libCEED memory type of a PETSc Vec's array
        self.length = nloc
        self.on_device = "cuda" in X_loc.getType()
        self.mem_type = libceed.MEM_DEVICE if self.on_device else libceed.MEM_HOST

    def set_ceed_array(self, ceed_vec, vec, mode):
        """Point a CeedVector at a PETSc Vec's array, without copying

        C: VecGetArrayAndMemType() + CeedVectorSetArray(CEED_USE_POINTER). petsc4py has no getArrayAndMemType,
        so the device pointer is fetched with getCUDAHandle and wrapped for libCEED; on the host, getArray()
        already returns a view of the Vec's own memory.

        Args:
            ceed_vec: CeedVector to point at the array
            vec: PETSc Vec whose array is borrowed
            mode: "r" for read only access, "w" for write access

        Returns:
            The CUDA handle to restore later, or None on the host
        """
        if self.on_device:
            handle = vec.getCUDAHandle(mode)
            array = DeviceArray(handle, self.length, self.ceed.scalar_type())
        else:
            handle = None
            array = vec.getArray(readonly=(mode == "r"))
        ceed_vec.set_array(array, memtype=self.mem_type, cmode=libceed.USE_POINTER)
        return handle

    def restore_ceed_array(self, vec, handle, mode):
        """Release a device array borrowed by set_ceed_array

        Args:
            vec: PETSc Vec whose array was borrowed
            handle: CUDA handle returned by set_ceed_array, or None
            mode: Access mode used in set_ceed_array
        """
        if handle is not None:
            vec.restoreCUDAHandle(handle, mode)

    def mult(self, A, X, Y):
        """Apply the libCEED operator to a global vector

        Args:
            A: The PETSc Mat wrapping this context
            X: Input global vector
            Y: Output global vector
        """
        # Global-to-local
        self.l_to_g.begin(X, self.X_loc, addv=PETSc.InsertMode.INSERT, mode=PETSc.Scatter.Mode.REVERSE)
        self.l_to_g.end(X, self.X_loc, addv=PETSc.InsertMode.INSERT, mode=PETSc.Scatter.Mode.REVERSE)

        # Setup libCEED vectors
        x_handle = self.set_ceed_array(self.x_ceed, self.X_loc, "r")
        y_handle = self.set_ceed_array(self.y_ceed, self.Y_loc, "w")

        # Apply libCEED operator
        self.op_apply.apply(self.x_ceed, self.y_ceed)

        # Restore arrays; the C also detaches them with CeedVectorTakeArray, which the Python bindings do not
        # wrap, so they are rebound instead
        self.restore_ceed_array(self.X_loc, x_handle, "r")
        self.restore_ceed_array(self.Y_loc, y_handle, "w")

        # Local-to-global
        Y.zeroEntries()
        self.l_to_g.begin(self.Y_loc, Y, addv=PETSc.InsertMode.ADD, mode=PETSc.Scatter.Mode.FORWARD)
        self.l_to_g.end(self.Y_loc, Y, addv=PETSc.InsertMode.ADD, mode=PETSc.Scatter.Mode.FORWARD)


def compute_error_max(ctx, op_error, X, target, mpi_comm):
    """Compute the maximum pointwise error against the true solution

    Args:
        ctx: CeedMatCtx providing the ceed context and scatters
        op_error: libCEED operator computing the pointwise error
        X: Global solution vector
        target: CeedVector holding the true solution
        mpi_comm: mpi4py communicator for the reduction

    Returns:
        float: Maximum pointwise error across all processes
    """
    length = target.get_length()
    collocated_error = ctx.ceed.Vector(length)
    collocated_error.set_value(0.0)

    # Global-to-local
    ctx.l_to_g.begin(X, ctx.X_loc, addv=PETSc.InsertMode.INSERT, mode=PETSc.Scatter.Mode.REVERSE)
    ctx.l_to_g.end(X, ctx.X_loc, addv=PETSc.InsertMode.INSERT, mode=PETSc.Scatter.Mode.REVERSE)

    # Setup libCEED vector
    x_handle = ctx.set_ceed_array(ctx.x_ceed, ctx.X_loc, "r")

    # Apply error operator
    op_error.apply(ctx.x_ceed, collocated_error)

    # Restore PETSc array
    ctx.restore_ceed_array(ctx.X_loc, x_handle, "r")

    # Reduce max error
    with collocated_error.array_read(memtype=libceed.MEM_HOST) as e:
        local_max = float(np.max(np.abs(e))) if length > 0 else 0.0

    return mpi_comm.allreduce(local_max, op=MPI.MAX)


def example_bps(args):
    """Solve a CEED benchmark problem using libCEED and PETSc

    Args:
        args: Parsed command line arguments

    Returns:
        int: 0 on success
    """
    dim = 3
    num_comp_x = 3

    comm = PETSc.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    bp_choice = args.problem
    bp_opts = BP_OPTIONS[bp_choice]
    num_comp_u = bp_opts["num_comp_u"]
    q_data_size = bp_opts["q_data_size"]
    q_extra = bp_opts["q_extra"] if args.q_extra is None else args.q_extra
    degree = args.degree
    local_nodes = args.local
    test_mode = args.test
    ksp_max_it_clip = args.ksp_max_it_clip
    P = degree + 1
    Q = P + q_extra

    # Set up libCEED
    ceed = libceed.Ceed(args.ceed)

    # MatMult hands PETSc's array to libCEED with USE_POINTER, so both must use the same scalar type
    if np.dtype(ceed.scalar_type()) != np.dtype(PETSc.ScalarType):
        raise SystemExit("libCEED and PETSc must use the same scalar type, "
                         f"got {ceed.scalar_type()} and "
                         f"{np.dtype(PETSc.ScalarType)}")

    if ceed.get_preferred_memtype() == libceed.MEM_HOST:
        mem_type_backend = "host"
        default_vec_type = PETSc.Vec.Type.STANDARD
    else:
        mem_type_backend = "device"
        resource = ceed.get_resource()
        if "/gpu/cuda" in resource:
            default_vec_type = PETSc.Vec.Type.CUDA
        elif "/gpu/hip" in resource:
            default_vec_type = PETSc.Vec.Type.HIP
        else:
            default_vec_type = PETSc.Vec.Type.STANDARD

    # Determine size of process grid
    p = split3(size, reverse=False)

    # Find a nicely composite number of elements no less than local_nodes
    start = max(1, local_nodes // (degree * degree * degree))
    for local_elem in range(start, 10 ** 9):
        mesh_elem = split3(local_elem, reverse=True)
        if max(mesh_elem) // min(mesh_elem) <= 2:
            break

    # Find my location in the process grid
    pstride = [p[1] * p[2], p[2], 1]
    rank_left = rank
    i_rank = [0, 0, 0]
    for d in range(3):
        i_rank[d] = rank_left // pstride[d]
        rank_left -= i_rank[d] * pstride[d]

    m_nodes = global_nodes(p, i_rank, degree, mesh_elem)

    # Setup global vector; setSizes takes (local, global), since its second positional argument is a block size
    X = PETSc.Vec().create(comm=comm)
    X.setType(default_vec_type)
    X.setSizes([m_nodes[0] * m_nodes[1] * m_nodes[2] * num_comp_u, PETSc.DECIDE])
    X.setFromOptions()
    X.setUp()

    # Print summary
    gsize = X.getSize()
    vec_type = X.getType()
    used_resource = ceed.get_resource()

    if not test_mode and rank == 0:
        print()
        print(f"-- CEED Benchmark Problem {BP_TYPES.index(bp_choice) + 1}"
              " -- libCEED + PETSc (Python) --")
        print("  PETSc:")
        print(f"    PETSc Vec Type                     : {vec_type}")
        print("  libCEED:")
        print(f"    libCEED Backend                    : {used_resource}")
        print(f"    libCEED Backend MemType            : {mem_type_backend}")
        print("  Mesh:")
        print(f"    Solution Order (P)                 : {P}")
        print(f"    Quadrature  Order (Q)              : {Q}")
        print(f"    Global nodes                       : {gsize // num_comp_u}")
        print(f"    Process Decomposition              : "
              f"{p[0]} {p[1]} {p[2]}")
        print(f"    Local Elements                     : "
              f"{mesh_elem[0] * mesh_elem[1] * mesh_elem[2]} = "
              f"{mesh_elem[0]} {mesh_elem[1]} {mesh_elem[2]}")
        print(f"    Owned nodes                        : "
              f"{m_nodes[0] * m_nodes[1] * m_nodes[2]} = "
              f"{m_nodes[0]} {m_nodes[1]} {m_nodes[2]}")
        print(f"    DoF per node                       : {num_comp_u}")

    l_nodes = [mesh_elem[d] * degree + 1 for d in range(dim)]
    l_size = l_nodes[0] * l_nodes[1] * l_nodes[2]

    X_loc = PETSc.Vec().create(comm=PETSc.COMM_SELF)
    X_loc.setType(default_vec_type)
    X_loc.setSizes([l_size * num_comp_u, PETSc.DECIDE])
    X_loc.setFromOptions()
    X_loc.setUp()

    # Create local-to-global scatter
    g_start = np.empty((2, 2, 2), dtype=PETSc.IntType)
    g_m_nodes = np.empty((2, 2, 2), dtype=object)
    for idx in np.ndindex(g_start.shape):
        ijk_rank = [i_rank[d] + idx[d] for d in range(3)]
        g_start[idx] = global_start(p, ijk_rank, degree, mesh_elem)
        g_m_nodes[idx] = global_nodes(p, ijk_rank, degree, mesh_elem)

    l_to_g_ind = np.empty(l_size, dtype=PETSc.IntType)
    l_to_g_ind_0 = np.empty(l_size, dtype=PETSc.IntType)
    loc_ind = np.empty(l_size, dtype=PETSc.IntType)
    l_0_count = 0
    for i in range(l_nodes[0]):
        ir = 1 if i >= m_nodes[0] else 0
        ii = i - ir * m_nodes[0]
        for j in range(l_nodes[1]):
            jr = 1 if j >= m_nodes[1] else 0
            jj = j - jr * m_nodes[1]
            for k in range(l_nodes[2]):
                kr = 1 if k >= m_nodes[2] else 0
                kk = k - kr * m_nodes[2]
                here = (i * l_nodes[1] + j) * l_nodes[2] + k
                l_to_g_ind[here] = (g_start[ir][jr][kr] +
                                    (ii * g_m_nodes[ir][jr][kr][1] + jj) * g_m_nodes[ir][jr][kr][2] + kk)
                if ((i_rank[0] == 0 and i == 0) or
                    (i_rank[1] == 0 and j == 0) or
                    (i_rank[2] == 0 and k == 0) or
                    (i_rank[0] + 1 == p[0] and i + 1 == l_nodes[0]) or
                    (i_rank[1] + 1 == p[1] and j + 1 == l_nodes[1]) or
                        (i_rank[2] + 1 == p[2] and k + 1 == l_nodes[2])):
                    continue
                l_to_g_ind_0[l_0_count] = l_to_g_ind[here]
                loc_ind[l_0_count] = here
                l_0_count += 1

    l_to_g_is = PETSc.IS().createBlock(num_comp_u, l_to_g_ind, comm=comm)
    l_to_g = PETSc.Scatter().create(X_loc, None, X, l_to_g_is)
    l_to_g_is_0 = PETSc.IS().createBlock(num_comp_u, l_to_g_ind_0[:l_0_count].copy(), comm=comm)
    loc_is = PETSc.IS().createBlock(num_comp_u, loc_ind[:l_0_count].copy(), comm=comm)
    l_to_g_0 = PETSc.Scatter().create(X_loc, loc_is, X, l_to_g_is_0)

    # Global-to-global scatter for Dirichlet values, i.e. everything that is not in the range of l_to_g_0
    count_D = 0
    X_loc.zeroEntries()
    X.set(1.0)
    l_to_g_0.begin(X_loc, X, addv=PETSc.InsertMode.INSERT, mode=PETSc.Scatter.Mode.FORWARD)
    l_to_g_0.end(X_loc, X, addv=PETSc.InsertMode.INSERT, mode=PETSc.Scatter.Mode.FORWARD)
    x_start, x_end = X.getOwnershipRange()
    x = X.getArray(readonly=True)
    ind_D = np.empty(x_end - x_start, dtype=PETSc.IntType)
    for i in range(x_end - x_start):
        if x[i] == 1.0:
            ind_D[count_D] = x_start + i
            count_D += 1
    is_D = PETSc.IS().createGeneral(ind_D[:count_D].copy(), comm=comm)
    g_to_g_D = PETSc.Scatter().create(X, is_D, X, is_D)
    is_D.destroy()
    l_to_g_is.destroy()
    l_to_g_is_0.destroy()
    loc_is.destroy()

    # CEED bases
    basis_u = ceed.BasisTensorH1Lagrange(dim, num_comp_u, P, Q, bp_opts["q_mode"])
    basis_x = ceed.BasisTensorH1Lagrange(dim, num_comp_x, 2, Q, bp_opts["q_mode"])

    # CEED restrictions
    elem_restr_u = create_restriction(ceed, mesh_elem, P, num_comp_u)
    elem_restr_x = create_restriction(ceed, mesh_elem, 2, dim)
    num_elem = mesh_elem[0] * mesh_elem[1] * mesh_elem[2]
    elem_size = Q ** 3

    # Strides are CeedInt, which is 32 bit; a 64 bit array is misread
    strides_u = np.array([1, num_comp_u, num_comp_u * elem_size], dtype=np.int32)
    elem_restr_u_i = ceed.StridedElemRestriction(num_elem, elem_size, num_comp_u,
                                                 num_comp_u * num_elem * elem_size, strides_u)
    strides_qd = np.array([1, q_data_size, q_data_size * elem_size], dtype=np.int32)
    elem_restr_qd_i = ceed.StridedElemRestriction(num_elem, elem_size, q_data_size,
                                                  q_data_size * num_elem * elem_size, strides_qd)

    # Set up the mesh coordinates on the unit cube
    shape = [mesh_elem[0] + 1, mesh_elem[1] + 1, mesh_elem[2] + 1]
    length = shape[0] * shape[1] * shape[2]
    x_loc = np.empty(length * num_comp_x, dtype=ceed.scalar_type())
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                node = (i * shape[1] + j) * shape[2] + k
                base = dim * node
                x_loc[base + 0] = (i_rank[0] * mesh_elem[0] + i) / (p[0] * mesh_elem[0])
                x_loc[base + 1] = (i_rank[1] * mesh_elem[1] + j) / (p[1] * mesh_elem[1])
                x_loc[base + 2] = (i_rank[2] * mesh_elem[2] + k) / (p[2] * mesh_elem[2])
    x_coord = ceed.Vector(length * num_comp_x)
    x_coord.set_array(x_loc, memtype=libceed.MEM_HOST, cmode=libceed.USE_POINTER)

    # Load the QFunctions written in C; the path is the source libCEED uses to JIT them on GPU backends
    qfs_so = common.load_qfs_so()
    qf_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qfunctions", "bps")

    bp_source = os.path.join(qf_dir, bp_opts["source"])
    common_source = os.path.join(qf_dir, "common.h")

    # Create the QFunction that builds the operator quadrature data
    qf_setup_geo = ceed.QFunction(1, getattr(qfs_so, bp_opts["setup_geo"]), f"{bp_source}:{bp_opts['setup_geo']}")
    qf_setup_geo.add_input("x", num_comp_x, libceed.EVAL_INTERP)
    qf_setup_geo.add_input("dx", num_comp_x * dim, libceed.EVAL_GRAD)
    qf_setup_geo.add_input("weight", 1, libceed.EVAL_WEIGHT)
    qf_setup_geo.add_output("q_data", q_data_size, libceed.EVAL_NONE)

    # Create the QFunction that sets up the RHS and true solution
    qf_setup_rhs = ceed.QFunction(1, getattr(qfs_so, bp_opts["setup_rhs"]), f"{bp_source}:{bp_opts['setup_rhs']}")
    qf_setup_rhs.add_input("x", num_comp_x, libceed.EVAL_INTERP)
    qf_setup_rhs.add_input("q_data", q_data_size, libceed.EVAL_NONE)
    qf_setup_rhs.add_output("true_soln", num_comp_u, libceed.EVAL_NONE)
    qf_setup_rhs.add_output("rhs", num_comp_u, libceed.EVAL_INTERP)

    # Create the QFunction that applies the operator
    qf_apply = ceed.QFunction(1, getattr(qfs_so, bp_opts["apply"]), f"{bp_source}:{bp_opts['apply']}")
    in_scale = dim if bp_opts["in_mode"] == libceed.EVAL_GRAD else 1
    out_scale = dim if bp_opts["out_mode"] == libceed.EVAL_GRAD else 1
    qf_apply.add_input("u", num_comp_u * in_scale, bp_opts["in_mode"])
    qf_apply.add_input("q_data", q_data_size, libceed.EVAL_NONE)
    qf_apply.add_output("v", num_comp_u * out_scale, bp_opts["out_mode"])

    # Create the QFunction that computes the error
    qf_error = ceed.QFunction(1, getattr(qfs_so, bp_opts["error"]), f"{common_source}:{bp_opts['error']}")
    qf_error.add_input("u", num_comp_u, libceed.EVAL_INTERP)
    qf_error.add_input("true_soln", num_comp_u, libceed.EVAL_NONE)
    qf_error.add_input("qdata", q_data_size, libceed.EVAL_NONE)
    qf_error.add_output("error", num_comp_u, libceed.EVAL_NONE)

    # Create the persistent vectors needed in setup
    num_qpts = basis_u.get_num_quadrature_points()
    q_data = ceed.Vector(q_data_size * num_elem * num_qpts)
    target = ceed.Vector(num_elem * num_qpts * num_comp_u)
    rhs_ceed = ceed.Vector(l_size * num_comp_u)

    # Create the operator that builds the quadrature data for the ceed operator
    op_setup_geo = ceed.Operator(qf_setup_geo)
    op_setup_geo.set_field("x", elem_restr_x, basis_x, libceed.VECTOR_ACTIVE)
    op_setup_geo.set_field("dx", elem_restr_x, basis_x, libceed.VECTOR_ACTIVE)
    op_setup_geo.set_field("weight", libceed.ELEMRESTRICTION_NONE, basis_x, libceed.VECTOR_NONE)
    op_setup_geo.set_field("q_data", elem_restr_qd_i, libceed.BASIS_NONE, libceed.VECTOR_ACTIVE)

    # Create the operator that builds the RHS and true solution
    op_setup_rhs = ceed.Operator(qf_setup_rhs)
    op_setup_rhs.set_field("x", elem_restr_x, basis_x, libceed.VECTOR_ACTIVE)
    op_setup_rhs.set_field("q_data", elem_restr_qd_i, libceed.BASIS_NONE, q_data)
    op_setup_rhs.set_field("true_soln", elem_restr_u_i, libceed.BASIS_NONE, target)
    op_setup_rhs.set_field("rhs", elem_restr_u, basis_u, libceed.VECTOR_ACTIVE)

    # Create the mass or diff operator
    op_apply = ceed.Operator(qf_apply)
    op_apply.set_field("u", elem_restr_u, basis_u, libceed.VECTOR_ACTIVE)
    op_apply.set_field("q_data", elem_restr_qd_i, libceed.BASIS_NONE, q_data)
    op_apply.set_field("v", elem_restr_u, basis_u, libceed.VECTOR_ACTIVE)

    # Create the error operator
    op_error = ceed.Operator(qf_error)
    op_error.set_field("u", elem_restr_u, basis_u, libceed.VECTOR_ACTIVE)
    op_error.set_field("true_soln", elem_restr_u_i, libceed.BASIS_NONE, target)
    op_error.set_field("qdata", elem_restr_qd_i, libceed.BASIS_NONE, q_data)
    op_error.set_field("error", elem_restr_u_i, libceed.BASIS_NONE, libceed.VECTOR_ACTIVE)

    # Set up the matrix-free operator as a PETSc Mat
    ctx = CeedMatCtx(ceed, op_apply, l_to_g, X_loc)
    n = m_nodes[0] * m_nodes[1] * m_nodes[2] * num_comp_u
    mat = PETSc.Mat().create(comm=comm)
    mat.setType(PETSc.Mat.Type.PYTHON)
    mat.setSizes(((n, None), (n, None)))
    mat.setPythonContext(ctx)
    mat.setUp()

    # Set up the RHS; this copy happens once during setup, not in the solve
    rhs = X.duplicate()
    rhs_loc = X_loc.duplicate()
    rhs_loc.zeroEntries()
    op_setup_geo.apply(x_coord, q_data)
    op_setup_rhs.apply(x_coord, rhs_ceed)
    with rhs_ceed.array_read(memtype=libceed.MEM_HOST) as rhs_array:
        rhs_loc.getArray(readonly=False)[:] = rhs_array
        rhs_loc.resetArray()

    rhs.zeroEntries()
    l_to_g.begin(rhs_loc, rhs, addv=PETSc.InsertMode.ADD, mode=PETSc.Scatter.Mode.FORWARD)
    l_to_g.end(rhs_loc, rhs, addv=PETSc.InsertMode.ADD, mode=PETSc.Scatter.Mode.FORWARD)

    # Jacobi with row sums only needs MatMult, unlike the default variant which needs MatGetDiagonal. petsc4py
    # has no PCJacobiSetType, so it is selected through the options database, before the first setup of the PC
    opts = PETSc.Options()
    if bp_choice in ("bp1", "bp2") and not opts.hasName("pc_jacobi_type"):
        opts.setValue("pc_jacobi_type", "rowsum")

    ksp = PETSc.KSP().create(comm=comm)
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.JACOBI if bp_choice in ("bp1", "bp2") else PETSc.PC.Type.NONE)
    pc.setFromOptions()
    ksp.setType(PETSc.KSP.Type.CG)
    ksp.setNormType(PETSc.KSP.NormType.NATURAL)
    ksp.setTolerances(rtol=1e-10)
    ksp.setOperators(mat, mat)

    # First run's performance log is not considered for benchmarking purposes
    mpi_comm = comm.tompi4py()
    ksp.setTolerances(rtol=1e-10, max_it=1)
    t0 = MPI.Wtime()
    ksp.solve(rhs, X)
    my_rt = mpi_comm.allreduce(MPI.Wtime() - t0, op=MPI.MIN)

    # Set maxits based on first iteration timing
    clip = ksp_max_it_clip[0] if my_rt > 0.02 else ksp_max_it_clip[1]
    ksp.setTolerances(rtol=1e-10, max_it=clip)
    ksp.setFromOptions()

    # Timed solve
    X.zeroEntries()
    comm.barrier()
    t0 = MPI.Wtime()
    ksp.solve(rhs, X)
    my_rt = mpi_comm.allreduce(MPI.Wtime() - t0, op=MPI.MIN)

    reason = ksp.getConvergedReason()
    its = ksp.getIterationNumber()
    rnorm = ksp.getResidualNorm()

    if (not test_mode) or reason < 0 or rnorm > 1e-8:
        if rank == 0:
            print("  KSP:")
            print(f"    KSP Type                           : {ksp.getType()}")
            print(f"    KSP Convergence                    : "
                  f"{KSP_CONVERGED_REASONS.get(reason, reason)}")
            print(f"    Total KSP Iterations               : {its}")
            print(f"    Final rnorm                        : {rnorm:e}")

    if not test_mode and rank == 0:
        print("  Performance:")

    max_error = compute_error_max(ctx, op_error, X, target, mpi_comm)
    rt_min = mpi_comm.allreduce(my_rt, op=MPI.MIN)
    rt_max = mpi_comm.allreduce(my_rt, op=MPI.MAX)

    tol = 5e-2
    if (not test_mode) or max_error > tol:
        if rank == 0:
            print(f"    Pointwise Error (max)              : {max_error:e}")
            print(f"    CG Solve Time                      : "
                  f"{rt_max:g} ({rt_min:g}) sec")

    if not test_mode and rank == 0 and rt_max > 0.0 and rt_min > 0.0:
        print(f"    DoFs/Sec in CG                     : "
              f"{1e-6 * gsize * its / rt_max:g} "
              f"({1e-6 * gsize * its / rt_min:g}) million")

    # Write solution for visualization
    if args.write_solution:
        viewer = PETSc.Viewer().createVTK("solution.vtu", "w", comm=comm)
        X.view(viewer)
        viewer.destroy()

    return 0


def main():
    """Run the CEED BPs example on the parsed command line arguments

    Returns:
        int: 0 on success
    """
    return example_bps(_args)


if __name__ == "__main__":
    sys.exit(main())
