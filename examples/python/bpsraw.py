#!/usr/bin/env python3
import argparse
import numpy as np
from petsc4py import PETSc
import libceed
import time
from mpi4py import MPI
# from libceed.ceed import ffi, lib

# Memory type mapping
# MEM_TYPES = {
#     lib.CEED_MEM_HOST: "host",
#     lib.CEED_MEM_DEVICE: "device"
# }

# CLI Options
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-degree', type=int, default=2, help='Polynomial degree (P)')
    parser.add_argument('-q_extra', type=int, default=1, help='Q - P')
    parser.add_argument('-ceed', type=str, default='/cpu/self', help='libCEED backend')
    parser.add_argument('-local', type=int, default=1000, help='Target number of local nodes per process')
    parser.add_argument('-test', action='store_true', help='Run test problem')
    parser.add_argument('-benchmark', action='store_true', help='Enable benchmarking output')
    return parser.parse_args()


def main():
    args = parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    ceed = libceed.Ceed(args.ceed)
    solution_order = args.degree + 1
    quadrature_order = solution_order + args.q_extra
    local_nodes = args.local
    num_dofs = local_nodes * size
    num_elements = (num_dofs - 1) // (solution_order - 1)

    if not args.test:
        # vec_type = "mpi"
        # memtype_ptr = ffi.new("CeedMemType *")
        # lib.CeedGetPreferredMemType(ceed.ptr, memtype_ptr)
        # memtype_str = MEM_TYPES[memtype_ptr[0]]
        if rank == 0:
            print("-- CEED Benchmark Problem 1 -- libCEED + PETSc --")
            # print("  PETSc:")
            # print(f"    PETSc Vec Type                     : {vec_type}")
            print("  libCEED:")
            print(f"    libCEED Backend                    : {args.ceed}")
            # print(f"    libCEED Backend MemType            : {memtype_str}")
            print("  Mesh:")
            print(f"    Solution Order (P)                 : {solution_order}")
            print(f"    Quadrature  Order (Q)              : {quadrature_order}")
            print(f"    Global nodes                       : {num_dofs}")
            print(f"    Process Decomposition              : {size} 1 1")
            print(f"    Local Elements                     : {num_elements}")
            print(f"    Owned nodes                        : {local_nodes}")
            print("    DoF per node                       : 1")

    offsets = [i * (solution_order - 1) + j for i in range(num_elements) for j in range(solution_order)]
    offsets = np.array(offsets, dtype=np.int32)

    elem_restriction = ceed.ElemRestriction(num_elements, solution_order, 1, 1, num_dofs, offsets, cmode=libceed.USE_POINTER)

    basis = ceed.BasisTensorH1Lagrange(1, 1, solution_order, quadrature_order, libceed.GAUSS)
    strides = np.array([1, quadrature_order, quadrature_order], dtype="int32")
    qdata_restriction = ceed.StridedElemRestriction(num_elements, quadrature_order, 1, num_elements * quadrature_order, strides)

    qdata = ceed.Vector(num_elements * quadrature_order)
    qfunction_setup = ceed.QFunctionByName("Mass1DBuild")
    op_setup = ceed.Operator(qfunction_setup)
    op_setup.set_field("dx", elem_restriction, basis, libceed.VECTOR_ACTIVE)
    op_setup.set_field("weights", libceed.ELEMRESTRICTION_NONE, basis, libceed.VECTOR_NONE)
    op_setup.set_field("qdata", qdata_restriction, libceed.BASIS_NONE, qdata)

    x_array = np.linspace(0, 1, num_dofs, dtype=np.float64)
    dx_vector = ceed.Vector(num_dofs)
    dx_vector.set_array(x_array, cmode=libceed.USE_POINTER)
    op_setup.apply(dx_vector, qdata)

    rhs_vector = ceed.Vector(num_dofs)
    rhs_vector.set_value(0.0)

    qfunction_rhs = ceed.QFunctionByName("MassApply")
    op_rhs = ceed.Operator(qfunction_rhs)
    op_rhs.set_field("u", elem_restriction, basis, libceed.VECTOR_ACTIVE)
    op_rhs.set_field("qdata", qdata_restriction, libceed.BASIS_NONE, qdata)
    op_rhs.set_field("v", elem_restriction, basis, libceed.VECTOR_ACTIVE)

    dummy_input = ceed.Vector(num_dofs)
    dummy_input.set_value(1.0)
    op_rhs.apply(dummy_input, rhs_vector)

    qfunction = ceed.QFunctionByName("MassApply")
    op = ceed.Operator(qfunction)
    op.set_field("u", elem_restriction, basis, libceed.VECTOR_ACTIVE)
    op.set_field("qdata", qdata_restriction, libceed.BASIS_NONE, qdata)
    op.set_field("v", elem_restriction, basis, libceed.VECTOR_ACTIVE)

    matrix = PETSc.Mat().createAIJ([num_dofs, num_dofs], comm=PETSc.COMM_WORLD)
    matrix.setFromOptions()
    matrix.setUp()
    for i in range(num_dofs):
        e_vec = ceed.Vector(num_dofs)
        r_vec = ceed.Vector(num_dofs)
        e_vec.set_value(0.0)
        r_vec.set_value(0.0)
        with e_vec.array_write() as arr:
            arr[i] = 1.0
        op.apply(e_vec, r_vec)
        with r_vec.array_read() as r_arr:
            matrix.setValues(range(num_dofs), [i], r_arr)
    matrix.assemble()

    rhs = PETSc.Vec().createMPI(num_dofs, comm=PETSc.COMM_WORLD)
    sol = PETSc.Vec().createMPI(num_dofs, comm=PETSc.COMM_WORLD)
    rhs.set(0.0)
    sol.set(0.0)
    with rhs_vector.array_read() as rhs_arr:
        rhs.setValues(range(num_dofs), rhs_arr)
    rhs.assemble()

    ksp = PETSc.KSP().create()
    ksp.setOperators(matrix)
    ksp.setType("cg")
    ksp.getPC().setType("jacobi")
    ksp.setFromOptions()

    comm.Barrier()
    start_time = time.time()
    ksp.solve(rhs, sol)
    end_time = time.time()

    iterations = ksp.getIterationNumber()
    residual_norm = ksp.getResidualNorm()
    average_value = sol.sum() / num_dofs if args.test else None

    solve_time = end_time - start_time
    dofs_per_sec = (num_dofs * iterations) / solve_time * 1e-6

    if rank == 0:
        print("  KSP:")
        print(f"    KSP Iterations                        : {iterations}")
        print(f"    Residual norm                         : {residual_norm}")
        if args.test:
            print(f"    Average value of solution             : {average_value}")
        if args.benchmark:
            print("  Performance:")
            print(f"    CG Solve Time                         : {solve_time:.8f} seconds")
            print(f"    DoFs/Sec in CG                        : {dofs_per_sec:.4f} million")

if __name__ == "__main__":
    main()
