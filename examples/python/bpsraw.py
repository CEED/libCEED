#!/usr/bin/env python3
import sys, petsc4py
petsc4py.init(sys.argv)
import argparse
import time
import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import libceed

def Split3(size, reverse=False):
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

def GlobalNodes(p, i_rank, degree, mesh_elem):
    return [degree * mesh_elem[d] + (1 if i_rank[d] == p[d] - 1 else 0)
            for d in range(3)]

def GlobalStart(p, i_rank, degree, mesh_elem):
    start = 0
    for i in range(p[0]):
        for j in range(p[1]):
            for k in range(p[2]):
                if [i, j, k] == list(i_rank):
                    return start
                m_nodes = GlobalNodes(p, (i, j, k), degree, mesh_elem)
                start += m_nodes[0] * m_nodes[1] * m_nodes[2]
    return -1

def CreateRestriction(ceed, mesh_elem, P, num_comp, ndofs, offsets):
    return ceed.ElemRestriction(mesh_elem, P, num_comp, 1, ndofs, offsets, cmode=libceed.USE_POINTER)

# def CreateRestriction(ceed, mesh_elem, P, num_comp):
#     num_elem = mesh_elem[0] * mesh_elem[1] * mesh_elem[2]
#     m_nodes = [mesh_elem[d] * (P - 1) + 1 for d in range(3)]

#     # Allocate idx array of length num_elem * P^3
#     idx = np.empty(num_elem * P * P * P, dtype=np.int64)
#     idx_p = 0
#     for i in range(mesh_elem[0]):
#         for j in range(mesh_elem[1]):
#             for k in range(mesh_elem[2]):
#                 for ii in range(P):
#                     for jj in range(P):
#                         for kk in range(P):
#                             # k,j,i ordering
#                             idx[idx_p] = num_comp * (((i*(P-1) + ii) * m_nodes[1] + (j*(P-1) + jj)) * m_nodes[2] + (k*(P-1) + kk))
#                             idx_p += 1
#     l_size = m_nodes[0] * m_nodes[1] * m_nodes[2] * num_comp
#     return ceed.ElemRestriction(num_elem, P * P * P, num_comp, 1, l_size, idx, cmode=libceed.USE_POINTER)

# Read command line options
def parse_args():
    parser = argparse.ArgumentParser(description="CEED BPs in PETSc using Python")
    parser.add_argument('-problem', type=int, default=1, help='Polynomial degree (P)')
    parser.add_argument('-degree', type=int, default=1, help='Polynomial degree (P)')
    parser.add_argument('-q_extra', type=int, default=1, help='Extra quadrature points (Q-P)')
    parser.add_argument('-ceed', type=str, default='/cpu/self', help='libCEED resource')
    parser.add_argument('-local', type=int, default=1000, help='Local nodes per MPI rank')
    parser.add_argument('-test', action='store_false', help='Run analytic test')
    parser.add_argument('-benchmark', action='store_false', help='Print performance stats')
    args, petsc_argv = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + petsc_argv
    return args

# PETSc Shell context class
class MatMassCtx:
    def __init__(self, ceed, op_mass, ndofs, comm):
        self.ceed = ceed
        self.op = op_mass
        self.ndofs = ndofs
        self.comm = comm
        self._xglob = np.zeros(ndofs, dtype=float)
        self.u = ceed.Vector(ndofs)
        self.v = ceed.Vector(ndofs)

    def mult(self, A: PETSc.Mat, x: PETSc.Vec, y: PETSc.Vec):
        lo, hi = x.getOwnershipRange()
        self._xglob.fill(0.0)
        self._xglob[lo:hi] = x.array_r
        self.u.set_array(self._xglob,cmode=libceed.USE_POINTER)
        self.v.set_value(0.0)

        self.op.apply(self.u, self.v)
        with self.v.array_read() as va:
            y.setValues(range(lo, hi), va[lo:hi])
        y.assemble()

    # def getRowSum(self, mat: PETSc.Mat, rs: PETSc.Vec):
    #     start, end = rs.getOwnershipRange()
    #     for i in range(start, end):
    #         cols, vals = mat.getRow(i)
    #         rs[i] = sum(abs(v) for v in vals)
    #         mat.restoreRow(i, cols, vals)
    #     rs.assemble()

    # def getDiagonal(self, mat: PETSc.Mat, diag: PETSc.Vec):
    #     ones = PETSc.Vec().createMPI(self.u.get_length(), comm=self.da.comm)
    #     ones.set(1.0)
    #     tmp = ones.duplicate()
    #     self.mult(mat, ones, tmp)
    #     diag.copy(tmp)
    #     diag.assemble()

# Main driver
if __name__ == '__main__':
    dim = 3
    num_comp_x = 3

    args = parse_args()
    comm = PETSc.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # read options / parameters
    opts = PETSc.Options()
    bp_choice = opts.getInt("-problem", 1)
    num_comp_u = 1
    degree = opts.getInt("-degree", 1)
    q_data_size = opts.getInt("-q_data_size", 1)
    q_extra = opts.getInt("-q_extra", 1)
    ceedres = opts.getString("-ceed", "/cpu/self")
    local_nodes = opts.getInt("-local", 1000)
    test_mode = opts.getBool("-test", False)
    benchmark = opts.getBool("-benchmark", False)
    write_sol = opts.getBool("-write_solution", False)
    P = degree + 1
    Q = P + q_extra

    # Set up libCEED
    ceed = libceed.Ceed(args.ceed)

    if ceed.get_preferred_memtype() == libceed.MEM_HOST:
        mem_type_backend = "MEM_HOST"
        default_vec_type = PETSc.Vec.Type.MPI
    else:
        mem_type_backend = "MEM_DEVICE"
        resource = ceed.get_resource()
        if "/gpu/cuda" in resource:
            default_vec_type = PETSc.Vec.Type.CUDA
        elif "/gpu/hip/occa" in resource:
            default_vec_type = PETSc.Vec.Type.MPI
        elif "/gpu/hip" in resource:
            default_vec_type = PETSc.Vec.Type.HIP
        else:
            default_vec_type = PETSc.Vec.Type.MPI

    local_num_elem = local_nodes
    # ndofs = size*local_num_elem*P + 1*size

    p = Split3(size, reverse=False)
    left_rank = rank
    i_rank = [left_rank % p[0], (left_rank//p[0]) % p[1], left_rank//(p[0]*p[1])]

    start = max(1, local_nodes//(degree*degree*degree))
    for local_elem in range(start, 10**9):
        mesh_elem = Split3(local_elem, reverse=True)
        if max(mesh_elem)/min(mesh_elem) <= 2:
            break

    m_nodes = GlobalNodes(p, i_rank, degree, mesh_elem)

    g_nodes_dims = [mesh_elem[0] * p[0] * degree + 1, mesh_elem[1] * p[1] * degree + 1, mesh_elem[2] * p[2] * degree + 1]
    global_nodes = g_nodes_dims[0] * g_nodes_dims[1] * g_nodes_dims[2]
    owned = m_nodes[0] * m_nodes[1] * m_nodes[2] * num_comp_u

    # Setup global vector
    X = PETSc.Vec().create(comm=comm)
    X.setType(default_vec_type)
    X.setSizes([owned, PETSc.DECIDE])
    X.setFromOptions()
    X.setUp()
    gsize = X.getSize()
    
    # ndofs = args.local * size
    # num_elem = (ndofs - 1) // (P - 1)
    vec_type = X.getType()
    used_resource = ceed.get_resource()  

    if rank == 0:
        print()
        print(f"-- CEED Benchmark Problem {bp_choice} -- (libCEED + PETSc).py --")
        print("  PETSc:")
        print(f"    PETSc Vec Type                     : {vec_type}")
        print("  libCEED:")
        print(f"    libCEED Backend                    : {used_resource}")
        print(f"    libCEED Backend MemType            : {mem_type_backend}")
        print("  Mesh:")
        print(f"    Solution Order (P)                 : {P}")
        print(f"    Quadrature  Order (Q)              : {Q}")
        print(f"    Global nodes                       : {global_nodes}")
        print(f"    Process Decomposition              : {p[0]} {p[1]} {p[2]}")
        print(f"    Local Elements                     : {mesh_elem[0]*mesh_elem[1]*mesh_elem[2]} = "
              f"{mesh_elem[0]} {mesh_elem[1]} {mesh_elem[2]}")
        print(f"    Owned nodes                        : {m_nodes[0]*m_nodes[1]*m_nodes[2]} = "
              f"{m_nodes[0]} {m_nodes[1]} {m_nodes[2]}")
        print(f"    DoF per node                       : {num_comp_u}")

    l_nodes = [mesh_elem[d] * degree + 1 for d in range(dim)]
    l_size = l_nodes[0] * l_nodes[1] * l_nodes[2]

    X_loc = PETSc.Vec().create(comm=PETSc.COMM_SELF)
    X_loc.setType(default_vec_type)
    X_loc.setSizes(l_size * num_comp_u, PETSc.DECIDE)
    X_loc.setFromOptions()
    X_loc.setUp()

    # Create local-to-global scatter
    g_start = np.empty((2, 2, 2), dtype=int)
    g_m_nodes = np.empty((2, 2, 2), dtype=object)
    for idx in np.ndindex(g_start.shape):
        ijk_rank = [i_rank[d] + idx[d] for d in range(3)]
        g_start[idx] = GlobalStart(p, ijk_rank, degree, mesh_elem)
        g_m_nodes[idx] = GlobalNodes(p, ijk_rank, degree, mesh_elem)

    l_to_g_ind = np.empty(l_size, dtype=np.int32)
    l_to_g_ind_0 = np.empty(l_size, dtype=np.int32)
    loc_ind = np.empty(l_size, dtype=np.int32)
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
                l_to_g_ind[here] = (g_start[ir][jr][kr] + (ii * g_m_nodes[ir][jr][kr][1] + jj) * g_m_nodes[ir][jr][kr][2] + kk)
                if (
                    (i_rank[0] == 0 and i == 0)
                    or (i_rank[1] == 0 and j == 0)
                    or (i_rank[2] == 0 and k == 0)
                    or (i_rank[0] + 1 == p[0] and i + 1 == l_nodes[0])
                    or (i_rank[1] + 1 == p[1] and j + 1 == l_nodes[1])
                    or (i_rank[2] + 1 == p[2] and k + 1 == l_nodes[2])
                ):
                    continue

                l_to_g_ind_0[l_0_count] = l_to_g_ind[here]
                loc_ind[l_0_count] = here
                l_0_count += 1
    l_to_g_is = PETSc.IS().createBlock(num_comp_u, l_to_g_ind, comm=comm)
    l_to_g = PETSc.Scatter().create(X_loc, None, X, l_to_g_is)
    l_to_g_is_0 = PETSc.IS().createBlock(num_comp_u, l_to_g_ind_0, comm=comm)
    loc_is = PETSc.IS().createBlock(num_comp_u, loc_ind, comm=comm)
    l_to_g_0 = PETSc.Scatter().create( X_loc, loc_is, X, l_to_g_is_0)

    # Create global-to-global scatter for Dirichlet values (everything not in l_to_g_is_0, which is the range of l_to_g_0)
    x_start = 0
    x_end = 0
    count_D = 0
    is_D = None
    X_loc.zeroEntries()
    X.set(1.0)
    l_to_g_0.begin(X_loc, X, addv=PETSc.InsertMode.INSERT, mode=PETSc.Scatter.Mode.FORWARD)
    l_to_g_0.end(X_loc, X, addv=PETSc.InsertMode.INSERT, mode=PETSc.Scatter.Mode.FORWARD)
    x_start, x_end = X.getOwnershipRange()
    x = X.getArray(readonly=True)
    ind_D = np.empty(x_end - x_start, dtype=np.int32)
    for i in range(x_end - x_start):
        if x[i] == 1.0:
            ind_D[count_D] = x_start + i
            count_D += 1
    x = X.getArray(readonly=True)
    is_D = PETSc.IS().createGeneral(ind_D, comm=comm)
    g_to_g_D = PETSc.Scatter().create(X, is_D, X, is_D)
    is_D.destroy()
    l_to_g_is.destroy()
    l_to_g_is_0.destroy()
    loc_is.destroy()

    # COMMENTED OUT 5/6
    # CEED bases
    # basis_u = ceed.BasisTensorH1Lagrange(dim, num_comp_u, P, Q, libceed.GAUSS) # "libceed.GAUSS" need to update for other bps
    # basis_x = ceed.BasisTensorH1Lagrange(dim, num_comp_x, 2, Q, libceed.GAUSS) # "libceed.GAUSS" need to update for other bps

    # # CEED restrictions
    # num_elem = mesh_elem[0] * mesh_elem[1] * mesh_elem[2]
    # offsets = np.array([i*(P-1)+j for i in range(num_elem) for j in range(P)], dtype='int32')
    # # elem_restr_u = CreateRestriction(ceed, num_elem, P, 1, ndofs, offsets)
    # # offsets = np.array([i*(P-1)+j for i in range(num_elem) for j in range(P)], dtype='int32')
    # # elem_restr_u = CreateRestriction(ceed, mesh_elem, P, 1, ndofs, offsets)
    # elem_restr_u = CreateRestriction(ceed, mesh_elem, P, num_comp_u)
    # elem_restr_x = CreateRestriction(ceed, mesh_elem, 2, dim)
    # num_elem = mesh_elem[0] * mesh_elem[1] * mesh_elem[2]
    
    # strides = np.array([1, Q, Q], dtype='int32')
    # strides_u_i  = np.array([num_comp_u, 1, num_comp_u * Q**3], dtype='int32')
    # strides_qd_i = np.array([q_data_size, 1, q_data_size * Q**3], dtype='int32')
    # # q_data_restr = ceed.StridedElemRestriction(num_elem, Q, 1, num_elem*Q, strides)
    # q_data_restr = ceed.StridedElemRestriction(num_elem, Q ** 3, num_comp_u, num_comp_u * num_elem * Q ** 3, strides)
    # elem_restr_u_i = ceed.StridedElemRestriction(num_elem, Q ** 3, num_comp_u, num_comp_u * num_elem * Q ** 3, strides_u_i)
    # elem_restr_qd_i = ceed.StridedElemRestriction(num_elem, Q ** 3, q_data_size, q_data_size * num_elem * Q ** 3, strides_qd_i)
    

    # #  Create the persistent vectors that will be needed in setup
    # num_qpts = basis_u.get_num_quadrature_points()
    # q_data = ceed.Vector(q_data_size * num_elem * num_qpts)
    # target = ceed.Vector(num_elem * num_qpts * num_comp_u)
    # rhs_ceed = ceed.Vector(l_size * num_comp_u)

    # # Create the operator that builds the quadrature data for the ceed operator
    # # op_setup_geo = ceed.Operator(qf_setup_geo,None,None)
    # # op_setup_geo = ceed.Operator(ceed.QFunctionByName("SetupMassGeo"))
    # # op_setup_geo.set_field("x", elem_restr_x, basis_x, libceed.VECTOR_ACTIVE)
    # # op_setup_geo.set_field("dx", elem_restr_x, basis_x, libceed.VECTOR_ACTIVE)
    # # op_setup_geo.set_field("weights", libceed.ELEMRESTRICTION_NONE, basis_x, libceed.VECTOR_NONE)
    # # op_setup_geo.set_field("qdata", elem_restr_qd_i, libceed.BASIS_NONE, q_data)
    # m_nodes_x = [mesh_elem[d]*(2-1) + 1 for d in range(dim)]
    # l_size_x = m_nodes_x[0] * m_nodes_x[1] * m_nodes_x[2] * num_comp_x

    ndofs = l_size * size
    num_elem = mesh_elem[0] * mesh_elem[1] * mesh_elem[2]
    offsets = np.array([i*(P-1)+j for i in range(num_elem) for j in range(P)], dtype='int32')
    elem_restr = CreateRestriction(ceed, num_elem, P, 1, ndofs, offsets)
    # basis = ceed.BasisTensorH1Lagrange(3, 1, P, Q, libceed.GAUSS)
    basis = ceed.BasisTensorH1Lagrange(1, 1, P, Q, libceed.GAUSS)
    strides = np.array([1, Q, Q], dtype='int32')
    qdata_restr = ceed.StridedElemRestriction(num_elem, Q, 1, num_elem*Q, strides)
    # qdata_restr = ceed.StridedElemRestriction(num_elem, Q**3, 1, num_elem*Q**3, strides)

    op_setup = ceed.Operator(ceed.QFunctionByName("Mass1DBuild"))
    op_setup.set_field("dx", elem_restr, basis, libceed.VECTOR_ACTIVE)
    op_setup.set_field("weights", libceed.ELEMRESTRICTION_NONE, basis, libceed.VECTOR_NONE)
    qdata = ceed.Vector(num_elem * Q)
    op_setup.set_field("qdata", qdata_restr, libceed.BASIS_NONE, qdata)
    x_coords = np.linspace(0, 1, ndofs, dtype='float64')
    xv = ceed.Vector(ndofs)
    xv.set_array(x_coords, cmode=libceed.USE_POINTER)
    op_setup.apply(xv, qdata)

    op_rhs = ceed.Operator(ceed.QFunctionByName("MassApply"))
    op_rhs.set_field("u", elem_restr, basis, libceed.VECTOR_ACTIVE)
    op_rhs.set_field("qdata", qdata_restr, libceed.BASIS_NONE, qdata)
    op_rhs.set_field("v", elem_restr, basis, libceed.VECTOR_ACTIVE)

    rhs_vec = ceed.Vector(ndofs)
    dummy = ceed.Vector(ndofs); dummy.set_value(1.0)
    op_rhs.apply(dummy, rhs_vec)

    op_mass = ceed.Operator(ceed.QFunctionByName("MassApply"))
    op_mass.set_field("u", elem_restr, basis, libceed.VECTOR_ACTIVE)
    op_mass.set_field("qdata", qdata_restr, libceed.BASIS_NONE, qdata)
    op_mass.set_field("v", elem_restr, basis, libceed.VECTOR_ACTIVE)

    ctx = MatMassCtx(ceed, op_mass, ndofs, comm)
    A = PETSc.Mat().create(comm=comm)
    A.setSizes([ndofs, ndofs])
    A.setType('python')
    A.setPythonContext(ctx)
    A.setUp()     
    A.setFromOptions()

    b = A.createVecRight()
    x_sol = A.createVecLeft()
    rhs_ceed = rhs_vec
    b.set(0.0)
    with rhs_ceed.array_read() as rhs_arr:
        start, end = b.getOwnershipRange()
        b.setValues(range(start, end), rhs_arr[start:end])
    b.assemble()

    opts = PETSc.Options()
    if not opts.hasName('pc_jacobi_type'):
        opts.setValue('pc_jacobi_type', 'rowsum')
    if not opts.hasName('pc_type'):
        opts.setValue('pc_type', 'jacobi')
    if not opts.hasName('ksp_type'):
        opts.setValue('ksp_type', 'cg')
    if not opts.hasName('ksp_rtol'):
        opts.setValue('ksp_rtol', '1e-10')

    ksp = PETSc.KSP().create(comm=comm)
    ksp.setOperators(A)
    ksp.setNormType(PETSc.KSP.NormType.NATURAL)
    ksp.setFromOptions()
    pc = ksp.getPC()

    # time the solve
    comm.Barrier() 
    my_rt_start = time.time()
    ksp.solve(b, x_sol)
    comm.Barrier()
    my_rt = time.time() - my_rt_start

    rt_min = comm.tompi4py().allreduce(my_rt, MPI.MIN)
    rt_max = comm.tompi4py().allreduce(my_rt, MPI.MAX)
    its = ksp.getIterationNumber()

    dofs_per_sec = ndofs/my_rt/1e6

    local_x = x_sol.getArray()
    local_err = abs(local_x - 1.0).max()
    global_err = comm.tompi4py().allreduce(local_err, MPI.MAX) # temp workaround

    if rank == 0:
        conv_reason_code = ksp.getConvergedReason()
        for name, val in vars(PETSc.KSP.ConvergedReason).items():
            if val == conv_reason_code:
                reason_name = name
                break
        else:
            reason_name = f"UNKNOWN({conv_reason_code})"

        print("  KSP:")
        print(f"    KSP Type                           : {ksp.getType()}")
        print(f"    KSP Convergence                    : {reason_name}")
        print(f"    Total KSP Iterations               : {ksp.getIterationNumber()}")
        print(f"    Final rnorm                        : {ksp.getResidualNorm():.6e}")
        print("  Performance:")
        print(f"    Pointwise Error (max)              : {global_err:e}")
        print(f"    CG Solve Time                      : {rt_max:.6f} ({rt_min:.6f}) sec")
        print(f"    DoFs/Sec in CG                     : {1e-6 * gsize * its / rt_max:.6f} ({1e-6 * gsize * its / rt_min:.6f}) million")