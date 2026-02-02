! Copyright (c) 2017-2026,  Lawrence Livermore National Security,  LLC and other CEED contributors.
! All Rights Reserved. See the top-level LICENSE and NOTICE files for details.

! SPDX-License-Identifier: BSD-2-Clause

! This file is part of CEED:  http:Cgithub.com/ceed

! libCEED Example 1

! This example illustrates a simple usage of libCEED to compute the volume of a 3D body using matrix-free application of a mass operator.
! Arbitrary mesh and solution degrees in 1D,  2D and 3D are supported from the same code.

! The example has no dependencies,  and is designed to be self-contained.
! For additional examples that use external discretization libraries (MFEM,  PETSc,  etc.) see the subdirectories in libceed/examples.

! All libCEED objects use a Ceed device object constructed based on a command line argument (-ceed).

! Build with:

!     make ex1-volume [CEED_DIR = </path/to/libceed>]

! Sample runs:

!     ./ex1-volume-f
!     ./ex1-volume-f -ceed /cpu/self
!     ./ex1-volume-f -ceed /gpu/cuda

! Test in 1D-3D
! TESTARGS(name = "1D User QFunction") -ceed {ceed_resource} -d 1 -t
! TESTARGS(name = "2D User QFunction") -ceed {ceed_resource} -d 2 -t
! TESTARGS(name = "3D User QFunction") -ceed {ceed_resource} -d 3 -t
! TESTARGS(name = "1D Gallery QFunction") -ceed {ceed_resource} -d 1 -t -g
! TESTARGS(name = "2D Gallery QFunction") -ceed {ceed_resource} -d 2 -t -g
! TESTARGS(name = "3D Gallery QFunction") -ceed {ceed_resource} -d 3 -t -g

!> @file
!> libCEED example using mass operator to compute volume

    include 'ex1-volume-f.h'

!-----------------------------------------------------------------------
subroutine getcartesianmeshsize(fe_dim, degree, prob_size, num_xyz)
    implicit none
    integer fe_dim
    integer degree
    integer prob_size
    integer num_xyz(3)

    integer num_elem
    integer s, r, d, sd
    num_elem = prob_size/(degree**fe_dim)
    s = 0

! Use the approximate formula:
!    prob_size ~ num_elem * degree^dim
! find s: num_elem/2 < 2^s <= num_elem

  do while (num_elem > 1)
    num_elem = num_elem/2
    s = s + 1
  end do
  r = mod(s, fe_dim)

  do d = 1, fe_dim
    sd = s/fe_dim
    if (r > 0) then
      sd = sd + 1
      r = r - 1
    end if
    num_xyz(d) = ISHFT(1, sd)
  end do
end

!-----------------------------------------------------------------------
subroutine buildcartesianrestriction(ceed, fe_dim, num_xyz, degree, num_comp, mesh_size, num_qpts, restriction,&
&     q_data_restriction, err)
    implicit none
    include 'ceed/fortran.h'

    integer ceed
    integer fe_dim
    integer num_xyz(3)
    integer degree
    integer num_comp
    integer mesh_size
    integer num_qpts
    integer restriction
    integer q_data_restriction
    integer err

    integer p
    integer num_nodes
    integer elem_qpts
    integer num_elem
    integer scalar_size
    integer nd(3)
    integer elem_nodes_size
    integer e_xyz(3),  re
    integer g_nodes, g_nodes_stride, r_nodes
    integer, dimension (:), allocatable :: elem_nodes

    integer i, j, k

    p = degree + 1
    num_nodes = p**fe_dim
    elem_qpts = num_qpts**fe_dim
    num_elem  = 1
    scalar_size = 1

    do i = 1, fe_dim
      num_elem = num_elem * num_xyz(i)
      nd(i) = num_xyz(i) * (p - 1) + 1
      scalar_size = scalar_size*nd(i)
    end do
    mesh_size = scalar_size*num_comp
! elem:       0         1             n-1
!         |---*-...-*---|---*-...-*---|- ... -|--...--|
! num_nodes:   0   1    p-1  p  p+1     2*p         n*p
    elem_nodes_size = num_elem*num_nodes
    allocate (elem_nodes(elem_nodes_size))

    do i = 1, num_elem
      e_xyz(1) = 1
      e_xyz(2) = 1
      e_xyz(3) = 1
      re = i - 1

      do j = 1, fe_dim
        e_xyz(j) = mod(re, num_xyz(j))
        re = re/num_xyz(j)
      end do

      do j = 1, num_nodes
        g_nodes = 0
        g_nodes_stride = 1
        r_nodes = j - 1

        do k = 1, fe_dim
          g_nodes = g_nodes + (e_xyz(k) * (p - 1) + mod(r_nodes, p)) * g_nodes_stride
          g_nodes_stride = g_nodes_stride * nd(k)
          r_nodes = r_nodes/p
        end do
        elem_nodes((i - 1) * num_nodes + j) = g_nodes
      end do
    end do

    call ceedelemrestrictioncreate(ceed, num_elem, num_nodes, num_comp, scalar_size, mesh_size, ceed_mem_host,&
             &ceed_copy_values, elem_nodes, restriction, err)
    if (q_data_restriction /=  ceed_qfunction_none) then
      call ceedelemrestrictioncreatestrided(ceed, num_elem, elem_qpts, num_comp, num_comp * elem_qpts * num_elem,&
               &ceed_strides_backend, q_data_restriction, err)
    end if
    deallocate (elem_nodes)
end

!-----------------------------------------------------------------------
subroutine transformmeshcoords(fe_dim, mesh_size, coords, exact_volume, err)
    implicit none

    integer fe_dim
    integer mesh_size, scalar_size
    real*8 coords(mesh_size)
    real*8 exact_volume
    real*8 m_pi, m_pi_2
    parameter(m_pi = 3.14159265358979323846d0)
    parameter(m_pi_2 = 1.57079632679489661923d0)
    integer err

    integer i
    real*8 u, v

    scalar_size = mesh_size/fe_dim
    select case (fe_dim)
    case (1)
      do i = 1, scalar_size
        coords(i) = 0.5d0 + (1.d0/sqrt(3.d0)) * sin((2.d0/3.d0) * m_pi * (coords(i) - 0.5d0))
      end do
      exact_volume = 1.d0

    case (2,  3)
      do i = 1, scalar_size
        u = 1.d0 + coords(i)
        v = m_pi_2 * coords(i + scalar_size)

        coords(i)               = u * cos(v)
        coords(i + scalar_size) = u * sin(v)
      end do
      exact_volume = 3.d0/4.d0 * m_pi
    end select
end

!-----------------------------------------------------------------------
subroutine setcartesianmeshcoords(fe_dim, num_xyz, mesh_degree, mesh_coords, exact_volume, err)
    implicit none
    include 'ceed/fortran.h'

    integer fe_dim
    integer num_xyz(3)
    integer mesh_degree
    integer mesh_coords
    real*8 exact_volume
    integer err

    integer p
    integer scalar_size
    integer coords_size
    integer r_nodes
    integer d_1d
    integer nd(3)
    real*8, dimension (:), allocatable :: nodes,  qpts
    real*8, dimension (:), allocatable :: coords
    integer*8 offset
    integer i, j
    p = mesh_degree + 1
    scalar_size = 1

    do i = 1, fe_dim
      nd(i) = num_xyz(i) * (p - 1) + 1
      scalar_size = scalar_size * nd(i)
    end do

    coords_size = scalar_size * fe_dim
    allocate (coords(coords_size))

! The H1 basis uses Lobatto quadrature points as nodes
    allocate (nodes(p))
    allocate (qpts(p))
    call ceedlobattoquadrature(p, nodes, qpts, err)
    deallocate(qpts)
    do i = 1, p
      nodes(i) = 0.5 + 0.5 * nodes(i)
    end do

    do i = 1, scalar_size
      r_nodes = i - 1

      do j = 1, fe_dim
        d_1d  =  mod(r_nodes, nd(j))
        coords(scalar_size * (j - 1) + i) = ((d_1d/(p - 1)) + nodes(mod(d_1d, p - 1) + 1))/num_xyz(j)
        r_nodes = r_nodes/nd(j)
      end do
    end do
    deallocate(nodes)

    call transformmeshcoords(fe_dim, coords_size, coords, exact_volume, err)

    offset = 0
    call ceedvectorsetarray(mesh_coords, ceed_mem_host, ceed_copy_values, coords, offset, err)
    deallocate(coords)
end

!-----------------------------------------------------------------------
program main
    implicit none
    include 'ceed/fortran.h'

    character ceed_spec*32
    integer fe_dim, num_comp_x, mesh_degree, sol_degree, num_qpts
    integer num_elem, num_xyz(3), elem_qpts
    integer prob_size, mesh_size, sol_size
    integer help, test, gallery, benchmark
    integer i, num_args, err
    character arg*32, arg_value*32
    real*8 exact_volume, computed_volume

    integer ceed
    real*8, dimension (:), allocatable :: u_array, v_array
    integer mesh_coords, q_data, u, v
    integer mesh_restriction, sol_restriction, q_data_restriction
    integer mesh_basis, sol_basis
    integer*8 offset
    integer build_ctx
    integer build_ctx_size
    parameter(build_ctx_size = 2)
    integer*8 build_ctx_data(build_ctx_size)
    integer qf_build, qf_apply
    integer op_build, op_apply

    external build_mass, apply_mass

! Initial values
    ceed_spec   = '/cpu/self'
    fe_dim      = 3
    num_comp_x  = 3
    mesh_degree = 4
    sol_degree  = 4
    num_qpts    = mesh_degree + 2
    prob_size   = -1
    help      = 0
    test      = 0
    gallery   = 0
    benchmark = 0

! Process command line arguments
   
    num_args = command_argument_count()
    do i = 1, num_args
      call get_command_argument(i, arg)

      select case (arg)
! LCOV_EXCL_START
        case ('-h')
          help = 1

        case ('-c',  '-ceed')
          call get_command_argument(i + 1, ceed_spec)

        case ('-d')
          call get_command_argument(i + 1, arg_value)
          read(arg_value, '(I10)') fe_dim
          num_comp_x = fe_dim

        case ('-m')
          call get_command_argument(i + 1, arg_value)
          read(arg_value, '(I10)') mesh_degree

        case ('-p')
          call get_command_argument(i + 1, arg_value)
          read(arg_value, '(I10)') sol_degree

        case ('-q')
          call get_command_argument(i + 1, arg_value)
          read(arg_value, '(I10)') num_qpts

        case ('-s')
          call get_command_argument(i + 1, arg_value)
          read(arg_value, '(I10)') prob_size

        case ('-b')
          call get_command_argument(i + 1, arg_value)
          read(arg_value, '(I10)') benchmark
! LCOV_EXCL_STOP

        case ('-t')
          test = 1

        case ('-g')
          gallery = 1
      end select
    end do

    if (prob_size < 0) then
      if (test == 1) then
        prob_size = 8 * 16
      else
        prob_size = 256 * 1024
      end if
    end if

! Print options
    if ((test /= 1) .OR. (help == 1)) then
! LCOV_EXCL_START
    write (*, *) 'Selected options: [command line option] : <current value>'
    write (*, *) '  Ceed specification     [-c] : ', ceed_spec
    write (*, *) '  Mesh dimension         [-d] : ', fe_dim
    write (*, *) '  Mesh degree            [-m] : ', mesh_degree
    write (*, *) '  Solution degree        [-p] : ', sol_degree
    write (*, *) '  Num. 1D quadrature pts [-q] : ', num_qpts
    write (*, *) '  Approx. # unknowns     [-s] : ', prob_size
    if (gallery == 1) then
      write (*, *) '  QFunction source       [-g] : gallery'
    else
      write (*, *) '  QFunction source       [-g] : header'
    end if
    if (help == 1) then
      if (test == 0) then
        write (*, *) 'Test/quiet mode is OFF (use -t to enable)'
      else
        write (*, *) 'Test/quiet mode is ON'
      end if
    end if
! LCOV_EXCL_STOP
    end if

! Select appropriate backend and logical device based on the (-ceed) command line argument
    call ceedinit(trim(ceed_spec)//char(0), ceed, err)

! Construct the mesh and solution bases
    call ceedbasiscreatetensorh1lagrange(ceed, fe_dim, num_comp_x, mesh_degree + 1, num_qpts, ceed_gauss, mesh_basis,&
             &err)
    call ceedbasiscreatetensorh1lagrange(ceed, fe_dim, 1, sol_degree + 1, num_qpts, ceed_gauss, sol_basis, err)

! Determine the mesh size based on the given approximate problem size
    call getcartesianmeshsize(fe_dim, sol_degree, prob_size, num_xyz)
    if (test == 0) then
! LCOV_EXCL_START
    write (*, '(A16, I8)', advance='no') 'Mesh size: nx = ', num_xyz(1)
    if (num_comp_x > 1) then
      write (*, '(A7, I8)', advance='no') ',  ny = ', num_xyz(2)
    end if
    if (num_comp_x > 2) then
      write (*, '(A7, I8)', advance='no') ',  nz = ', num_xyz(3)
    end if
    write (*, *)
! LCOV_EXCL_STOP
    endif

! Build CeedElemRestriction objects describing the mesh and solution discrete representation
    call buildcartesianrestriction(ceed, fe_dim, num_xyz, mesh_degree, num_comp_x, mesh_size, num_qpts,&
             &mesh_restriction, ceed_qfunction_none, err)
    call buildcartesianrestriction(ceed, fe_dim, num_xyz, sol_degree, 1, sol_size, num_qpts, sol_restriction,&
             &q_data_restriction, err)

    if (test == 0) then
! LCOV_EXCL_START
      write (*, *) 'Number of mesh nodes     : ', mesh_size/fe_dim
      write (*, *) 'Number of solution nodes : ', sol_size
! LCOV_EXCL_STOP
    end if

! Create a CeedVector with the mesh coordinates
! Apply a transformation to the mesh
    call ceedvectorcreate(ceed, mesh_size, mesh_coords, err)
    call setcartesianmeshcoords(fe_dim, num_xyz, mesh_degree, mesh_coords, exact_volume, err)

! Context data to be passed to the 'build_mass' QFunction
    build_ctx_data(1) = fe_dim
    build_ctx_data(2) = num_comp_x
    call ceedqfunctioncontextcreate(ceed, build_ctx, err)
! Note: The context technically only takes arrays of double precision values, but we can pass arrays of ints of the same length
    offset = 0
    call ceedqfunctioncontextsetdata(build_ctx, ceed_mem_host, ceed_use_pointer, build_ctx_size, build_ctx_data,&
             &offset, err)

! Create the QFunction that builds the mass operator (i.e. computes its quadrature data) and set its context data
    if (gallery == 1) then
      select case (fe_dim)
        case (1)
          call ceedqfunctioncreateinteriorbyname(ceed, 'Mass1DBuild', qf_build, err)

        case (2)
          call ceedqfunctioncreateinteriorbyname(ceed, 'Mass2DBuild', qf_build, err)

        case (3)
          call ceedqfunctioncreateinteriorbyname(ceed, 'Mass3DBuild', qf_build, err)
      end select
    else
      call ceedqfunctioncreateinterior(ceed, 1, build_mass,&
               &SOURCE_DIR&
               &//'ex1-volume-f-c.h:build_mass'//char(0), qf_build, err)
      call ceedqfunctionaddinput(qf_build, 'dx', num_comp_x * fe_dim, ceed_eval_grad, err)
      call ceedqfunctionaddinput(qf_build, 'weights', 1, ceed_eval_weight, err)
      call ceedqfunctionaddoutput(qf_build, 'qdata', 1, ceed_eval_none, err)
      call ceedqfunctionsetcontext(qf_build, build_ctx, err)
    end if

! Create the operator that builds the quadrature data for the mass operator
    call ceedoperatorcreate(ceed, qf_build, ceed_qfunction_none, ceed_qfunction_none, op_build, err)
    call ceedoperatorsetfield(op_build, 'dx', mesh_restriction, mesh_basis, ceed_vector_active, err)
    call ceedoperatorsetfield(op_build, 'weights', ceed_elemrestriction_none, mesh_basis, ceed_vector_none, err)
    call ceedoperatorsetfield(op_build, 'qdata', q_data_restriction, ceed_basis_none, ceed_vector_active, err)

! Compute the quadrature data for the mass operator
    num_elem  = 1
    elem_qpts = num_qpts**fe_dim
    do i = 1, fe_dim
      num_elem = num_elem * num_xyz(i)
    end do
    call ceedvectorcreate(ceed, num_elem * elem_qpts, q_data, err)
    call ceedoperatorapply(op_build, mesh_coords, q_data, ceed_request_immediate, err)

! Create the QFunction that defines the action of the mass operator
    if (gallery == 1) then
      call ceedqfunctioncreateinteriorbyname(ceed, 'MassApply', qf_apply, err)
    else
      call ceedqfunctioncreateinterior(ceed, 1, apply_mass,&
               &SOURCE_DIR&
               &//'ex1-volume-f-c.h:apply_mass'//char(0), qf_apply, err)
      call ceedqfunctionaddinput(qf_apply, 'u', 1, ceed_eval_interp, err)
      call ceedqfunctionaddinput(qf_apply, 'qdata', 1, ceed_eval_none, err)
      call ceedqfunctionaddoutput(qf_apply, 'v', 1, ceed_eval_interp, err)
    end if

! Create the mass operator
    call ceedoperatorcreate(ceed, qf_apply, ceed_qfunction_none, ceed_qfunction_none, op_apply, err)
    call ceedoperatorsetfield(op_apply, 'u', sol_restriction, sol_basis, ceed_vector_active, err)
    call ceedoperatorsetfield(op_apply, 'qdata', q_data_restriction, ceed_basis_none, q_data, err)
    call ceedoperatorsetfield(op_apply, 'v', sol_restriction, sol_basis, ceed_vector_active, err)

! Create auxiliary solution-size vectors
    allocate (u_array(sol_size))
    allocate (v_array(sol_size))

    call ceedvectorcreate(ceed, sol_size, u, err)
    offset = 0
    call ceedvectorsetarray(u, ceed_mem_host, ceed_use_pointer, u_array, offset, err)
    call ceedvectorcreate(ceed, sol_size, v, err)
    offset = 0
    call ceedvectorsetarray(v, ceed_mem_host, ceed_use_pointer, v_array, offset, err)

! Initialize 'u' with ones
    call ceedvectorsetvalue(u, 1.d0, err)

! Compute the mesh volume using the mass operator: volume = 1^T \cdot M \cdot 1
    call ceedoperatorapply(op_apply, u, v, ceed_request_immediate, err)

! Benchmark runs
    if (test /= 1 .AND. benchmark /= 0) then
! LCOV_EXCL_START
      write (*, *) ' Executing ', benchmark, ' benchmarking runs...'
! LCOV_EXCL_STOP
    end if
    do i = 1, benchmark
! LCOV_EXCL_START
      call ceedoperatorapply(op_apply, u, v, ceed_request_immediate, err)
! LCOV_EXCL_STOP
    end do

! Compute and print the sum of the entries of 'v' giving the mesh volume
    computed_volume = 0.d0

    call ceedvectorgetarrayread(v, ceed_mem_host, v_array, offset, err)
    do i = 1, sol_size
      computed_volume = computed_volume + v_array(offset + i)
    end do
    call ceedvectorrestorearrayread(v, v_array, offset, err)

    if (test /= 1) then
! LCOV_EXCL_START
      write (*, *) ' done.'
      write (*, *) 'Exact mesh volume    :', exact_volume
      write (*, *) 'Computed mesh volume :', computed_volume
      write (*, *) 'Volume error         :', (exact_volume - computed_volume)
! LCOV_EXCL_STOP
    else
      if (fe_dim == 1) then
        if (abs(exact_volume - computed_volume) > 200.d0 * 1e-15) then
! LCOV_EXCL_START
          write (*, *) 'Volume error : ', (exact_volume - computed_volume)
! LCOV_EXCL_STOP
        end if
      else
        if (abs(exact_volume - computed_volume) > 1e-5) then
! LCOV_EXCL_START
          write (*, *) 'Volume error : ', (exact_volume - computed_volume)
! LCOV_EXCL_STOP
        end if
      end if
    end if

! Free dynamically allocated memory
    call ceedvectordestroy(mesh_coords, err)
    call ceedvectordestroy(q_data, err)
    call ceedvectordestroy(u, err)
    call ceedvectordestroy(v, err)
    deallocate (u_array)
    deallocate (v_array)
    call ceedbasisdestroy(sol_basis, err)
    call ceedbasisdestroy(mesh_basis, err)
    call ceedqfunctioncontextdestroy(build_ctx, err)
    call ceedqfunctiondestroy(qf_build, err)
    call ceedqfunctiondestroy(qf_apply, err)
    call ceedoperatordestroy(op_build, err)
    call ceedoperatordestroy(op_apply, err)
    call ceeddestroy(ceed, err)
end
!-----------------------------------------------------------------------
