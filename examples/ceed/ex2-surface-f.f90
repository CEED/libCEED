! Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and other CEED contributors.
! All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
!
! SPDX-License-Identifier: BSD-2-Clause
!
! This file is part of CEED:  http:Cgithub.com/ceed

!                             libCEED Example 2
!
! This example illustrates a simple usage of libCEED to compute the surface
! area of a 3D body using matrix-free application of a diffusion operator.
! Arbitrary mesh and solution degrees in 1D, 2D and 3D are supported from
! the same code.
!
! The example has no dependencies, and is designed to be self-contained.
! For additional examples that use external discretization libraries
! (MFEM, PETSc, etc.) see the subdirectories in libceed/examples.
!
! All libCEED objects use a Ceed device object constructed based on a
! command line argument (-ceed).
!
! Build with:
!
!     make ex2-surface [CEED_DIR=</path/to/libceed>]
!
! Sample runs:
!
!     ./ex2-surface-f
!     ./ex2-surface-f -ceed /cpu/self
!     ./ex2-surface-f -ceed /gpu/cuda
!
! Test in 1D-3D
! TESTARGS(name = "1D User QFunction") -ceed {ceed_resource} -d 1 -t
! TESTARGS(name = "2D User QFunction") -ceed {ceed_resource} -d 2 -t
! TESTARGS(name = "3D User QFunction") -ceed {ceed_resource} -d 3 -t
! TESTARGS(name = "1D Gallery QFunction") -ceed {ceed_resource} -d 1 -t -g
! TESTARGS(name = "2D Gallery QFunction") -ceed {ceed_resource} -d 2 -t -g
! TESTARGS(name = "3D Gallery QFunction") -ceed {ceed_resource} -d 3 -t -g

!> @file
!> libCEED example using diffusion operator to compute surface area

    include 'ex2-surface-f.h'

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

    if (restriction /= ceed_qfunction_none) then
      call ceedelemrestrictioncreate(ceed, num_elem, num_nodes, num_comp, scalar_size, num_comp * scalar_size,&
               &ceed_mem_host, ceed_copy_values, elem_nodes, restriction, err)
    end if
    if (q_data_restriction /= ceed_qfunction_none) then
      call ceedelemrestrictioncreatestrided(ceed, num_elem, elem_qpts, num_comp, num_comp * elem_qpts * num_elem,&
               &ceed_strides_backend, q_data_restriction, err)
    end if
    deallocate (elem_nodes)
end

!-----------------------------------------------------------------------
subroutine transformmeshcoords(fe_dim, mesh_size, coords, exact_surface_area, err)
    implicit none

    integer fe_dim
    integer mesh_size
    real*8 coords(mesh_size)
    real*8 exact_surface_area
    real*8 m_pi
    parameter(m_pi = 3.14159265358979323846d0)
    integer err

    integer i


    select case (fe_dim)
    case (1)
      exact_surface_area = 2.d0
    case (2)
      exact_surface_area = 4.d0
    case (3)
      exact_surface_area = 6.d0
    end select

    do i = 1, mesh_size
      coords(i) = 0.5d0 + (1.d0/sqrt(3.d0)) * sin((2.d0/3.d0) * m_pi * (coords(i) - 0.5d0))
    end do
end

!-----------------------------------------------------------------------
subroutine setcartesianmeshcoords(fe_dim, num_xyz, mesh_degree, mesh_coords, exact_surface_area, err)
    implicit none
    include 'ceed/fortran.h'

    integer fe_dim
    integer num_xyz(3)
    integer mesh_degree
    integer mesh_coords
    real*8 exact_surface_area
    integer err

    integer p
    integer scalar_size
    integer coords_size
    integer r_nodes
    integer d_1d
    integer nd(3)
    real*8, dimension (:), allocatable :: nodes, qpts
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
        d_1d = mod(r_nodes, nd(j))
        coords(scalar_size * (j - 1) + i) = ((d_1d/(p - 1)) + nodes(mod(d_1d, p - 1) + 1))/num_xyz(j)
        r_nodes = r_nodes/nd(j)
      end do
    end do
    deallocate(nodes)

    call transformmeshcoords(fe_dim, coords_size, coords, exact_surface_area, err)

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
    integer prob_size, mesh_size, sol_size, q_data_size
    integer help, test, gallery, benchmark
    integer i, d, num_args, err
    character arg*32, arg_value*32
    real*8 exact_surface_area, computed_surface_area
    real*8 tol

    integer ceed
    real*8, dimension (:), allocatable :: u_array, v_array
    real*8, dimension (:), allocatable :: x_array
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
    character gallery_name*25

    external build_diff, apply_diff

! Initial values
    ceed_spec   = '/cpu/self'
    fe_dim      = 3
    num_comp_x  = 3
    mesh_degree = 4
    sol_degree  = 4
    num_qpts    = sol_degree + 2
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

        case ('-c', '-ceed')
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


    mesh_degree = max(mesh_degree, sol_degree)
    sol_degree  = mesh_degree
    num_qpts    = sol_degree + 2

    if (prob_size < 0) then
      if (test == 1) then
        prob_size = 16 * 16 * fe_dim * fe_dim
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
        stop
    end if
      write (*, *)
! LCOV_EXCL_STOP
    end if

! Select appropriate backend and logical device based on the (-ceed) command line argument
    call ceedinit(trim(ceed_spec)//char(0), ceed, err)

! Construct the mesh and solution bases
    call ceedbasiscreatetensorh1lagrange(ceed, fe_dim, num_comp_x, mesh_degree + 1, num_qpts, ceed_gauss, mesh_basis, err)
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
    end if

! Build CeedElemRestriction objects describing the mesh and solution discrete representations
! Note: ex2 makes THREE calls to buildcartesianrestriction (unlike ex1 which makes two)
    call buildcartesianrestriction(ceed, fe_dim, num_xyz, mesh_degree, num_comp_x, mesh_size, num_qpts,&
             &mesh_restriction, ceed_qfunction_none, err)
    call buildcartesianrestriction(ceed, fe_dim, num_xyz, sol_degree, fe_dim*(fe_dim+1)/2, sol_size, num_qpts,&
             &ceed_qfunction_none, q_data_restriction, err)
    call buildcartesianrestriction(ceed, fe_dim, num_xyz, sol_degree, 1, sol_size, num_qpts,&
             &sol_restriction, ceed_qfunction_none, err)

    if (test == 0) then
! LCOV_EXCL_START
      write (*, *) 'Number of mesh nodes     : ', mesh_size/fe_dim
      write (*, *) 'Number of solution nodes : ', sol_size
! LCOV_EXCL_STOP
    end if

! Create a CeedVector with the mesh coordinates
    call ceedvectorcreate(ceed, mesh_size, mesh_coords, err)
    call setcartesianmeshcoords(fe_dim, num_xyz, mesh_degree, mesh_coords, exact_surface_area, err)

! Context data to be passed to the 'build_diff' QFunction
    build_ctx_data(1) = fe_dim
    build_ctx_data(2) = fe_dim
    call ceedqfunctioncontextcreate(ceed, build_ctx, err)
    offset = 0
    call ceedqfunctioncontextsetdata(build_ctx, ceed_mem_host, ceed_use_pointer, build_ctx_size, build_ctx_data,&
             &offset, err)

! Create the QFunction that builds the diff operator (i.e. computes its quadrature data) and set its context data
    if (gallery == 1) then
! LCOV_EXCL_START
      select case (fe_dim)
        case (1)
          call ceedqfunctioncreateinteriorbyname(ceed, 'Poisson1DBuild', qf_build, err)
        case (2)
          call ceedqfunctioncreateinteriorbyname(ceed, 'Poisson2DBuild', qf_build, err)
        case (3)
          call ceedqfunctioncreateinteriorbyname(ceed, 'Poisson3DBuild', qf_build, err)
      end select
! LCOV_EXCL_STOP
    else
      call ceedqfunctioncreateinterior(ceed, 1, build_diff,&
               &SOURCE_DIR&
               &//'ex2-surface-f-c.h:build_diff'//char(0), qf_build, err)
      call ceedqfunctionaddinput(qf_build, 'dx', num_comp_x * fe_dim, ceed_eval_grad, err)
      call ceedqfunctionaddinput(qf_build, 'weights', 1, ceed_eval_weight, err)
      call ceedqfunctionaddoutput(qf_build, 'qdata', fe_dim * (fe_dim + 1) / 2, ceed_eval_none, err)
      call ceedqfunctionsetcontext(qf_build, build_ctx, err)
    end if

! Create the operator that builds the quadrature data for the diff operator
    call ceedoperatorcreate(ceed, qf_build, ceed_qfunction_none, ceed_qfunction_none, op_build, err)
    call ceedoperatorsetfield(op_build, 'dx', mesh_restriction, mesh_basis, ceed_vector_active, err)
    call ceedoperatorsetfield(op_build, 'weights', ceed_elemrestriction_none, mesh_basis, ceed_vector_none, err)
    call ceedoperatorsetfield(op_build, 'qdata', q_data_restriction, ceed_basis_none, ceed_vector_active, err)

! Compute the quadrature data for the diff operator
    num_elem  = 1
    elem_qpts = num_qpts**fe_dim
    do i = 1, fe_dim
      num_elem = num_elem * num_xyz(i)
    end do
    q_data_size = num_elem * elem_qpts * fe_dim * (fe_dim + 1) / 2
    call ceedvectorcreate(ceed, q_data_size, q_data, err)
    call ceedoperatorapply(op_build, mesh_coords, q_data, ceed_request_immediate, err)

! Create the QFunction that defines the action of the diff operator
    if (gallery == 1) then
! LCOV_EXCL_START
      select case (fe_dim)
        case (1)
          call ceedqfunctioncreateinteriorbyname(ceed, 'Poisson1DApply', qf_apply, err)
        case (2)
          call ceedqfunctioncreateinteriorbyname(ceed, 'Poisson2DApply', qf_apply, err)
        case (3)
          call ceedqfunctioncreateinteriorbyname(ceed, 'Poisson3DApply', qf_apply, err)
      end select
! LCOV_EXCL_STOP
    else
      call ceedqfunctioncreateinterior(ceed, 1, apply_diff,&
               &SOURCE_DIR&
               &//'ex2-surface-f-c.h:apply_diff'//char(0), qf_apply, err)
      call ceedqfunctionaddinput(qf_apply, 'du', fe_dim, ceed_eval_grad, err)
      call ceedqfunctionaddinput(qf_apply, 'qdata', fe_dim * (fe_dim + 1) / 2, ceed_eval_none, err)
      call ceedqfunctionaddoutput(qf_apply, 'dv', fe_dim, ceed_eval_grad, err)
      call ceedqfunctionsetcontext(qf_apply, build_ctx, err)
    end if

! Create the diff operator
    call ceedoperatorcreate(ceed, qf_apply, ceed_qfunction_none, ceed_qfunction_none, op_apply, err)
    call ceedoperatorsetfield(op_apply, 'du', sol_restriction, sol_basis, ceed_vector_active, err)
    call ceedoperatorsetfield(op_apply, 'qdata', q_data_restriction, ceed_basis_none, q_data, err)
    call ceedoperatorsetfield(op_apply, 'dv', sol_restriction, sol_basis, ceed_vector_active, err)

! Create auxiliary solution-size vectors
    allocate (u_array(sol_size))
    allocate (v_array(sol_size))
    allocate (x_array(mesh_size))

    call ceedvectorcreate(ceed, sol_size, u, err)
    call ceedvectorcreate(ceed, sol_size, v, err)

! Initialize 'u' with the sum of  coordinates, x + y + z
    offset = 0
    call ceedvectorgetarrayread(mesh_coords, ceed_mem_host, x_array, offset, err)
    do i = 1, sol_size
      u_array(i) = 0.d0
      do d = 1, fe_dim
        u_array(i) = u_array(i) + x_array(offset + i + (d - 1) * sol_size)
      end do
    end do
    call ceedvectorrestorearrayread(mesh_coords, x_array, offset, err)
    deallocate (x_array)

    offset = 0
    call ceedvectorsetarray(u, ceed_mem_host, ceed_use_pointer, u_array, offset, err)
    offset = 0
    call ceedvectorsetarray(v, ceed_mem_host, ceed_use_pointer, v_array, offset, err)

! Compute the mesh surface area using the diffusion operator:
!   surface_area = 1^T * |K * u|
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

! Compute and print the sum of abs of entries of 'v' giving the mesh surface area

    computed_surface_area = 0.d0

    offset = 0
    call ceedvectorgetarrayread(v, ceed_mem_host, v_array, offset, err)
    do i = 1, sol_size
      computed_surface_area = computed_surface_area + abs(v_array(offset + i))
    end do
    call ceedvectorrestorearrayread(v, v_array, offset, err)

    if (test /= 1) then
! LCOV_EXCL_START
      write (*, *) ' done.'
      write (*, *) 'Exact mesh surface area    :', exact_surface_area
      write (*, *) 'Computed mesh surface area :', computed_surface_area
      write (*, *) 'Surface area error         :', (computed_surface_area - exact_surface_area)
! LCOV_EXCL_STOP
    else
      if (fe_dim == 1) then
        tol = 10000.d0 * 1e-15
      else
        tol = 1e-1
      end if
      if (abs(computed_surface_area - exact_surface_area) > tol) then
! LCOV_EXCL_START
        write (*, *) 'Surface area error : ', (computed_surface_area - exact_surface_area)
! LCOV_EXCL_STOP
      end if
    end if

! Free dynamically allocated memory
    call ceedvectordestroy(u, err)
    call ceedvectordestroy(v, err)
    call ceedvectordestroy(q_data, err)
    call ceedvectordestroy(mesh_coords, err)
    call ceedoperatordestroy(op_apply, err)
    call ceedqfunctiondestroy(qf_apply, err)
    call ceedqfunctioncontextdestroy(build_ctx, err)
    call ceedoperatordestroy(op_build, err)
    call ceedqfunctiondestroy(qf_build, err)
    call ceedelemrestrictiondestroy(sol_restriction, err)
    call ceedelemrestrictiondestroy(mesh_restriction, err)
    call ceedelemrestrictiondestroy(q_data_restriction, err)
    call ceedbasisdestroy(sol_basis, err)
    call ceedbasisdestroy(mesh_basis, err)
    call ceeddestroy(ceed, err)
    deallocate (u_array)
    deallocate (v_array)
end
!-----------------------------------------------------------------------