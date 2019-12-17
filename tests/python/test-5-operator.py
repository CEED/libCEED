# Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
# the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
# reserved. See files LICENSE and NOTICE for details.
#
# This file is part of CEED, a collection of benchmarks, miniapps, software
# libraries and APIs for efficient high-order finite element and spectral
# element discretizations for exascale applications. For more information and
# source code availability see http://github.com/ceed.
#
# The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
# a collaborative effort of two U.S. Department of Energy organizations (Office
# of Science and the National Nuclear Security Administration) responsible for
# the planning and preparation of a capable exascale ecosystem, including
# software, applications, hardware, advanced system engineering and early
# testbed platforms, in support of the nation's exascale computing imperative.

# @file
# Test Ceed Operator functionality

import os
import glob
import ctypes
import libceed
import numpy as np
import buildmats as bm

#-------------------------------------------------------------------------------
# Utility
#-------------------------------------------------------------------------------
def load_qfs_so():
  # Filename
  file_dir = os.path.dirname(os.path.abspath(__file__))

  # Rename, if needed
  qfs_so = glob.glob("libceed_qfunctions.*.so")
  if len(qfs_so) > 0:
    os.rename(qfs_so[0], file_dir + "/qfs.so")

  # Load library
  qfs = ctypes.cdll.LoadLibrary('./qfs.so')

  return qfs

#-------------------------------------------------------------------------------
# Test creation, action, and destruction for mass matrix operator
#-------------------------------------------------------------------------------
def test_500(ceed_resource):
  ceed = libceed.Ceed(ceed_resource)

  nelem = 15
  p = 5
  q = 8
  nx = nelem + 1
  nu = nelem*(p-1) + 1

  # Vectors
  x = ceed.Vector(nx)
  x_array = np.zeros(nx)
  for i in range(nx):
    x_array[i] = i / (nx - 1.0)
  x.set_array(x_array, cmode=libceed.USE_POINTER)

  qdata = ceed.Vector(nelem*q)
  u = ceed.Vector(nu)
  v = ceed.Vector(nu)

  # Restrictions
  indx = np.zeros(nx*2, dtype="int32")
  for i in range(nx):
    indx[2*i+0] = i
    indx[2*i+1] = i+1
  rx = ceed.ElemRestriction(nelem, 2, nx, 1, indx, cmode=libceed.USE_POINTER)
  rxi = ceed.IdentityElemRestriction(nelem, 2, nelem*2, 1)

  indu = np.zeros(nelem*p, dtype="int32")
  for i in range(nelem):
    for j in range(p):
      indu[p*i+j] = i*(p-1) + j
  ru = ceed.ElemRestriction(nelem, p, nu, 1, indu, cmode=libceed.USE_POINTER)
  rui = ceed.IdentityElemRestriction(nelem, q, q*nelem, 1)

  # Bases
  bx = ceed.BasisTensorH1Lagrange(1, 1, 2, q, libceed.GAUSS)
  bu = ceed.BasisTensorH1Lagrange(1, 1, p, q, libceed.GAUSS)

  # QFunctions
  file_dir = os.path.dirname(os.path.abspath(__file__))
  qfs = load_qfs_so()

  qf_setup = ceed.QFunction(1, qfs.setup_mass,
                            os.path.join(file_dir, "test-qfunctions.h:setup_mass"))
  qf_setup.add_input("weights", 1, libceed.EVAL_WEIGHT)
  qf_setup.add_input("dx", 1, libceed.EVAL_GRAD)
  qf_setup.add_output("rho", 1, libceed.EVAL_NONE)

  qf_mass = ceed.QFunction(1, qfs.apply_mass,
                           os.path.join(file_dir, "test-qfunctions.h:apply_mass"))
  qf_mass.add_input("rho", 1, libceed.EVAL_NONE)
  qf_mass.add_input("u", 1, libceed.EVAL_INTERP)
  qf_mass.add_output("v", 1, libceed.EVAL_INTERP)

  # Operators
  op_setup = ceed.Operator(qf_setup)
  op_setup.set_field("weights", rxi, bx, libceed.VECTOR_NONE)
  op_setup.set_field("dx", rx, bx, libceed.VECTOR_ACTIVE)
  op_setup.set_field("rho", rui, libceed.BASIS_COLLOCATED,
                     libceed.VECTOR_ACTIVE)

  op_mass = ceed.Operator(qf_mass)
  op_mass.set_field("rho", rui, libceed.BASIS_COLLOCATED, qdata)
  op_mass.set_field("u", ru, bu, libceed.VECTOR_ACTIVE)
  op_mass.set_field("v", ru, bu, libceed.VECTOR_ACTIVE)

  # Setup
  op_setup.apply(x, qdata)

  # Apply mass matrix
  u.set_value(0)
  op_mass.apply(u, v)

  # Check
  v_array = v.get_array_read()
  for i in range(q):
    assert abs(v_array[i]) < 1E-14

  v.restore_array_read()

#-------------------------------------------------------------------------------
# Test creation, action, and destruction for mass matrix operator
#-------------------------------------------------------------------------------
def test_501(ceed_resource):
  ceed = libceed.Ceed(ceed_resource)

  nelem = 15
  p = 5
  q = 8
  nx = nelem + 1
  nu = nelem*(p-1) + 1

  # Vectors
  x = ceed.Vector(nx)
  x_array = np.zeros(nx)
  for i in range(nx):
    x_array[i] = i / (nx - 1.0)
  x.set_array(x_array, cmode=libceed.USE_POINTER)

  qdata = ceed.Vector(nelem*q)
  u = ceed.Vector(nu)
  v = ceed.Vector(nu)

  # Restrictions
  indx = np.zeros(nx*2, dtype="int32")
  for i in range(nx):
    indx[2*i+0] = i
    indx[2*i+1] = i+1
  rx = ceed.ElemRestriction(nelem, 2, nx, 1, indx, cmode=libceed.USE_POINTER)
  rxi = ceed.IdentityElemRestriction(nelem, 2, nelem*2, 1)

  indu = np.zeros(nelem*p, dtype="int32")
  for i in range(nelem):
    for j in range(p):
      indu[p*i+j] = i*(p-1) + j
  ru = ceed.ElemRestriction(nelem, p, nu, 1, indu, cmode=libceed.USE_POINTER)
  rui = ceed.IdentityElemRestriction(nelem, q, q*nelem, 1)

  # Bases
  bx = ceed.BasisTensorH1Lagrange(1, 1, 2, q, libceed.GAUSS)
  bu = ceed.BasisTensorH1Lagrange(1, 1, p, q, libceed.GAUSS)

  # QFunctions
  file_dir = os.path.dirname(os.path.abspath(__file__))
  qfs = load_qfs_so()

  qf_setup = ceed.QFunction(1, qfs.setup_mass,
                            os.path.join(file_dir, "test-qfunctions.h:setup_mass"))
  qf_setup.add_input("weights", 1, libceed.EVAL_WEIGHT)
  qf_setup.add_input("dx", 1, libceed.EVAL_GRAD)
  qf_setup.add_output("rho", 1, libceed.EVAL_NONE)

  qf_mass = ceed.QFunction(1, qfs.apply_mass,
                           os.path.join(file_dir, "test-qfunctions.h:apply_mass"))
  qf_mass.add_input("rho", 1, libceed.EVAL_NONE)
  qf_mass.add_input("u", 1, libceed.EVAL_INTERP)
  qf_mass.add_output("v", 1, libceed.EVAL_INTERP)

  # Operators
  op_setup = ceed.Operator(qf_setup)
  op_setup.set_field("weights", rxi, bx, libceed.VECTOR_NONE)
  op_setup.set_field("dx", rx, bx, libceed.VECTOR_ACTIVE)
  op_setup.set_field("rho", rui, libceed.BASIS_COLLOCATED,
                     libceed.VECTOR_ACTIVE)

  op_mass = ceed.Operator(qf_mass)
  op_mass.set_field("rho", rui, libceed.BASIS_COLLOCATED, qdata)
  op_mass.set_field("u", ru, bu, libceed.VECTOR_ACTIVE)
  op_mass.set_field("v", ru, bu, libceed.VECTOR_ACTIVE)

  # Setup
  op_setup.apply(x, qdata)

  # Apply mass matrix
  u.set_value(1.)
  op_mass.apply(u, v)

  # Check
  v_array = v.get_array_read()
  total = 0.0
  for i in range(nu):
    total = total + v_array[i]
  assert abs(total - 1.0) < 1E-14

  v.restore_array_read()

#-------------------------------------------------------------------------------
# Test creation, action, and destruction for mass matrix operator
#-------------------------------------------------------------------------------
def test_502(ceed_resource):
  ceed = libceed.Ceed(ceed_resource)

  nelem = 15
  p = 5
  q = 8
  nx = nelem + 1
  nu = nelem*(p-1) + 1

  # Vectors
  x = ceed.Vector(nx)
  x_array = np.zeros(nx)
  for i in range(nx):
    x_array[i] = i / (nx - 1.0)
  x.set_array(x_array, cmode=libceed.USE_POINTER)

  qdata = ceed.Vector(nelem*q)
  u = ceed.Vector(2*nu)
  v = ceed.Vector(2*nu)

  # Restrictions
  indx = np.zeros(nx*2, dtype="int32")
  for i in range(nx):
    indx[2*i+0] = i
    indx[2*i+1] = i+1
  rx = ceed.ElemRestriction(nelem, 2, nx, 1, indx, cmode=libceed.USE_POINTER)
  rxi = ceed.IdentityElemRestriction(nelem, 2, nelem*2, 1)

  indu = np.zeros(nelem*p, dtype="int32")
  for i in range(nelem):
    for j in range(p):
      indu[p*i+j] = i*(p-1) + j
  ru = ceed.ElemRestriction(nelem, p, nu, 2, indu, cmode=libceed.USE_POINTER)
  rui = ceed.IdentityElemRestriction(nelem, q, q*nelem, 1)

  # Bases
  bx = ceed.BasisTensorH1Lagrange(1, 1, 2, q, libceed.GAUSS)
  bu = ceed.BasisTensorH1Lagrange(1, 2, p, q, libceed.GAUSS)

  # QFunctions
  file_dir = os.path.dirname(os.path.abspath(__file__))
  qfs = load_qfs_so()

  qf_setup = ceed.QFunction(1, qfs.setup_mass,
                            os.path.join(file_dir, "test-qfunctions.h:setup_mass"))
  qf_setup.add_input("weights", 1, libceed.EVAL_WEIGHT)
  qf_setup.add_input("dx", 1, libceed.EVAL_GRAD)
  qf_setup.add_output("rho", 1, libceed.EVAL_NONE)

  qf_mass = ceed.QFunction(1, qfs.apply_mass_two,
                           os.path.join(file_dir, "test-qfunctions.h:apply_mass_two"))
  qf_mass.add_input("rho", 1, libceed.EVAL_NONE)
  qf_mass.add_input("u", 2, libceed.EVAL_INTERP)
  qf_mass.add_output("v", 2, libceed.EVAL_INTERP)

  # Operators
  op_setup = ceed.Operator(qf_setup)
  op_setup.set_field("weights", rxi, bx, libceed.VECTOR_NONE)
  op_setup.set_field("dx", rx, bx, libceed.VECTOR_ACTIVE)
  op_setup.set_field("rho", rui, libceed.BASIS_COLLOCATED,
                     libceed.VECTOR_ACTIVE)

  op_mass = ceed.Operator(qf_mass)
  op_mass.set_field("rho", rui, libceed.BASIS_COLLOCATED, qdata)
  op_mass.set_field("u", ru, bu, libceed.VECTOR_ACTIVE, lmode=libceed.TRANSPOSE)
  op_mass.set_field("v", ru, bu, libceed.VECTOR_ACTIVE, lmode=libceed.TRANSPOSE)

  # Setup
  op_setup.apply(x, qdata)

  # Apply mass matrix
  u_array = u.get_array()
  for i in range(nu):
    u_array[2*i] = 1.
    u_array[2*i+1] = 2.
  u.restore_array()
  op_mass.apply(u, v)

  # Check
  v_array = v.get_array_read()
  total_1 = 0.0
  total_2 = 0.0
  for i in range(nu):
    total_1 = total_1 + v_array[2*i]
    total_2 = total_2 + v_array[2*i+1]
  assert abs(total_1 - 1.0) < 1E-13
  assert abs(total_2 - 2.0) < 1E-13

  v.restore_array_read()

#-------------------------------------------------------------------------------
# Test creation, action, and destruction for mass matrix operator with passive
#   inputs and outputs
#-------------------------------------------------------------------------------
def test_503(ceed_resource):
  ceed = libceed.Ceed(ceed_resource)

  nelem = 15
  p = 5
  q = 8
  nx = nelem + 1
  nu = nelem*(p-1) + 1

  # Vectors
  x = ceed.Vector(nx)
  x_array = np.zeros(nx)
  for i in range(nx):
    x_array[i] = i / (nx - 1.0)
  x.set_array(x_array, cmode=libceed.USE_POINTER)

  qdata = ceed.Vector(nelem*q)
  u = ceed.Vector(nu)
  v = ceed.Vector(nu)

  # Restrictions
  indx = np.zeros(nx*2, dtype="int32")
  for i in range(nx):
    indx[2*i+0] = i
    indx[2*i+1] = i+1
  rx = ceed.ElemRestriction(nelem, 2, nx, 1, indx, cmode=libceed.USE_POINTER)
  rxi = ceed.IdentityElemRestriction(nelem, 2, nelem*2, 1)

  indu = np.zeros(nelem*p, dtype="int32")
  for i in range(nelem):
    for j in range(p):
      indu[p*i+j] = i*(p-1) + j
  ru = ceed.ElemRestriction(nelem, p, nu, 1, indu, cmode=libceed.USE_POINTER)
  rui = ceed.IdentityElemRestriction(nelem, q, q*nelem, 1)

  # Bases
  bx = ceed.BasisTensorH1Lagrange(1, 1, 2, q, libceed.GAUSS)
  bu = ceed.BasisTensorH1Lagrange(1, 1, p, q, libceed.GAUSS)

  # QFunctions
  file_dir = os.path.dirname(os.path.abspath(__file__))
  qfs = load_qfs_so()

  qf_setup = ceed.QFunction(1, qfs.setup_mass,
                            os.path.join(file_dir, "test-qfunctions.h:setup_mass"))
  qf_setup.add_input("weights", 1, libceed.EVAL_WEIGHT)
  qf_setup.add_input("dx", 1, libceed.EVAL_GRAD)
  qf_setup.add_output("rho", 1, libceed.EVAL_NONE)

  qf_mass = ceed.QFunction(1, qfs.apply_mass,
                           os.path.join(file_dir, "test-qfunctions.h:apply_mass"))
  qf_mass.add_input("rho", 1, libceed.EVAL_NONE)
  qf_mass.add_input("u", 1, libceed.EVAL_INTERP)
  qf_mass.add_output("v", 1, libceed.EVAL_INTERP)

  # Operators
  op_setup = ceed.Operator(qf_setup)
  op_setup.set_field("weights", rxi, bx, libceed.VECTOR_NONE)
  op_setup.set_field("dx", rx, bx, libceed.VECTOR_ACTIVE)
  op_setup.set_field("rho", rui, libceed.BASIS_COLLOCATED,
                     libceed.VECTOR_ACTIVE)

  op_mass = ceed.Operator(qf_mass)
  op_mass.set_field("rho", rui, libceed.BASIS_COLLOCATED, qdata)
  op_mass.set_field("u", ru, bu, u, lmode=libceed.TRANSPOSE)
  op_mass.set_field("v", ru, bu, v, lmode=libceed.TRANSPOSE)

  # Setup
  op_setup.apply(x, qdata)

  # Apply mass matrix
  u.set_value(1)
  op_mass.apply(libceed.VECTOR_NONE, libceed.VECTOR_NONE)

  # Check
  v_array = v.get_array_read()
  total = 0.0
  for i in range(nu):
    total = total + v_array[i]
  assert abs(total - 1.0) < 1E-13

  v.restore_array_read()

#-------------------------------------------------------------------------------
# Test viewing of mass matrix operator
#-------------------------------------------------------------------------------
def test_504(ceed_resource, capsys):
  ceed = libceed.Ceed(ceed_resource)

  nelem = 15
  p = 5
  q = 8
  nx = nelem + 1
  nu = nelem*(p-1) + 1

  # Vectors
  qdata = ceed.Vector(nelem*q)

  # Restrictions
  indx = np.zeros(nx*2, dtype="int32")
  for i in range(nx):
    indx[2*i+0] = i
    indx[2*i+1] = i+1
  rx = ceed.ElemRestriction(nelem, 2, nx, 1, indx, cmode=libceed.USE_POINTER)
  rxi = ceed.IdentityElemRestriction(nelem, 2, nelem*2, 1)

  indu = np.zeros(nelem*p, dtype="int32")
  for i in range(nelem):
    for j in range(p):
      indu[p*i+j] = i*(p-1) + j
  ru = ceed.ElemRestriction(nelem, p, nu, 1, indu, cmode=libceed.USE_POINTER)
  rui = ceed.IdentityElemRestriction(nelem, q, q*nelem, 1)

  # Bases
  bx = ceed.BasisTensorH1Lagrange(1, 1, 2, q, libceed.GAUSS)
  bu = ceed.BasisTensorH1Lagrange(1, 1, p, q, libceed.GAUSS)

  # QFunctions
  file_dir = os.path.dirname(os.path.abspath(__file__))
  qfs = load_qfs_so()

  qf_setup = ceed.QFunction(1, qfs.setup_mass,
                            os.path.join(file_dir, "test-qfunctions.h:setup_mass"))
  qf_setup.add_input("weights", 1, libceed.EVAL_WEIGHT)
  qf_setup.add_input("dx", 1, libceed.EVAL_GRAD)
  qf_setup.add_output("rho", 1, libceed.EVAL_NONE)

  qf_mass = ceed.QFunction(1, qfs.apply_mass,
                           os.path.join(file_dir, "test-qfunctions.h:apply_mass"))
  qf_mass.add_input("rho", 1, libceed.EVAL_NONE)
  qf_mass.add_input("u", 1, libceed.EVAL_INTERP)
  qf_mass.add_output("v", 1, libceed.EVAL_INTERP)

  # Operators
  op_setup = ceed.Operator(qf_setup)
  op_setup.set_field("weights", rxi, bx, libceed.VECTOR_NONE)
  op_setup.set_field("dx", rx, bx, libceed.VECTOR_ACTIVE)
  op_setup.set_field("rho", rui, libceed.BASIS_COLLOCATED,
                     libceed.VECTOR_ACTIVE)

  op_mass = ceed.Operator(qf_mass)
  op_mass.set_field("rho", rui, libceed.BASIS_COLLOCATED, qdata)
  op_mass.set_field("u", ru, bu, libceed.VECTOR_ACTIVE)
  op_mass.set_field("v", ru, bu, libceed.VECTOR_ACTIVE)

  # View
  print(op_setup)
  print(op_mass)

  stdout, stderr = capsys.readouterr()
  with open(os.path.abspath("./output/test_504.out")) as output_file:
    true_output = output_file.read()

  assert stdout == true_output

#-------------------------------------------------------------------------------
# Test CeedOperatorApplyAdd
#-------------------------------------------------------------------------------
def test_505(ceed_resource):
  ceed = libceed.Ceed(ceed_resource)

  nelem = 15
  p = 5
  q = 8
  nx = nelem + 1
  nu = nelem*(p-1) + 1

  # Vectors
  x = ceed.Vector(nx)
  x_array = np.zeros(nx)
  for i in range(nx):
    x_array[i] = i / (nx - 1.0)
  x.set_array(x_array, cmode=libceed.USE_POINTER)

  qdata = ceed.Vector(nelem*q)
  u = ceed.Vector(nu)
  v = ceed.Vector(nu)

  # Restrictions
  indx = np.zeros(nx*2, dtype="int32")
  for i in range(nx):
    indx[2*i+0] = i
    indx[2*i+1] = i+1
  rx = ceed.ElemRestriction(nelem, 2, nx, 1, indx, cmode=libceed.USE_POINTER)
  rxi = ceed.IdentityElemRestriction(nelem, 2, nelem*2, 1)

  indu = np.zeros(nelem*p, dtype="int32")
  for i in range(nelem):
    for j in range(p):
      indu[p*i+j] = i*(p-1) + j
  ru = ceed.ElemRestriction(nelem, p, nu, 1, indu, cmode=libceed.USE_POINTER)
  rui = ceed.IdentityElemRestriction(nelem, q, q*nelem, 1)

  # Bases
  bx = ceed.BasisTensorH1Lagrange(1, 1, 2, q, libceed.GAUSS)
  bu = ceed.BasisTensorH1Lagrange(1, 1, p, q, libceed.GAUSS)

  # QFunctions
  file_dir = os.path.dirname(os.path.abspath(__file__))
  qfs = load_qfs_so()

  qf_setup = ceed.QFunction(1, qfs.setup_mass,
                            os.path.join(file_dir, "test-qfunctions.h:setup_mass"))
  qf_setup.add_input("weights", 1, libceed.EVAL_WEIGHT)
  qf_setup.add_input("dx", 1, libceed.EVAL_GRAD)
  qf_setup.add_output("rho", 1, libceed.EVAL_NONE)

  qf_mass = ceed.QFunction(1, qfs.apply_mass,
                           os.path.join(file_dir, "test-qfunctions.h:apply_mass"))
  qf_mass.add_input("rho", 1, libceed.EVAL_NONE)
  qf_mass.add_input("u", 1, libceed.EVAL_INTERP)
  qf_mass.add_output("v", 1, libceed.EVAL_INTERP)

  # Operators
  op_setup = ceed.Operator(qf_setup)
  op_setup.set_field("weights", rxi, bx, libceed.VECTOR_NONE)
  op_setup.set_field("dx", rx, bx, libceed.VECTOR_ACTIVE)
  op_setup.set_field("rho", rui, libceed.BASIS_COLLOCATED,
                     libceed.VECTOR_ACTIVE)

  op_mass = ceed.Operator(qf_mass)
  op_mass.set_field("rho", rui, libceed.BASIS_COLLOCATED, qdata)
  op_mass.set_field("u", ru, bu, libceed.VECTOR_ACTIVE)
  op_mass.set_field("v", ru, bu, libceed.VECTOR_ACTIVE)

  # Setup
  op_setup.apply(x, qdata)

  # Apply mass matrix with v = 0
  u.set_value(1.)
  v.set_value(0.)
  op_mass.apply_add(u, v)

  # Check
  v_array = v.get_array_read()
  total = 0.0
  for i in range(nu):
    total = total + v_array[i]
  assert abs(total - 1.0) < 1E-14

  v.restore_array_read()

  # Apply mass matrix with v = 0
  v.set_value(1.)
  op_mass.apply_add(u, v)

  # Check
  v_array = v.get_array_read()
  total = -nu
  for i in range(nu):
    total = total + v_array[i]
  assert abs(total - 1.0) < 1E-10

  v.restore_array_read()

#-------------------------------------------------------------------------------
# Test creation, action, and destruction for mass matrix operator
#-------------------------------------------------------------------------------
def test_510(ceed_resource):
  ceed = libceed.Ceed(ceed_resource)

  nelem = 12
  dim = 2
  p = 6
  q = 4
  nx, ny = 3, 2
  ndofs = (nx*2+1)*(ny*2+1)
  nqpts = nelem*q

  # Vectors
  x = ceed.Vector(dim*ndofs)
  x_array = np.zeros(dim*ndofs)
  for i in range(ndofs):
    x_array[i] = (1. / (nx*2)) * (i % (nx*2+1))
    x_array[i+ndofs] = (1. / (ny*2)) * (i / (nx*2+1))
  x.set_array(x_array, cmode=libceed.USE_POINTER)

  qdata = ceed.Vector(nqpts)
  u = ceed.Vector(ndofs)
  v = ceed.Vector(ndofs)

  # Restrictions
  indx = np.zeros(nelem*p, dtype="int32")
  for i in range(nelem//2):
    col = i % nx;
    row = i // nx;
    offset = col*2 + row*(nx*2+1)*2

    indx[i*2*p+ 0] =  2 + offset
    indx[i*2*p+ 1] =  9 + offset
    indx[i*2*p+ 2] = 16 + offset
    indx[i*2*p+ 3] =  1 + offset
    indx[i*2*p+ 4] =  8 + offset
    indx[i*2*p+ 5] =  0 + offset

    indx[i*2*p+ 6] = 14 + offset
    indx[i*2*p+ 7] =  7 + offset
    indx[i*2*p+ 8] =  0 + offset
    indx[i*2*p+ 9] = 15 + offset
    indx[i*2*p+10] =  8 + offset
    indx[i*2*p+11] = 16 + offset

  rx = ceed.ElemRestriction(nelem, p, ndofs, dim, indx,
                            cmode=libceed.USE_POINTER)
  rxi = ceed.IdentityElemRestriction(nelem, p, nelem*p, dim)

  ru = ceed.ElemRestriction(nelem, p, ndofs, 1, indx, cmode=libceed.USE_POINTER)
  rui = ceed.IdentityElemRestriction(nelem, q, nqpts, 1)

  # Bases
  qref = np.empty(dim*q, dtype="float64")
  qweight = np.empty(q, dtype="float64")
  interp, grad = bm.buildmats(qref, qweight)

  bx = ceed.BasisH1(libceed.TRIANGLE, dim, p, q, interp, grad, qref, qweight)
  bu = ceed.BasisH1(libceed.TRIANGLE, 1, p, q, interp, grad, qref, qweight)

  # QFunctions
  file_dir = os.path.dirname(os.path.abspath(__file__))
  qfs = load_qfs_so()

  qf_setup = ceed.QFunction(1, qfs.setup_mass_2d,
                            os.path.join(file_dir, "test-qfunctions.h:setup_mass_2d"))
  qf_setup.add_input("weights", 1, libceed.EVAL_WEIGHT)
  qf_setup.add_input("dx", dim*dim, libceed.EVAL_GRAD)
  qf_setup.add_output("rho", 1, libceed.EVAL_NONE)

  qf_mass = ceed.QFunction(1, qfs.apply_mass,
                           os.path.join(file_dir, "test-qfunctions.h:apply_mass"))
  qf_mass.add_input("rho", 1, libceed.EVAL_NONE)
  qf_mass.add_input("u", 1, libceed.EVAL_INTERP)
  qf_mass.add_output("v", 1, libceed.EVAL_INTERP)

  # Operators
  op_setup = ceed.Operator(qf_setup)
  op_setup.set_field("weights", rxi, bx, libceed.VECTOR_NONE)
  op_setup.set_field("dx", rx, bx, libceed.VECTOR_ACTIVE)
  op_setup.set_field("rho", rui, libceed.BASIS_COLLOCATED,
                     libceed.VECTOR_ACTIVE)

  op_mass = ceed.Operator(qf_mass)
  op_mass.set_field("rho", rui, libceed.BASIS_COLLOCATED, qdata)
  op_mass.set_field("u", ru, bu, libceed.VECTOR_ACTIVE)
  op_mass.set_field("v", ru, bu, libceed.VECTOR_ACTIVE)

  # Setup
  op_setup.apply(x, qdata)

  # Apply mass matrix
  u.set_value(0.)
  op_mass.apply(u, v)

  # Check
  v_array = v.get_array_read()
  for i in range(ndofs):
    assert abs(v_array[i]) < 1E-14

  v.restore_array_read()

#-------------------------------------------------------------------------------
# Test creation, action, and destruction for mass matrix operator
#-------------------------------------------------------------------------------
def test_511(ceed_resource):
  ceed = libceed.Ceed(ceed_resource)

  nelem = 12
  dim = 2
  p = 6
  q = 4
  nx, ny = 3, 2
  ndofs = (nx*2+1)*(ny*2+1)
  nqpts = nelem*q

  # Vectors
  x = ceed.Vector(dim*ndofs)
  x_array = np.zeros(dim*ndofs)
  for i in range(ndofs):
    x_array[i] = (1. / (nx*2)) * (i % (nx*2+1))
    x_array[i+ndofs] = (1. / (ny*2)) * (i / (nx*2+1))
  x.set_array(x_array, cmode=libceed.USE_POINTER)

  qdata = ceed.Vector(nqpts)
  u = ceed.Vector(ndofs)
  v = ceed.Vector(ndofs)

  # Restrictions
  indx = np.zeros(nelem*p, dtype="int32")
  for i in range(nelem//2):
    col = i % nx;
    row = i // nx;
    offset = col*2 + row*(nx*2+1)*2

    indx[i*2*p+ 0] =  2 + offset
    indx[i*2*p+ 1] =  9 + offset
    indx[i*2*p+ 2] = 16 + offset
    indx[i*2*p+ 3] =  1 + offset
    indx[i*2*p+ 4] =  8 + offset
    indx[i*2*p+ 5] =  0 + offset

    indx[i*2*p+ 6] = 14 + offset
    indx[i*2*p+ 7] =  7 + offset
    indx[i*2*p+ 8] =  0 + offset
    indx[i*2*p+ 9] = 15 + offset
    indx[i*2*p+10] =  8 + offset
    indx[i*2*p+11] = 16 + offset

  rx = ceed.ElemRestriction(nelem, p, ndofs, dim, indx,
                            cmode=libceed.USE_POINTER)
  rxi = ceed.IdentityElemRestriction(nelem, p, nelem*p, dim)

  ru = ceed.ElemRestriction(nelem, p, ndofs, 1, indx, cmode=libceed.USE_POINTER)
  rui = ceed.IdentityElemRestriction(nelem, q, nqpts, 1)

  # Bases
  qref = np.empty(dim*q, dtype="float64")
  qweight = np.empty(q, dtype="float64")
  interp, grad = bm.buildmats(qref, qweight)

  bx = ceed.BasisH1(libceed.TRIANGLE, dim, p, q, interp, grad, qref, qweight)
  bu = ceed.BasisH1(libceed.TRIANGLE, 1, p, q, interp, grad, qref, qweight)

  # QFunctions
  file_dir = os.path.dirname(os.path.abspath(__file__))
  qfs = load_qfs_so()

  qf_setup = ceed.QFunction(1, qfs.setup_mass_2d,
                            os.path.join(file_dir, "test-qfunctions.h:setup_mass_2d"))
  qf_setup.add_input("weights", 1, libceed.EVAL_WEIGHT)
  qf_setup.add_input("dx", dim*dim, libceed.EVAL_GRAD)
  qf_setup.add_output("rho", 1, libceed.EVAL_NONE)

  qf_mass = ceed.QFunction(1, qfs.apply_mass,
                           os.path.join(file_dir, "test-qfunctions.h:apply_mass"))
  qf_mass.add_input("rho", 1, libceed.EVAL_NONE)
  qf_mass.add_input("u", 1, libceed.EVAL_INTERP)
  qf_mass.add_output("v", 1, libceed.EVAL_INTERP)

  # Operators
  op_setup = ceed.Operator(qf_setup)
  op_setup.set_field("weights", rxi, bx, libceed.VECTOR_NONE)
  op_setup.set_field("dx", rx, bx, libceed.VECTOR_ACTIVE)
  op_setup.set_field("rho", rui, libceed.BASIS_COLLOCATED,
                     libceed.VECTOR_ACTIVE)

  op_mass = ceed.Operator(qf_mass)
  op_mass.set_field("rho", rui, libceed.BASIS_COLLOCATED, qdata)
  op_mass.set_field("u", ru, bu, libceed.VECTOR_ACTIVE)
  op_mass.set_field("v", ru, bu, libceed.VECTOR_ACTIVE)

  # Setup
  op_setup.apply(x, qdata)

  # Apply mass matrix
  u.set_value(1.)
  op_mass.apply(u, v)

  # Check
  v_array = v.get_array_read()
  total = 0.0
  for i in range(ndofs):
    total = total + v_array[i]
  assert abs(total - 1.0) < 1E-10

  v.restore_array_read()

#-------------------------------------------------------------------------------
# Test creation, action, and destruction for mass matrix operator
#-------------------------------------------------------------------------------
def test_520(ceed_resource):
  ceed = libceed.Ceed(ceed_resource)

  nelem_tet, p_tet, q_tet = 6, 6, 4
  nelem_hex, p_hex, q_hex = 6, 3, 4
  nx, ny = 3, 3
  dim = 2
  nx_tet, ny_tet, nx_hex = 3, 1, 3
  ndofs = (nx*2+1)*(ny*2+1)
  nqpts_tet, nqpts_hex = nelem_tet*q_tet, nelem_hex*q_hex*q_hex

  # Vectors
  x = ceed.Vector(dim*ndofs)
  x_array = np.zeros(dim*ndofs)
  for i in range(ny*2+1):
    for j in range(nx*2+1):
      x_array[i+j*(ny*2+1)] = i/(2*ny)
      x_array[i+j*(ny*2+1)+ndofs] = j/(2*nx)
  x.set_array(x_array, cmode=libceed.USE_POINTER)

  qdata_hex = ceed.Vector(nqpts_hex)
  qdata_tet = ceed.Vector(nqpts_tet)
  u = ceed.Vector(ndofs)
  v = ceed.Vector(ndofs)

  ## ------------------------- Tet Elements -------------------------

  # Restrictions
  indx_tet = np.zeros(nelem_tet*p_tet, dtype="int32")
  for i in range(nelem_tet//2):
    col = i % nx;
    row = i // nx;
    offset = col*2 + row*(nx*2+1)*2

    indx_tet[i*2*p_tet+ 0] =  2 + offset
    indx_tet[i*2*p_tet+ 1] =  9 + offset
    indx_tet[i*2*p_tet+ 2] = 16 + offset
    indx_tet[i*2*p_tet+ 3] =  1 + offset
    indx_tet[i*2*p_tet+ 4] =  8 + offset
    indx_tet[i*2*p_tet+ 5] =  0 + offset

    indx_tet[i*2*p_tet+ 6] = 14 + offset
    indx_tet[i*2*p_tet+ 7] =  7 + offset
    indx_tet[i*2*p_tet+ 8] =  0 + offset
    indx_tet[i*2*p_tet+ 9] = 15 + offset
    indx_tet[i*2*p_tet+10] =  8 + offset
    indx_tet[i*2*p_tet+11] = 16 + offset

  rx_tet = ceed.ElemRestriction(nelem_tet, p_tet, ndofs, dim, indx_tet,
                                cmode=libceed.USE_POINTER)
  rxi_tet = ceed.IdentityElemRestriction(nelem_tet, p_tet, nelem_tet*p_tet, dim)

  ru_tet = ceed.ElemRestriction(nelem_tet, p_tet, ndofs, 1, indx_tet,
                                cmode=libceed.USE_POINTER)
  rui_tet = ceed.IdentityElemRestriction(nelem_tet, q_tet, nqpts_tet, 1)

  # Bases
  qref = np.empty(dim*q_tet, dtype="float64")
  qweight = np.empty(q_tet, dtype="float64")
  interp, grad = bm.buildmats(qref, qweight)

  bx_tet = ceed.BasisH1(libceed.TRIANGLE, dim, p_tet, q_hex, interp, grad, qref,
                        qweight)
  bu_tet = ceed.BasisH1(libceed.TRIANGLE, 1, p_tet, q_hex, interp, grad, qref,
                        qweight)

  # QFunctions
  file_dir = os.path.dirname(os.path.abspath(__file__))
  qfs = load_qfs_so()

  qf_setup_tet = ceed.QFunction(1, qfs.setup_mass_2d,
                                os.path.join(file_dir, "test-qfunctions.h:setup_mass_2d"))
  qf_setup_tet.add_input("weights", 1, libceed.EVAL_WEIGHT)
  qf_setup_tet.add_input("dx", dim*dim, libceed.EVAL_GRAD)
  qf_setup_tet.add_output("rho", 1, libceed.EVAL_NONE)

  qf_mass_tet = ceed.QFunction(1, qfs.apply_mass,
                               os.path.join(file_dir, "test-qfunctions.h:apply_mass"))
  qf_mass_tet.add_input("rho", 1, libceed.EVAL_NONE)
  qf_mass_tet.add_input("u", 1, libceed.EVAL_INTERP)
  qf_mass_tet.add_output("v", 1, libceed.EVAL_INTERP)

  # Operators
  op_setup_tet = ceed.Operator(qf_setup_tet)
  op_setup_tet.set_field("weights", rxi_tet, bx_tet, libceed.VECTOR_NONE)
  op_setup_tet.set_field("dx", rx_tet, bx_tet, libceed.VECTOR_ACTIVE)
  op_setup_tet.set_field("rho", rui_tet, libceed.BASIS_COLLOCATED,
                         qdata_tet)

  op_mass_tet = ceed.Operator(qf_mass_tet)
  op_mass_tet.set_field("rho", rui_tet, libceed.BASIS_COLLOCATED, qdata_tet)
  op_mass_tet.set_field("u", ru_tet, bu_tet, libceed.VECTOR_ACTIVE)
  op_mass_tet.set_field("v", ru_tet, bu_tet, libceed.VECTOR_ACTIVE)

  ## ------------------------- Hex Elements -------------------------

  # Restrictions
  indx_hex = np.zeros(nelem_hex*p_hex*p_hex, dtype="int32")
  for i in range(nelem_hex):
    col = i % nx_hex;
    row = i // nx_hex;
    offset = (nx_tet*2+1)*(ny_tet*2)*(1+row)+col*2

    for j in range(p_hex):
      for k in range(p_hex):
        indx_hex[p_hex*(p_hex*i+k)+j] = offset + k*(nx_hex*2+1) + j

  rx_hex = ceed.ElemRestriction(nelem_hex, p_hex*p_hex, ndofs, dim, indx_hex,
                            cmode=libceed.USE_POINTER)
  rxi_hex = ceed.IdentityElemRestriction(nelem_hex, p_hex*p_hex,
                                         nelem_hex*p_hex*p_hex, dim)

  ru_hex = ceed.ElemRestriction(nelem_hex, p_hex*p_hex, ndofs, 1, indx_hex,
                                cmode=libceed.USE_POINTER)
  rui_hex = ceed.IdentityElemRestriction(nelem_hex, q_hex*q_hex, nqpts_hex, 1)

  # Bases
  bx_hex = ceed.BasisTensorH1Lagrange(dim, dim, p_hex, q_hex, libceed.GAUSS)
  bu_hex = ceed.BasisTensorH1Lagrange(dim, 1, p_hex, q_hex, libceed.GAUSS)

  # QFunctions
  qf_setup_hex = ceed.QFunction(1, qfs.setup_mass_2d,
                                os.path.join(file_dir, "test-qfunctions.h:setup_mass_2d"))
  qf_setup_hex.add_input("weights", 1, libceed.EVAL_WEIGHT)
  qf_setup_hex.add_input("dx", dim*dim, libceed.EVAL_GRAD)
  qf_setup_hex.add_output("rho", 1, libceed.EVAL_NONE)

  qf_mass_hex = ceed.QFunction(1, qfs.apply_mass,
                               os.path.join(file_dir, "test-qfunctions.h:apply_mass"))
  qf_mass_hex.add_input("rho", 1, libceed.EVAL_NONE)
  qf_mass_hex.add_input("u", 1, libceed.EVAL_INTERP)
  qf_mass_hex.add_output("v", 1, libceed.EVAL_INTERP)

  # Operators
  op_setup_hex = ceed.Operator(qf_setup_tet)
  op_setup_hex.set_field("weights", rxi_hex, bx_hex, libceed.VECTOR_NONE)
  op_setup_hex.set_field("dx", rx_hex, bx_hex, libceed.VECTOR_ACTIVE)
  op_setup_hex.set_field("rho", rui_hex, libceed.BASIS_COLLOCATED,
                         qdata_hex)

  op_mass_hex = ceed.Operator(qf_mass_hex)
  op_mass_hex.set_field("rho", rui_hex, libceed.BASIS_COLLOCATED, qdata_hex)
  op_mass_hex.set_field("u", ru_hex, bu_hex, libceed.VECTOR_ACTIVE)
  op_mass_hex.set_field("v", ru_hex, bu_hex, libceed.VECTOR_ACTIVE)

  ## ------------------------- Composite Operators -------------------------

  # Setup
  op_setup = ceed.CompositeOperator()
  op_setup.add_sub(op_setup_tet)
  op_setup.add_sub(op_setup_hex)
  op_setup.apply(x, libceed.VECTOR_NONE)

  # Apply mass matrix
  op_mass = ceed.CompositeOperator()
  op_mass.add_sub(op_mass_tet)
  op_mass.add_sub(op_mass_hex)

  u.set_value(0.)
  op_mass.apply(u, v)

  # Check
  v_array = v.get_array_read()
  for i in range(ndofs):
    assert abs(v_array[i]) < 1E-14

  v.restore_array_read()

#-------------------------------------------------------------------------------
# Test creation, action, and destruction for mass matrix operator
#-------------------------------------------------------------------------------
def test_521(ceed_resource):
  ceed = libceed.Ceed(ceed_resource)

  nelem_tet, p_tet, q_tet = 6, 6, 4
  nelem_hex, p_hex, q_hex = 6, 3, 4
  nx, ny = 3, 3
  dim = 2
  nx_tet, ny_tet, nx_hex = 3, 1, 3
  ndofs = (nx*2+1)*(ny*2+1)
  nqpts_tet, nqpts_hex = nelem_tet*q_tet, nelem_hex*q_hex*q_hex

  # Vectors
  x = ceed.Vector(dim*ndofs)
  x_array = np.zeros(dim*ndofs)
  for i in range(ny*2+1):
    for j in range(nx*2+1):
      x_array[i+j*(ny*2+1)] = i/(2*ny)
      x_array[i+j*(ny*2+1)+ndofs] = j/(2*nx)
  x.set_array(x_array, cmode=libceed.USE_POINTER)

  qdata_hex = ceed.Vector(nqpts_hex)
  qdata_tet = ceed.Vector(nqpts_tet)
  u = ceed.Vector(ndofs)
  v = ceed.Vector(ndofs)

  ## ------------------------- Tet Elements -------------------------

  # Restrictions
  indx_tet = np.zeros(nelem_tet*p_tet, dtype="int32")
  for i in range(nelem_tet//2):
    col = i % nx;
    row = i // nx;
    offset = col*2 + row*(nx*2+1)*2

    indx_tet[i*2*p_tet+ 0] =  2 + offset
    indx_tet[i*2*p_tet+ 1] =  9 + offset
    indx_tet[i*2*p_tet+ 2] = 16 + offset
    indx_tet[i*2*p_tet+ 3] =  1 + offset
    indx_tet[i*2*p_tet+ 4] =  8 + offset
    indx_tet[i*2*p_tet+ 5] =  0 + offset

    indx_tet[i*2*p_tet+ 6] = 14 + offset
    indx_tet[i*2*p_tet+ 7] =  7 + offset
    indx_tet[i*2*p_tet+ 8] =  0 + offset
    indx_tet[i*2*p_tet+ 9] = 15 + offset
    indx_tet[i*2*p_tet+10] =  8 + offset
    indx_tet[i*2*p_tet+11] = 16 + offset

  rx_tet = ceed.ElemRestriction(nelem_tet, p_tet, ndofs, dim, indx_tet,
                            cmode=libceed.USE_POINTER)
  rxi_tet = ceed.IdentityElemRestriction(nelem_tet, p_tet, nelem_tet*p_tet, dim)

  ru_tet = ceed.ElemRestriction(nelem_tet, p_tet, ndofs, 1, indx_tet,
                                cmode=libceed.USE_POINTER)
  rui_tet = ceed.IdentityElemRestriction(nelem_tet, q_tet, nqpts_tet, 1)

  # Bases
  qref = np.empty(dim*q_tet, dtype="float64")
  qweight = np.empty(q_tet, dtype="float64")
  interp, grad = bm.buildmats(qref, qweight)

  bx_tet = ceed.BasisH1(libceed.TRIANGLE, dim, p_tet, q_hex, interp, grad, qref,
                        qweight)
  bu_tet = ceed.BasisH1(libceed.TRIANGLE, 1, p_tet, q_hex, interp, grad, qref,
                        qweight)

  # QFunctions
  file_dir = os.path.dirname(os.path.abspath(__file__))
  qfs = load_qfs_so()

  qf_setup_tet = ceed.QFunction(1, qfs.setup_mass_2d,
                                os.path.join(file_dir, "test-qfunctions.h:setup_mass_2d"))
  qf_setup_tet.add_input("weights", 1, libceed.EVAL_WEIGHT)
  qf_setup_tet.add_input("dx", dim*dim, libceed.EVAL_GRAD)
  qf_setup_tet.add_output("rho", 1, libceed.EVAL_NONE)

  qf_mass_tet = ceed.QFunction(1, qfs.apply_mass,
                               os.path.join(file_dir, "test-qfunctions.h:apply_mass"))
  qf_mass_tet.add_input("rho", 1, libceed.EVAL_NONE)
  qf_mass_tet.add_input("u", 1, libceed.EVAL_INTERP)
  qf_mass_tet.add_output("v", 1, libceed.EVAL_INTERP)

  # Operators
  op_setup_tet = ceed.Operator(qf_setup_tet)
  op_setup_tet.set_field("weights", rxi_tet, bx_tet, libceed.VECTOR_NONE)
  op_setup_tet.set_field("dx", rx_tet, bx_tet, libceed.VECTOR_ACTIVE)
  op_setup_tet.set_field("rho", rui_tet, libceed.BASIS_COLLOCATED,
                         qdata_tet)

  op_mass_tet = ceed.Operator(qf_mass_tet)
  op_mass_tet.set_field("rho", rui_tet, libceed.BASIS_COLLOCATED, qdata_tet)
  op_mass_tet.set_field("u", ru_tet, bu_tet, libceed.VECTOR_ACTIVE)
  op_mass_tet.set_field("v", ru_tet, bu_tet, libceed.VECTOR_ACTIVE)

  ## ------------------------- Hex Elements -------------------------

  # Restrictions
  indx_hex = np.zeros(nelem_hex*p_hex*p_hex, dtype="int32")
  for i in range(nelem_hex):
    col = i % nx_hex;
    row = i // nx_hex;
    offset = (nx_tet*2+1)*(ny_tet*2)*(1+row)+col*2

    for j in range(p_hex):
      for k in range(p_hex):
        indx_hex[p_hex*(p_hex*i+k)+j] = offset + k*(nx_hex*2+1) + j

  rx_hex = ceed.ElemRestriction(nelem_hex, p_hex*p_hex, ndofs, dim, indx_hex,
                                cmode=libceed.USE_POINTER)
  rxi_hex = ceed.IdentityElemRestriction(nelem_hex, p_hex*p_hex,
                                         nelem_hex*p_hex*p_hex, dim)

  ru_hex = ceed.ElemRestriction(nelem_hex, p_hex*p_hex, ndofs, 1, indx_hex,
                                cmode=libceed.USE_POINTER)
  rui_hex = ceed.IdentityElemRestriction(nelem_hex, q_hex*q_hex, nqpts_hex, 1)

  # Bases
  bx_hex = ceed.BasisTensorH1Lagrange(dim, dim, p_hex, q_hex, libceed.GAUSS)
  bu_hex = ceed.BasisTensorH1Lagrange(dim, 1, p_hex, q_hex, libceed.GAUSS)

  # QFunctions
  qf_setup_hex = ceed.QFunction(1, qfs.setup_mass_2d,
                                os.path.join(file_dir, "test-qfunctions.h:setup_mass_2d"))
  qf_setup_hex.add_input("weights", 1, libceed.EVAL_WEIGHT)
  qf_setup_hex.add_input("dx", dim*dim, libceed.EVAL_GRAD)
  qf_setup_hex.add_output("rho", 1, libceed.EVAL_NONE)

  qf_mass_hex = ceed.QFunction(1, qfs.apply_mass,
                               os.path.join(file_dir, "test-qfunctions.h:apply_mass"))
  qf_mass_hex.add_input("rho", 1, libceed.EVAL_NONE)
  qf_mass_hex.add_input("u", 1, libceed.EVAL_INTERP)
  qf_mass_hex.add_output("v", 1, libceed.EVAL_INTERP)

  # Operators
  op_setup_hex = ceed.Operator(qf_setup_tet)
  op_setup_hex.set_field("weights", rxi_hex, bx_hex, libceed.VECTOR_NONE)
  op_setup_hex.set_field("dx", rx_hex, bx_hex, libceed.VECTOR_ACTIVE)
  op_setup_hex.set_field("rho", rui_hex, libceed.BASIS_COLLOCATED,
                         qdata_hex)

  op_mass_hex = ceed.Operator(qf_mass_hex)
  op_mass_hex.set_field("rho", rui_hex, libceed.BASIS_COLLOCATED, qdata_hex)
  op_mass_hex.set_field("u", ru_hex, bu_hex, libceed.VECTOR_ACTIVE)
  op_mass_hex.set_field("v", ru_hex, bu_hex, libceed.VECTOR_ACTIVE)

  ## ------------------------- Composite Operators -------------------------

  # Setup
  op_setup = ceed.CompositeOperator()
  op_setup.add_sub(op_setup_tet)
  op_setup.add_sub(op_setup_hex)
  op_setup.apply(x, libceed.VECTOR_NONE)

  # Apply mass matrix
  op_mass = ceed.CompositeOperator()
  op_mass.add_sub(op_mass_tet)
  op_mass.add_sub(op_mass_hex)
  u.set_value(1.)
  op_mass.apply(u, v)

  # Check
  v_array = v.get_array_read()
  total = 0.0
  for i in range(ndofs):
    total = total + v_array[i]
  assert abs(total - 1.0) < 1E-10

  v.restore_array_read()

#-------------------------------------------------------------------------------
# Test viewing of composite mass matrix operator
#-------------------------------------------------------------------------------
def test_523(ceed_resource, capsys):
  ceed = libceed.Ceed(ceed_resource)

  nelem_tet, p_tet, q_tet = 6, 6, 4
  nelem_hex, p_hex, q_hex = 6, 3, 4
  nx, ny = 3, 3
  dim = 2
  nx_tet, ny_tet, nx_hex = 3, 1, 3
  ndofs = (nx*2+1)*(ny*2+1)
  nqpts_tet, nqpts_hex = nelem_tet*q_tet, nelem_hex*q_hex*q_hex

  # Vectors
  qdata_hex = ceed.Vector(nqpts_hex)
  qdata_tet = ceed.Vector(nqpts_tet)

  ## ------------------------- Tet Elements -------------------------

  # Restrictions
  indx_tet = np.zeros(nelem_tet*p_tet, dtype="int32")
  for i in range(nelem_tet//2):
    col = i % nx;
    row = i // nx;
    offset = col*2 + row*(nx*2+1)*2

    indx_tet[i*2*p_tet+ 0] =  2 + offset
    indx_tet[i*2*p_tet+ 1] =  9 + offset
    indx_tet[i*2*p_tet+ 2] = 16 + offset
    indx_tet[i*2*p_tet+ 3] =  1 + offset
    indx_tet[i*2*p_tet+ 4] =  8 + offset
    indx_tet[i*2*p_tet+ 5] =  0 + offset

    indx_tet[i*2*p_tet+ 6] = 14 + offset
    indx_tet[i*2*p_tet+ 7] =  7 + offset
    indx_tet[i*2*p_tet+ 8] =  0 + offset
    indx_tet[i*2*p_tet+ 9] = 15 + offset
    indx_tet[i*2*p_tet+10] =  8 + offset
    indx_tet[i*2*p_tet+11] = 16 + offset

  rx_tet = ceed.ElemRestriction(nelem_tet, p_tet, ndofs, dim, indx_tet,
                            cmode=libceed.USE_POINTER)
  rxi_tet = ceed.IdentityElemRestriction(nelem_tet, p_tet, nelem_tet*p_tet, dim)

  ru_tet = ceed.ElemRestriction(nelem_tet, p_tet, ndofs, 1, indx_tet,
                                cmode=libceed.USE_POINTER)
  rui_tet = ceed.IdentityElemRestriction(nelem_tet, q_tet, nqpts_tet, 1)

  # Bases
  qref = np.empty(dim*q_tet, dtype="float64")
  qweight = np.empty(q_tet, dtype="float64")
  interp, grad = bm.buildmats(qref, qweight)

  bx_tet = ceed.BasisH1(libceed.TRIANGLE, dim, p_tet, q_hex, interp, grad, qref,
                        qweight)
  bu_tet = ceed.BasisH1(libceed.TRIANGLE, 1, p_tet, q_hex, interp, grad, qref,
                        qweight)

  # QFunctions
  file_dir = os.path.dirname(os.path.abspath(__file__))
  qfs = load_qfs_so()

  qf_setup_tet = ceed.QFunction(1, qfs.setup_mass_2d,
                                os.path.join(file_dir, "test-qfunctions.h:setup_mass_2d"))
  qf_setup_tet.add_input("weights", 1, libceed.EVAL_WEIGHT)
  qf_setup_tet.add_input("dx", dim*dim, libceed.EVAL_GRAD)
  qf_setup_tet.add_output("rho", 1, libceed.EVAL_NONE)

  qf_mass_tet = ceed.QFunction(1, qfs.apply_mass,
                               os.path.join(file_dir, "test-qfunctions.h:apply_mass"))
  qf_mass_tet.add_input("rho", 1, libceed.EVAL_NONE)
  qf_mass_tet.add_input("u", 1, libceed.EVAL_INTERP)
  qf_mass_tet.add_output("v", 1, libceed.EVAL_INTERP)

  # Operators
  op_setup_tet = ceed.Operator(qf_setup_tet)
  op_setup_tet.set_field("weights", rxi_tet, bx_tet, libceed.VECTOR_NONE)
  op_setup_tet.set_field("dx", rx_tet, bx_tet, libceed.VECTOR_ACTIVE)
  op_setup_tet.set_field("rho", rui_tet, libceed.BASIS_COLLOCATED,
                         qdata_tet)

  op_mass_tet = ceed.Operator(qf_mass_tet)
  op_mass_tet.set_field("rho", rui_tet, libceed.BASIS_COLLOCATED, qdata_tet)
  op_mass_tet.set_field("u", ru_tet, bu_tet, libceed.VECTOR_ACTIVE)
  op_mass_tet.set_field("v", ru_tet, bu_tet, libceed.VECTOR_ACTIVE)

  ## ------------------------- Hex Elements -------------------------

  # Restrictions
  indx_hex = np.zeros(nelem_hex*p_hex*p_hex, dtype="int32")
  for i in range(nelem_hex):
    col = i % nx_hex;
    row = i // nx_hex;
    offset = (nx_tet*2+1)*(ny_tet*2)*(1+row)+col*2

    for j in range(p_hex):
      for k in range(p_hex):
        indx_hex[p_hex*(p_hex*i+k)+j] = offset + k*(nx_hex*2+1) + j

  rx_hex = ceed.ElemRestriction(nelem_hex, p_hex*p_hex, ndofs, dim, indx_hex,
                                cmode=libceed.USE_POINTER)
  rxi_hex = ceed.IdentityElemRestriction(nelem_hex, p_hex*p_hex,
                                         nelem_hex*p_hex*p_hex, dim)

  ru_hex = ceed.ElemRestriction(nelem_hex, p_hex*p_hex, ndofs, 1, indx_hex,
                                cmode=libceed.USE_POINTER)
  rui_hex = ceed.IdentityElemRestriction(nelem_hex, q_hex*q_hex, nqpts_hex, 1)

  # Bases
  bx_hex = ceed.BasisTensorH1Lagrange(dim, dim, p_hex, q_hex, libceed.GAUSS)
  bu_hex = ceed.BasisTensorH1Lagrange(dim, 1, p_hex, q_hex, libceed.GAUSS)

  # QFunctions
  qf_setup_hex = ceed.QFunction(1, qfs.setup_mass_2d,
                                os.path.join(file_dir, "test-qfunctions.h:setup_mass_2d"))
  qf_setup_hex.add_input("weights", 1, libceed.EVAL_WEIGHT)
  qf_setup_hex.add_input("dx", dim*dim, libceed.EVAL_GRAD)
  qf_setup_hex.add_output("rho", 1, libceed.EVAL_NONE)

  qf_mass_hex = ceed.QFunction(1, qfs.apply_mass,
                               os.path.join(file_dir, "test-qfunctions.h:apply_mass"))
  qf_mass_hex.add_input("rho", 1, libceed.EVAL_NONE)
  qf_mass_hex.add_input("u", 1, libceed.EVAL_INTERP)
  qf_mass_hex.add_output("v", 1, libceed.EVAL_INTERP)

  # Operators
  op_setup_hex = ceed.Operator(qf_setup_tet)
  op_setup_hex.set_field("weights", rxi_hex, bx_hex, libceed.VECTOR_NONE)
  op_setup_hex.set_field("dx", rx_hex, bx_hex, libceed.VECTOR_ACTIVE)
  op_setup_hex.set_field("rho", rui_hex, libceed.BASIS_COLLOCATED,
                         qdata_hex)

  op_mass_hex = ceed.Operator(qf_mass_hex)
  op_mass_hex.set_field("rho", rui_hex, libceed.BASIS_COLLOCATED, qdata_hex)
  op_mass_hex.set_field("u", ru_hex, bu_hex, libceed.VECTOR_ACTIVE)
  op_mass_hex.set_field("v", ru_hex, bu_hex, libceed.VECTOR_ACTIVE)

  ## ------------------------- Composite Operators -------------------------

  # Setup
  op_setup = ceed.CompositeOperator()
  op_setup.add_sub(op_setup_tet)
  op_setup.add_sub(op_setup_hex)

  # Apply mass matrix
  op_mass = ceed.CompositeOperator()
  op_mass.add_sub(op_mass_tet)
  op_mass.add_sub(op_mass_hex)

  # View
  print(op_setup)
  print(op_mass)

  stdout, stderr = capsys.readouterr()
  with open(os.path.abspath("./output/test_523.out")) as output_file:
    true_output = output_file.read()

  assert stdout == true_output

#-------------------------------------------------------------------------------
# Test creation, action, and destruction for mass matrix operator
#-------------------------------------------------------------------------------
def test_524(ceed_resource):
  ceed = libceed.Ceed(ceed_resource)

  nelem_tet, p_tet, q_tet = 6, 6, 4
  nelem_hex, p_hex, q_hex = 6, 3, 4
  nx, ny = 3, 3
  dim = 2
  nx_tet, ny_tet, nx_hex = 3, 1, 3
  ndofs = (nx*2+1)*(ny*2+1)
  nqpts_tet, nqpts_hex = nelem_tet*q_tet, nelem_hex*q_hex*q_hex

  # Vectors
  x = ceed.Vector(dim*ndofs)
  x_array = np.zeros(dim*ndofs)
  for i in range(ny*2+1):
    for j in range(nx*2+1):
      x_array[i+j*(ny*2+1)] = i/(2*ny)
      x_array[i+j*(ny*2+1)+ndofs] = j/(2*nx)
  x.set_array(x_array, cmode=libceed.USE_POINTER)

  qdata_hex = ceed.Vector(nqpts_hex)
  qdata_tet = ceed.Vector(nqpts_tet)
  u = ceed.Vector(ndofs)
  v = ceed.Vector(ndofs)

  ## ------------------------- Tet Elements -------------------------

  # Restrictions
  indx_tet = np.zeros(nelem_tet*p_tet, dtype="int32")
  for i in range(nelem_tet//2):
    col = i % nx;
    row = i // nx;
    offset = col*2 + row*(nx*2+1)*2

    indx_tet[i*2*p_tet+ 0] =  2 + offset
    indx_tet[i*2*p_tet+ 1] =  9 + offset
    indx_tet[i*2*p_tet+ 2] = 16 + offset
    indx_tet[i*2*p_tet+ 3] =  1 + offset
    indx_tet[i*2*p_tet+ 4] =  8 + offset
    indx_tet[i*2*p_tet+ 5] =  0 + offset

    indx_tet[i*2*p_tet+ 6] = 14 + offset
    indx_tet[i*2*p_tet+ 7] =  7 + offset
    indx_tet[i*2*p_tet+ 8] =  0 + offset
    indx_tet[i*2*p_tet+ 9] = 15 + offset
    indx_tet[i*2*p_tet+10] =  8 + offset
    indx_tet[i*2*p_tet+11] = 16 + offset

  rx_tet = ceed.ElemRestriction(nelem_tet, p_tet, ndofs, dim, indx_tet,
                            cmode=libceed.USE_POINTER)
  rxi_tet = ceed.IdentityElemRestriction(nelem_tet, p_tet, nelem_tet*p_tet, dim)

  ru_tet = ceed.ElemRestriction(nelem_tet, p_tet, ndofs, 1, indx_tet,
                                cmode=libceed.USE_POINTER)
  rui_tet = ceed.IdentityElemRestriction(nelem_tet, q_tet, nqpts_tet, 1)

  # Bases
  qref = np.empty(dim*q_tet, dtype="float64")
  qweight = np.empty(q_tet, dtype="float64")
  interp, grad = bm.buildmats(qref, qweight)

  bx_tet = ceed.BasisH1(libceed.TRIANGLE, dim, p_tet, q_hex, interp, grad, qref,
                        qweight)
  bu_tet = ceed.BasisH1(libceed.TRIANGLE, 1, p_tet, q_hex, interp, grad, qref,
                        qweight)

  # QFunctions
  file_dir = os.path.dirname(os.path.abspath(__file__))
  qfs = load_qfs_so()

  qf_setup_tet = ceed.QFunction(1, qfs.setup_mass_2d,
                                os.path.join(file_dir, "test-qfunctions.h:setup_mass_2d"))
  qf_setup_tet.add_input("weights", 1, libceed.EVAL_WEIGHT)
  qf_setup_tet.add_input("dx", dim*dim, libceed.EVAL_GRAD)
  qf_setup_tet.add_output("rho", 1, libceed.EVAL_NONE)

  qf_mass_tet = ceed.QFunction(1, qfs.apply_mass,
                               os.path.join(file_dir, "test-qfunctions.h:apply_mass"))
  qf_mass_tet.add_input("rho", 1, libceed.EVAL_NONE)
  qf_mass_tet.add_input("u", 1, libceed.EVAL_INTERP)
  qf_mass_tet.add_output("v", 1, libceed.EVAL_INTERP)

  # Operators
  op_setup_tet = ceed.Operator(qf_setup_tet)
  op_setup_tet.set_field("weights", rxi_tet, bx_tet, libceed.VECTOR_NONE)
  op_setup_tet.set_field("dx", rx_tet, bx_tet, libceed.VECTOR_ACTIVE)
  op_setup_tet.set_field("rho", rui_tet, libceed.BASIS_COLLOCATED,
                         qdata_tet)

  op_mass_tet = ceed.Operator(qf_mass_tet)
  op_mass_tet.set_field("rho", rui_tet, libceed.BASIS_COLLOCATED, qdata_tet)
  op_mass_tet.set_field("u", ru_tet, bu_tet, libceed.VECTOR_ACTIVE)
  op_mass_tet.set_field("v", ru_tet, bu_tet, libceed.VECTOR_ACTIVE)

  ## ------------------------- Hex Elements -------------------------

  # Restrictions
  indx_hex = np.zeros(nelem_hex*p_hex*p_hex, dtype="int32")
  for i in range(nelem_hex):
    col = i % nx_hex;
    row = i // nx_hex;
    offset = (nx_tet*2+1)*(ny_tet*2)*(1+row)+col*2

    for j in range(p_hex):
      for k in range(p_hex):
        indx_hex[p_hex*(p_hex*i+k)+j] = offset + k*(nx_hex*2+1) + j

  rx_hex = ceed.ElemRestriction(nelem_hex, p_hex*p_hex, ndofs, dim, indx_hex,
                                cmode=libceed.USE_POINTER)
  rxi_hex = ceed.IdentityElemRestriction(nelem_hex, p_hex*p_hex,
                                         nelem_hex*p_hex*p_hex, dim)

  ru_hex = ceed.ElemRestriction(nelem_hex, p_hex*p_hex, ndofs, 1, indx_hex,
                                cmode=libceed.USE_POINTER)
  rui_hex = ceed.IdentityElemRestriction(nelem_hex, q_hex*q_hex, nqpts_hex, 1)

  # Bases
  bx_hex = ceed.BasisTensorH1Lagrange(dim, dim, p_hex, q_hex, libceed.GAUSS)
  bu_hex = ceed.BasisTensorH1Lagrange(dim, 1, p_hex, q_hex, libceed.GAUSS)

  # QFunctions
  qf_setup_hex = ceed.QFunction(1, qfs.setup_mass_2d,
                                os.path.join(file_dir, "test-qfunctions.h:setup_mass_2d"))
  qf_setup_hex.add_input("weights", 1, libceed.EVAL_WEIGHT)
  qf_setup_hex.add_input("dx", dim*dim, libceed.EVAL_GRAD)
  qf_setup_hex.add_output("rho", 1, libceed.EVAL_NONE)

  qf_mass_hex = ceed.QFunction(1, qfs.apply_mass,
                               os.path.join(file_dir, "test-qfunctions.h:apply_mass"))
  qf_mass_hex.add_input("rho", 1, libceed.EVAL_NONE)
  qf_mass_hex.add_input("u", 1, libceed.EVAL_INTERP)
  qf_mass_hex.add_output("v", 1, libceed.EVAL_INTERP)

  # Operators
  op_setup_hex = ceed.Operator(qf_setup_tet)
  op_setup_hex.set_field("weights", rxi_hex, bx_hex, libceed.VECTOR_NONE)
  op_setup_hex.set_field("dx", rx_hex, bx_hex, libceed.VECTOR_ACTIVE)
  op_setup_hex.set_field("rho", rui_hex, libceed.BASIS_COLLOCATED,
                         qdata_hex)

  op_mass_hex = ceed.Operator(qf_mass_hex)
  op_mass_hex.set_field("rho", rui_hex, libceed.BASIS_COLLOCATED, qdata_hex)
  op_mass_hex.set_field("u", ru_hex, bu_hex, libceed.VECTOR_ACTIVE)
  op_mass_hex.set_field("v", ru_hex, bu_hex, libceed.VECTOR_ACTIVE)

  ## ------------------------- Composite Operators -------------------------

  # Setup
  op_setup = ceed.CompositeOperator()
  op_setup.add_sub(op_setup_tet)
  op_setup.add_sub(op_setup_hex)
  op_setup.apply(x, libceed.VECTOR_NONE)

  # Apply mass matrix
  op_mass = ceed.CompositeOperator()
  op_mass.add_sub(op_mass_tet)
  op_mass.add_sub(op_mass_hex)
  u.set_value(1.)
  op_mass.apply(u, v)

  # Check
  v_array = v.get_array_read()
  total = 0.0
  for i in range(ndofs):
    total = total + v_array[i]
  assert abs(total - 1.0) < 1E-10

  v.restore_array_read()

  # ApplyAdd mass matrix
  v.set_value(1.)
  op_mass.apply_add(u, v)

  # Check
  v_array = v.get_array_read()
  total = -ndofs
  for i in range(ndofs):
    total = total + v_array[i]
  assert abs(total - 1.0) < 1E-10

  v.restore_array_read()

#-------------------------------------------------------------------------------
