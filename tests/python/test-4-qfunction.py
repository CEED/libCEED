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
# Test Ceed QFunction functionality

import os
import glob
import ctypes
import libceed
import numpy as np

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
# Test creation, evaluation, and destruction for qfunction
#-------------------------------------------------------------------------------
def test_400(ceed_resource):
  ceed = libceed.Ceed(ceed_resource)

  file_dir = os.path.dirname(os.path.abspath(__file__))
  qfs = load_qfs_so()

  qf_setup = ceed.QFunction(1, qfs.setup_mass,
                            os.path.join(file_dir, "test-qfunctions.h:setup_mass"))
  qf_setup.add_input("w", 1, libceed.EVAL_WEIGHT)
  qf_setup.add_input("dx", 1, libceed.EVAL_GRAD)
  qf_setup.add_output("qdata", 1, libceed.EVAL_NONE)

  qf_mass = ceed.QFunction(1, qfs.apply_mass,
                           os.path.join(file_dir, "test-qfunctions.h:apply_mass"))
  qf_mass.add_input("qdata", 1, libceed.EVAL_NONE)
  qf_mass.add_input("u", 1, libceed.EVAL_INTERP)
  qf_mass.add_output("v", 1, libceed.EVAL_INTERP)

  q = 8

  w_array = np.zeros(q, dtype="float64")
  u_array = np.zeros(q, dtype="float64")
  v_true  = np.zeros(q, dtype="float64")
  for i in range(q):
    x = 2.*i/(q-1) - 1
    w_array[i] = 1 - x*x
    u_array[i] = 2 + 3*x + 5*x*x
    v_true[i]  = w_array[i] * u_array[i]

  dx = ceed.Vector(q)
  dx.set_value(1)
  w = ceed.Vector(q)
  w.set_array(w_array, cmode=libceed.USE_POINTER)
  u = ceed.Vector(q)
  u.set_array(u_array, cmode=libceed.USE_POINTER)
  v = ceed.Vector(q)
  v.set_value(0)
  qdata = ceed.Vector(q)
  qdata.set_value(0)

  inputs = [ dx, w ]
  outputs = [ qdata ]
  qf_setup.apply(q, inputs, outputs)

  inputs = [ qdata, u ]
  outputs = [ v ]
  qf_mass.apply(q, inputs, outputs)

  v_array = v.get_array_read()
  for i in range(q):
    assert v_array[i] == v_true[i]

  v.restore_array_read()

#-------------------------------------------------------------------------------
# Test creation, evaluation, and destruction for qfunction
#-------------------------------------------------------------------------------
def test_401(ceed_resource):
  ceed = libceed.Ceed(ceed_resource)

  file_dir = os.path.dirname(os.path.abspath(__file__))
  qfs = load_qfs_so()

  qf_setup = ceed.QFunction(1, qfs.setup_mass,
                            os.path.join(file_dir, "test-qfunctions.h:setup_mass"))
  qf_setup.add_input("w", 1, libceed.EVAL_WEIGHT)
  qf_setup.add_input("dx", 1, libceed.EVAL_GRAD)
  qf_setup.add_output("qdata", 1, libceed.EVAL_NONE)

  qf_mass = ceed.QFunction(1, qfs.apply_mass,
                           os.path.join(file_dir, "t400-qfunction.h:apply_mass"))
  qf_mass.add_input("qdata", 1, libceed.EVAL_NONE)
  qf_mass.add_input("u", 1, libceed.EVAL_INTERP)
  qf_mass.add_output("v", 1, libceed.EVAL_INTERP)

  ctx = np.array([1., 2., 3., 4., 5.])
  qf_mass.set_context(ctx)

  q = 8

  w_array = np.zeros(q, dtype="float64")
  u_array = np.zeros(q, dtype="float64")
  v_true  = np.zeros(q, dtype="float64")
  for i in range(q):
    x = 2.*i/(q-1) - 1
    w_array[i] = 1 - x*x
    u_array[i] = 2 + 3*x + 5*x*x
    v_true[i]  = 5* w_array[i] * u_array[i]

  dx = ceed.Vector(q)
  dx.set_value(1)
  w = ceed.Vector(q)
  w.set_array(w_array, cmode=libceed.USE_POINTER)
  u = ceed.Vector(q)
  u.set_array(u_array, cmode=libceed.USE_POINTER)
  v = ceed.Vector(q)
  v.set_value(0)
  qdata = ceed.Vector(q)
  qdata.set_value(0)

  inputs = [ dx, w ]
  outputs = [ qdata ]
  qf_setup.apply(q, inputs, outputs)

  inputs = [ qdata, u ]
  outputs = [ v ]
  qf_mass.apply(q, inputs, outputs)

  v_array = v.get_array_read()
  for i in range(q):
    assert v_array[i] == v_true[i]

  v.restore_array_read()

#-------------------------------------------------------------------------------
# Test viewing of qfunction
#-------------------------------------------------------------------------------
def test_402(ceed_resource, capsys):
  ceed = libceed.Ceed(ceed_resource)

  file_dir = os.path.dirname(os.path.abspath(__file__))
  qfs = load_qfs_so()

  qf_setup = ceed.QFunction(1, qfs.setup_mass,
                            os.path.join(file_dir, "test-qfunctions.h:setup_mass"))
  qf_setup.add_input("w", 1, libceed.EVAL_WEIGHT)
  qf_setup.add_input("dx", 1, libceed.EVAL_GRAD)
  qf_setup.add_output("qdata", 1, libceed.EVAL_NONE)

  qf_mass = ceed.QFunction(1, qfs.apply_mass,
                           os.path.join(file_dir, "t400-qfunction.h:apply_mass"))
  qf_mass.add_input("qdata", 1, libceed.EVAL_NONE)
  qf_mass.add_input("u", 1, libceed.EVAL_INTERP)
  qf_mass.add_output("v", 1, libceed.EVAL_INTERP)

  print(qf_setup)
  print(qf_mass)

  stdout, stderr = capsys.readouterr()
  with open(os.path.abspath("./output/test_402.out")) as output_file:
    true_output = output_file.read()

  assert stdout == true_output

#-------------------------------------------------------------------------------
# Test creation, evaluation, and destruction for qfunction by name
#-------------------------------------------------------------------------------
def test_410(ceed_resource):
  ceed = libceed.Ceed(ceed_resource)

  qf_setup = ceed.QFunctionByName("Mass1DBuild")
  qf_mass = ceed.QFunctionByName("MassApply")

  q = 8

  j_array = np.zeros(q, dtype="float64")
  w_array = np.zeros(q, dtype="float64")
  u_array = np.zeros(q, dtype="float64")
  v_true  = np.zeros(q, dtype="float64")
  for i in range(q):
    x = 2.*i/(q-1) - 1
    j_array[i] = 1
    w_array[i] = 1 - x*x
    u_array[i] = 2 + 3*x + 5*x*x
    v_true[i]  = w_array[i] * u_array[i]

  j = ceed.Vector(q)
  j.set_array(j_array, cmode=libceed.USE_POINTER)
  w = ceed.Vector(q)
  w.set_array(w_array, cmode=libceed.USE_POINTER)
  u = ceed.Vector(q)
  u.set_array(u_array, cmode=libceed.USE_POINTER)
  v = ceed.Vector(q)
  v.set_value(0)
  qdata = ceed.Vector(q)
  qdata.set_value(0)

  inputs = [ j, w ]
  outputs = [ qdata ]
  qf_setup.apply(q, inputs, outputs)

  inputs = [ w, u ]
  outputs = [ v ]
  qf_mass.apply(q, inputs, outputs)

  v_array = v.get_array_read()
  for i in range(q):
    assert v_array[i] == v_true[i]

  v.restore_array_read()

#-------------------------------------------------------------------------------
# Test creation, evaluation, and destruction of identity qfunction
#-------------------------------------------------------------------------------
def test_411(ceed_resource):
  ceed = libceed.Ceed(ceed_resource)

  qf = ceed.IdentityQFunction(1, libceed.EVAL_INTERP, libceed.EVAL_INTERP)

  q = 8

  u_array = np.zeros(q, dtype="float64")
  for i in range(q):
    u_array[i] = i*i

  u = ceed.Vector(q)
  u.set_array(u_array, cmode=libceed.USE_POINTER)
  v = ceed.Vector(q)
  v.set_value(0)

  inputs = [ u ]
  outputs = [ v ]
  qf.apply(q, inputs, outputs)

  v_array = v.get_array_read()
  for i in range(q):
    assert v_array[i] == i*i

  v.restore_array_read()

#-------------------------------------------------------------------------------
# Test creation, evaluation, and destruction of identity qfunction with size>1
#-------------------------------------------------------------------------------
def test_412(ceed_resource):
  ceed = libceed.Ceed(ceed_resource)

  size = 3
  qf = ceed.IdentityQFunction(size, libceed.EVAL_INTERP, libceed.EVAL_INTERP)

  q = 8

  u_array = np.zeros(q*size, dtype="float64")
  for i in range(q*size):
    u_array[i] = i*i

  u = ceed.Vector(q*size)
  u.set_array(u_array, cmode=libceed.USE_POINTER)
  v = ceed.Vector(q*size)
  v.set_value(0)

  inputs = [ u ]
  outputs = [ v ]
  qf.apply(q, inputs, outputs)

  v_array = v.get_array_read()
  for i in range(q*size):
    assert v_array[i] == i*i

  v.restore_array_read()

#-------------------------------------------------------------------------------
# Test viewing of qfunction by name
#-------------------------------------------------------------------------------
def test_413(ceed_resource, capsys):
  ceed = libceed.Ceed(ceed_resource)

  qf_setup = ceed.QFunctionByName("Mass1DBuild")
  qf_mass = ceed.QFunctionByName("MassApply")

  print(qf_setup)
  print(qf_mass)

  stdout, stderr = capsys.readouterr()
  with open(os.path.abspath("./output/test_413.out")) as output_file:
    true_output = output_file.read()

  assert stdout == true_output

#-------------------------------------------------------------------------------
