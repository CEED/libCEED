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
# Test Ceed functionality

import libceed

#-------------------------------------------------------------------------------
# Test creation and destruction of a Ceed object
#-------------------------------------------------------------------------------
def test_000(ceed_resource):
  ceed = libceed.Ceed(ceed_resource)

#-------------------------------------------------------------------------------
# Test return of Ceed backend prefered memory type
#-------------------------------------------------------------------------------
def test_001(ceed_resource):
  ceed = libceed.Ceed(ceed_resource)

  memtype = ceed.get_preferred_memtype()

  assert memtype != "error"

#-------------------------------------------------------------------------------
# Test return of a CEED object full resource name
#-------------------------------------------------------------------------------
def test_002(ceed_resource):
  ceed = libceed.Ceed(ceed_resource)

  resource = ceed.get_resource()

  assert resource == ceed_resource

#-------------------------------------------------------------------------------
