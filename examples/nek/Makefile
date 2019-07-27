# Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
# the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
# reserved. See files LICENSE and NOTICE for details.
#
# This file is part of CEED, a collection of benchmarks, miniapps, software
# libraries and APIs for efficient high-order finite element and spectral
# element discretizations for exascale applications. For more information and
# source code availability see http://github.com/ceed.
#
# The CEED research is supported by the Exascale Computing Project (17-SC-20-SC)
# a collaborative effort of two U.S. Department of Energy organizations (Office
# of Science and the National Nuclear Security Administration) responsible for
# the planning and preparation of a capable exascale ecosystem, including
# software, applications, hardware, advanced system engineering and early
# testbed platforms, in support of the nation's exascale computing imperative.

NEK5K_DIR ?= $(abspath ../../../Nek5000)
MPI ?= 1

ifeq ($(MPI),0)
  ifeq (,$(filter-out undefined default,$(origin CC)))
    CC = gcc
  endif
  ifeq (,$(filter-out undefined default,$(origin FC)))
    FC = gfortran
  endif
else
  FC = mpif77
  CC = mpicc
endif

CEED_DIR ?= $(abspath ../..)
CEED_FLAGS ?= -I$(CEED_DIR)/include
CEED_LIBS ?= -Wl,-rpath,$(CEED_DIR)/lib -L$(CEED_DIR)/lib -lceed

.PHONY: all bps clean

all: bps

bps: bps/bps.usr
	FC=$(FC) CC=$(CC) MPI=$(MPI) NEK5K_DIR=$(NEK5K_DIR) CEED_DIR=$(CEED_DIR) && ./nek-examples.sh -m

clean:
	NEK5K_DIR=$(NEK5K_DIR) && ./nek-examples.sh -clean
