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

CEED_DIR ?= $(abspath ..)
CEED_FLAGS ?= -I$(CEED_DIR)/include
CEED_LIBS ?= -Wl,-rpath,$(CEED_DIR)/lib -L$(CEED_DIR)/lib -lceed

# Use an MFEM build directory or an MFEM install directory
MFEM_DIR ?= $(abspath ../../mfem)
CONFIG_MK ?= $(or $(wildcard $(MFEM_DIR)/config/config.mk),$(wildcard\
  $(MFEM_DIR)/share/mfem/config.mk),invalid_mfem_dir)

NEK5K_DIR ?= $(abspath ../../Nek5000)
NEK5K_FC = $(if $(findstring mpi,$(FC)),$(FC),mpif77)
NEK5K_CC = $(if $(findstring mpi,$(CC)),$(FC),mpicc)
NEK5K_EXAMPLES ?= bp1

PETSC_DIR ?=
PETSC_ARCH ?=

.PHONY: all nek ns mfem petsc ceed clean

all: ceed nek ns mfem petsc ceed

ceed:
	make CEED_DIR=$(CEED_DIR) -C ceed all

nek:
	cd nek5000 && NEK5K_DIR=$(NEK5K_DIR) CEED_DIR=$(CEED_DIR) \
		 FC=$(NEK5K_FC) CC=$(NEK5K_CC) \
		 EXAMPLES=$(NEK5K_EXAMPLES) ./make-nek-examples.sh

prepnektests:
	cd nek5000 && NEK5K_DIR=$(NEK5K_DIR) ./make-nek-tests.sh

mfem:
	make CEED_DIR=$(CEED_DIR) MFEM_DIR=$(MFEM_DIR) -C mfem all

petsc:
	make CEED_DIR=$(CEED_DIR) PETSC_DIR=$(PETSC_DIR) PETSC_ARCH=$(PETSC_ARCH) -C petsc all

ns:
	make CEED_DIR=$(CEED_DIR) PETSC_DIR=$(PETSC_DIR) PETSC_ARCH=$(PETSC_ARCH) -C navier-stokes all

clean:
	cd nek5000 && ./make-nek-examples.sh clean
	make -C mfem clean
	make -C petsc clean
	make -C navier-stokes clean
