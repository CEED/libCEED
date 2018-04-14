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

NEK5K_DIR ?= ../../../Nek5000
CEED_DIR ?= ../..
FC_EXAMPLES = $(if $(findstring mpi,$(FC)),$(FC),mpif77)
CC_EXAMPLES = $(if $(findstring mpi,$(CC)),$(FC),mpicc)

NEK5K_EXAMPLES ?= bp1

.PHONY: all examples clean

all: examples

examples:
	export NEK5K_DIR=$(NEK5K_DIR) && export CEED_DIR=$(CEED_DIR) && \
		export FC=$(FC_EXAMPLES) && export CC=$(CC_EXAMPLES) && \
		export EXAMPLES=$(NEK5K_EXAMPLES) && ./make-nek-examples.sh

clean:
	./make-nek-examples.sh clean
