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

# Ceed directory
CEED_DIR ?= ../..

NVCC ?= nvcc
ifeq ($(shell which $(NVCC)),)
	CEED_FLAGS ?= -I$(CEED_DIR)/include -std=c99
	CEED_LIBS ?= -Wl,-rpath,$(abspath $(CEED_DIR)/lib) -L$(CEED_DIR)/lib -lceed -lm
else
	CC = $(NVCC)
	CEED_FLAGS ?= -I$(CEED_DIR)/include -x=cu
	CEED_LIBS ?= -Xlinker="-rpath,$(abspath $(CEED_DIR)/lib)" -L$(CEED_DIR)/lib -lceed -lm -lcudart
endif

EXAMPLES.c = $(wildcard ex*.c)
EXAMPLES = $(EXAMPLES.c:%.c=%)

.SUFFIXES:
.SUFFIXES: .c
.PHONY: all clean

all: $(EXAMPLES)

# Remove built-in rules
%: %.c

# Rules for building the examples
%: %.c
	$(CC) $(CEED_FLAGS) $< -o $@ $(CEED_LIBS)

clean:
	rm -f *~ $(EXAMPLES)
	rm -rf *.dSYM *.TVD.*breakpoints
