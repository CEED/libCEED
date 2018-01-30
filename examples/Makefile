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
CEED_DIR ?= ..
CEED_FLAGS ?= -I$(CEED_DIR) -std=c99
CEED_LIBS ?= -Wl,-rpath,$(abspath $(CEED_DIR)) -L$(CEED_DIR) -lceed -lm

OCCA_DIR ?= ../../occa
ifneq ($(wildcard $(OCCA_DIR)/lib/libocca.*),)
  OCCA_LIBS ?= -Wl,-rpath,$(abspath $(OCCA_DIR)/lib) -L$(OCCA_DIR)/lib -locca
endif

EXAMPLES = ex1

.SUFFIXES:
.SUFFIXES: .c
.PHONY: all clean

all: $(EXAMPLES)

# Remove built-in rules
%: %.c

# Rules for building the examples
%: %.c
	$(LINK.c) $(CEED_FLAGS) $< -o $@ $(CEED_LIBS) $(OCCA_LIBS)

clean:
	rm -f *~ $(EXAMPLES)
	rm -rf *.dSYM *.TVD.*breakpoints
