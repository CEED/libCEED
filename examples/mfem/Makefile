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
CEED_FLAGS ?= -I$(CEED_DIR)/include
CEED_LIBS ?= -Wl,-rpath,$(abspath $(CEED_DIR)/lib) -L$(CEED_DIR)/lib -lceed

# Use an MFEM build directory or an MFEM install directory
MFEM_DIR ?= ../../../mfem
CONFIG_MK ?= $(or $(wildcard $(MFEM_DIR)/config/config.mk),$(wildcard\
  $(MFEM_DIR)/share/mfem/config.mk),invalid_mfem_dir)
MFEM_DEF = -DMFEM_DIR="\"$(abspath $(MFEM_DIR))\""

MFEM_LIB_FILE = mfem_is_not_built
-include $(CONFIG_MK)

MFEM_EXAMPLES = bp1 bp3

.SUFFIXES:
.SUFFIXES: .cpp
.PHONY: all clean

all: $(MFEM_EXAMPLES)

# Remove built-in rules
%: %.cpp

# Rules for building the examples
%: %.cpp $(CONFIG_MK) $(MFEM_LIB_FILE)
	$(MFEM_CXX) $(CEED_FLAGS) $(MFEM_FLAGS) $(MFEM_DEF) $< -o $@ \
	  $(CEED_LIBS) $(MFEM_LIBS)

# Generate an error message if the MFEM library is not built and exit
$(MFEM_LIB_FILE):
	$(if $(wildcard $@),,$(error The MFEM library is not built))

$(CONFIG_MK):
	$(if $(wildcard $@),,$(error \
	  MFEM config file not found. Please set MFEM_DIR))

clean:
	rm -f *~ $(MFEM_EXAMPLES)
	rm -rf *.dSYM *.TVD.*breakpoints
