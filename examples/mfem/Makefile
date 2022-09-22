# Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors
# All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
#
# SPDX-License-Identifier: BSD-2-Clause
#
# This file is part of CEED:  http://github.com/ceed

COMMON ?= ../../common.mk
-include $(COMMON)

OPT ?= -O -g

# Ceed directory
CEED_DIR ?= ../..
CEED_FLAGS ?= -I$(CEED_DIR)/include $(OPT)
CEED_LIBS ?= -Wl,-rpath,$(abspath $(CEED_DIR)/lib) -L$(CEED_DIR)/lib -lceed

# Use an MFEM build directory or an MFEM install directory
MFEM_DIR ?= ../../../mfem
CONFIG_MK ?= $(or $(wildcard $(MFEM_DIR)/config/config.mk),$(wildcard\
  $(MFEM_DIR)/share/mfem/config.mk),invalid_mfem_dir)
MFEM_DEF = -DMFEM_DIR="\"$(abspath $(MFEM_DIR))\""

MFEM_LIB_FILE = mfem_is_not_built
-include $(wildcard $(CONFIG_MK))

MFEM_EXAMPLES =  bp1 bp3

.SUFFIXES:
.SUFFIXES: .cpp
.PHONY: all clean

all: $(MFEM_EXAMPLES)

# Quiet, color output
quiet ?= $($(1))

# Rules for building the examples
%: %.cpp $(CONFIG_MK) $(MFEM_LIB_FILE)
	$(call quiet,MFEM_CXX) $(CEED_FLAGS) $(MFEM_FLAGS) $(MFEM_DEF) $(abspath $<) -o $@ \
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
