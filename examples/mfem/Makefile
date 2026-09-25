# Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and other CEED contributors
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
ifneq ($(wildcard $(CEED_DIR)/lib/libceed.a),)
  pc_flags := --static
endif

CEED_PC = $(abspath $(CEED_DIR))/lib/pkgconfig/ceed.pc
CEED_FLAGS = $(call pkgconf, $(pc_flags) --cflags $(CEED_PC)) $(call pkgconf, $(pc_flags) --variable=cflags_extra $(CEED_PC))
CEED_LIBS = $(call pkgconf, $(pc_flags) --libs-only-l $(CEED_PC))
CEED_LDPATHS = $(call pkgconf, $(pc_flags) --libs-only-L $(CEED_PC))
CEED_LDFLAGS = $(CEED_LDPATHS) $(CEED_LDPATHS:-L%=-Wl,-rpath,%) \
	$(call pkgconf, $(pc_flags) --libs-only-other $(CEED_PC))

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
	$(call quiet,MFEM_CXX) $(CEED_FLAGS) $(CEED_LDFLAGS) $(MFEM_FLAGS) $(MFEM_DEF) $(abspath $<) -o $@ \
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
