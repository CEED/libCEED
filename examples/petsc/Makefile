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

COMMON ?= ../../common.mk
-include $(COMMON)

# Note: PETSC_ARCH can be undefined or empty for installations which do not use
#       PETSC_ARCH - for example when using PETSc installed through Spack.
PETSc.pc := $(PETSC_DIR)/$(PETSC_ARCH)/lib/pkgconfig/PETSc.pc
CEED_DIR ?= ../..
ceed.pc := $(CEED_DIR)/lib/pkgconfig/ceed.pc

CC = $(call pkgconf, --variable=ccompiler $(PETSc.pc) $(ceed.pc))
CFLAGS = -std=c99 $(call pkgconf, --variable=cflags_extra $(PETSc.pc)) $(call pkgconf, --cflags-only-other $(PETSc.pc)) $(OPT)
CPPFLAGS = $(call pkgconf, --cflags-only-I $(PETSc.pc) $(ceed.pc))
LDFLAGS = $(call pkgconf, --libs-only-L --libs-only-other $(PETSc.pc) $(ceed.pc))
LDFLAGS += $(patsubst -L%, $(call pkgconf, --variable=ldflag_rpath $(PETSc.pc))%, $(call pkgconf, --libs-only-L $(PETSc.pc) $(ceed.pc)))
LDLIBS = $(call pkgconf, --libs-only-l $(PETSc.pc) $(ceed.pc)) -lm

area.c := area.c
area := $(area.c:%.c=%)

bps.c := bps.c
bps := $(bps.c:%.c=%)

bpsraw.c := bpsraw.c
bpsraw := $(bpsraw.c:%.c=%)

bpssphere.c := bpssphere.c
bpssphere := $(bpssphere.c:%.c=%)

multigrid.c := multigrid.c
multigrid := $(multigrid.c:%.c=%)

all: $(area) $(bps) $(bpsraw) $(bpssphere) $(multigrid) 
$(area): | $(PETSc.pc) $(ceed.pc)
$(bps): | $(PETSc.pc) $(ceed.pc)
$(bpsraw): | $(PETSc.pc) $(ceed.pc)
$(bpssphere): | $(PETSc.pc) $(ceed.pc)
$(multigrid): | $(PETSc.pc) $(ceed.pc)

# Quiet, color output
quiet ?= $($(1))

# Rules for building the examples
%: %.c
	$(call quiet,CC) $(CFLAGS) $(CPPFLAGS) $(LDFLAGS) $(CEED_LDFLAGS) $(abspath $<) -o $@ \
	  $(LDLIBS)

print: $(PETSc.pc) $(ceed.pc)
	$(info CC      : $(CC))
	$(info CFLAGS  : $(CFLAGS))
	$(info CPPFLAGS: $(CPPFLAGS))
	$(info LDFLAGS : $(LDFLAGS))
	$(info LDLIBS  : $(LDLIBS))
	@true

clean:
	$(RM) $(area) $(bps) $(bpsraw) $(bpssphere) $(multigrid)

$(PETSc.pc):
	$(if $(wildcard $@),,$(error \
	  PETSc config not found at $@. Please set PETSC_DIR and PETSC_ARCH))

.PHONY: all print clean

pkgconf = $(shell pkg-config $1 | sed -e 's/^"//g' -e 's/"$$//g')
