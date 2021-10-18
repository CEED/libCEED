# Copyright (c) 2017-2018, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory. LLNL-CODE-734707.
# All Rights reserved. See files LICENSE and NOTICE for details.
#
# This file is part of CEED, a collection of benchmarks, miniapps, software
# libraries and APIs for efficient high-order finite element and spectral
# element discretizations for exascale applications. For more information and
# source code availability see http://github.com/ceed
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
CFLAGS = -std=c99 \
  $(call pkgconf, --variable=cflags_extra $(PETSc.pc)) \
  $(call pkgconf, --cflags-only-other $(PETSc.pc)) \
  $(OPT)
CPPFLAGS = $(call pkgconf, --cflags-only-I $(PETSc.pc) $(ceed.pc)) \
  $(call pkgconf, --variable=cflags_dep $(PETSc.pc))
LDFLAGS = $(call pkgconf, --libs-only-L --libs-only-other $(PETSc.pc) $(ceed.pc))
LDFLAGS += $(patsubst -L%, $(call pkgconf, --variable=ldflag_rpath $(PETSc.pc))%, $(call pkgconf, --libs-only-L $(PETSc.pc) $(ceed.pc)))
LDLIBS = $(call pkgconf, --libs-only-l $(PETSc.pc) $(ceed.pc)) -lm

# Enzyme-AD
ENZYME_LIB ?=
ENZYMEFLAGS = -Xclang -load -Xclang $(ENZYME_LIB) -O2 -fno-unroll-loops -mllvm -enzyme-print

ifneq ($(ENZYME_LIB),)
  CFLAGS += $(ENZYMEFLAGS)
else
  $(error Must export ENZYME_LIB=/path/to/ClangEnzyme-XX.so)
endif

OBJDIR := build
SRCDIR := src
PROBLEMDIR := problems

src.c := elasticity.c $(sort $(wildcard $(PROBLEMDIR)/*.c)) $(sort $(wildcard $(SRCDIR)/*.c))
src.o = $(src.c:%.c=$(OBJDIR)/%.o)

all: elasticity

elasticity: $(src.o) | $(PETSc.pc) $(ceed.pc)
	$(call quiet,LINK.o) $(CEED_LDFLAGS) $^ $(LOADLIBES) $(LDLIBS) -o $@

.SECONDEXPANSION: # to expand $$(@D)/.DIR
%/.DIR :
	@mkdir -p $(@D)
	@touch $@

# Quiet, color output
quiet ?= $($(1))

$(OBJDIR)/%.o : %.c | $$(@D)/.DIR
	$(call quiet,CC) $(CPPFLAGS) $(CFLAGS) -c -o $@ $(abspath $<)

# Rules for building the examples
#%: %.c

print: $(PETSc.pc) $(ceed.pc)
	$(info CC      : $(CC))
	$(info CFLAGS  : $(CFLAGS))
	$(info CPPFLAGS: $(CPPFLAGS))
	$(info LDFLAGS : $(LDFLAGS))
	$(info LDLIBS  : $(LDLIBS))
	@true

clean:
	$(RM) -r $(OBJDIR) elasticity *.vtu

$(PETSc.pc):
	$(if $(wildcard $@),,$(error \
	  PETSc config not found at $@. Please set PETSC_DIR and PETSC_ARCH))

.PHONY: all print clean

pkgconf = $(shell pkg-config $1 | sed -e 's/^"//g' -e 's/"$$//g')

# E = 2 * mu + (1 + nu)
tests-output/NH-strain.csv: args = -problem FSInitial-NH1 -E 2.8 -nu 0.4
tests-output/MR-strain.csv: args = -problem FSInitial-MR1 -mu_1 1 -mu_2 0.0 -nu .4
tests-output/MR-strain1.csv: args = -problem FSInitial-MR1 -mu_1 .5 -mu_2 0.5 -nu .4

tests-output/%.csv: elasticity
	./$<  $(args) -degree 2 -dm_plex_box_faces 3,3,3 -num_steps 15 -bc_clamp 6 -bc_traction 5 -bc_traction_5 0,0,-1 -snes_converged_reason -snes_linesearch_atol 1e-30 -strain_energy_monitor $@

-include $(src.o:%.o=%.d)
