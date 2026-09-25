# Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and other CEED contributors.
# All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
#
# SPDX-License-Identifier: BSD-2-Clause
#
# This file is part of CEED:  http://github.com/ceed

COMMON ?= ../../common.mk
-include $(COMMON)

SED ?= sed
PKGCONF ?= pkgconf
pkgconf = $(shell env PKG_CONFIG_PATH=$(PKG_CONFIG_PATH) $(PKGCONF) $1 | $(SED) -e 's/^"//g' -e 's/"$$//g')

ifneq ($(wildcard $(PETSC_DIR)/$(PETSC_ARCH)/lib/pkgconfig/PETSc.pc),)
  export PKG_CONFIG_PATH := $(abspath $(PETSC_DIR))/$(PETSC_ARCH)/lib/pkgconfig$(if $(PKG_CONFIG_PATH),:$(PKG_CONFIG_PATH),)
endif
PETSc.pc = PETSc

CEED_DIR ?= ../..
ifneq ($(wildcard $(CEED_DIR)/lib/pkgconfig/ceed.pc),)
  export PKG_CONFIG_PATH := $(abspath $(CEED_DIR))/lib/pkgconfig$(if $(PKG_CONFIG_PATH),:$(PKG_CONFIG_PATH),)
endif
ceed.pc = ceed
ceed_pc_flags =
ifneq ($(wildcard $(call pkgconf, --variable=prefix $(ceed.pc))/lib/libceed.a),)
  ceed_pc_flags := --static
endif

PETSC_PC_PATH = $(call pkgconf, --path $(PETSc.pc))
CEED_PC_PATH = $(call pkgconf, --path $(ceed.pc))

CC = $(call pkgconf, --variable=ccompiler $(PETSc.pc))
CFLAGS = -std=c11 \
  $(call pkgconf, --variable=cflags_extra $(PETSc.pc)) \
  $(call pkgconf, --cflags-only-other $(PETSc.pc)) \
  $(OPT) $(OPT_EXAMPLES)
CPPFLAGS = $(call pkgconf, --cflags-only-I $(PETSc.pc)) \
  $(call pkgconf, --cflags-only-I $(ceed_pc_flags) $(ceed.pc)) \
  $(call pkgconf, --variable=cflags_dep $(PETSc.pc))
LDFLAGS = $(call pkgconf, --libs-only-L --libs-only-other $(PETSc.pc)) \
  $(call pkgconf, --libs-only-L --libs-only-other $(ceed_pc_flags) $(ceed.pc))
LDFLAGS += $(patsubst -L%, $(call pkgconf, --variable=ldflag_rpath $(PETSc.pc))%, $(call pkgconf, --libs-only-L $(PETSc.pc))) \
  $(patsubst -L%, $(call pkgconf, --variable=ldflag_rpath $(PETSc.pc))%, $(call pkgconf, --libs-only-L $(ceed_pc_flags) $(ceed.pc)))
LDLIBS = $(call pkgconf, --libs-only-l $(PETSc.pc)) $(call pkgconf, --libs-only-l $(ceed_pc_flags) $(ceed.pc)) -lm

OBJDIR := build
SRCDIR := src
PROBLEMDIR := problems

src.c := elasticity.c $(sort $(wildcard $(PROBLEMDIR)/*.c)) $(sort $(wildcard $(SRCDIR)/*.c))
src.o = $(src.c:%.c=$(OBJDIR)/%.o)

all: elasticity

elasticity: $(src.o) | $(PETSC_PC_PATH) $(CEED_PC_PATH)
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

print: $(PETSC_PC_PATH) $(CEED_PC_PATH)
	$(info CC      : $(CC))
	$(info CFLAGS  : $(CFLAGS))
	$(info CPPFLAGS: $(CPPFLAGS))
	$(info LDFLAGS : $(LDFLAGS))
	$(info LDLIBS  : $(LDLIBS))
	@true

clean:
	$(RM) -r $(OBJDIR) elasticity *.vtu

$(PETSC_PC_PATH):
	$(if $(wildcard $@),,$(error \
	  PETSc config not found at $@. Please set PETSC_DIR and PETSC_ARCH))

.PHONY: all print clean

# E = 2 * mu + (1 + nu)
tests-output/NH-strain.csv: args = -problem FSInitial-NH1 -E 2.8 -nu 0.4
tests-output/MR-strain.csv: args = -problem FSInitial-MR1 -mu_1 1 -mu_2 0.0 -nu .4
tests-output/MR-strain1.csv: args = -problem FSInitial-MR1 -mu_1 .5 -mu_2 0.5 -nu .4

tests-output/%.csv: elasticity
	./$<  $(args) -degree 2 -dm_plex_box_faces 3,3,3 -num_steps 15 -bc_clamp 6 -bc_traction 5 -bc_traction_5 0,0,-1 -snes_converged_reason -snes_linesearch_atol 1e-30 -strain_energy_monitor $@

-include $(src.o:%.o=%.d)
