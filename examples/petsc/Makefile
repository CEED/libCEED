# Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and other CEED contributors
# All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
#
# SPDX-License-Identifier: BSD-2-Clause
#
# This file is part of CEED:  http://github.com/ceed

CONFIG ?= ../../config.mk
-include $(CONFIG)
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

all: area bps bpsraw bpssphere bpsswarm dmswarm multigrid

utils.c := $(sort $(wildcard $(SRCDIR)/*.c))
utils.o = $(utils.c:%.c=$(OBJDIR)/%.o)
libutils.a: $(utils.o)
	$(call quiet,AR) $(ARFLAGS) $@ $^

area.c := area.c
area.o = $(area.c:%.c=$(OBJDIR)/%.o)
area: $(area.o) libutils.a | $(PETSC_PC_PATH) $(CEED_PC_PATH)
	$(call quiet,LINK.o) $(CFLAGS) $(CPPFLAGS) $(LDFLAGS) $^ $(LDLIBS) -o $@

bps.c := bps.c
bps.o = $(bps.c:%.c=$(OBJDIR)/%.o)
bps: $(bps.o) libutils.a | $(PETSC_PC_PATH) $(CEED_PC_PATH)
	$(call quiet,LINK.o) $(CFLAGS) $(CPPFLAGS) $(LDFLAGS) $^ $(LDLIBS) -o $@

bpsraw.c := bpsraw.c
bpsraw.o = $(bpsraw.c:%.c=$(OBJDIR)/%.o)
bpsraw: $(bpsraw.o) libutils.a | $(PETSC_PC_PATH) $(CEED_PC_PATH)
	$(call quiet,LINK.o) $(CFLAGS) $(CPPFLAGS) $(LDFLAGS) $^ $(LDLIBS) -o $@

bpssphere.c := bpssphere.c
bpssphere.o = $(bpssphere.c:%.c=$(OBJDIR)/%.o)
bpssphere: $(bpssphere.o) libutils.a | $(PETSC_PC_PATH) $(CEED_PC_PATH)
	$(call quiet,LINK.o) $(CFLAGS) $(CPPFLAGS) $(LDFLAGS) $^ $(LDLIBS) -o $@

bpsswarm.c := bpsswarm.c
bpsswarm.o = $(bpsswarm.c:%.c=$(OBJDIR)/%.o)
bpsswarm: $(bpsswarm.o) libutils.a | $(PETSC_PC_PATH) $(CEED_PC_PATH)
	$(call quiet,LINK.o) $(CFLAGS) $(CPPFLAGS) $(LDFLAGS) $^ $(LDLIBS) -o $@


dmswarm.c := dmswarm.c
dmswarm.o = $(dmswarm.c:%.c=$(OBJDIR)/%.o)
dmswarm: $(dmswarm.o) libutils.a | $(PETSC_PC_PATH) $(CEED_PC_PATH)
	$(call quiet,LINK.o) $(CFLAGS) $(CPPFLAGS) $(LDFLAGS) $^ $(LDLIBS) -o $@

multigrid.c := multigrid.c
multigrid.o = $(multigrid.c:%.c=$(OBJDIR)/%.o)
multigrid: $(multigrid.o) libutils.a | $(PETSC_PC_PATH) $(CEED_PC_PATH)
	$(call quiet,LINK.o) $(CFLAGS) $(CPPFLAGS) $(LDFLAGS) $^ $(LDLIBS) -o $@

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

info: $(PETSC_PC_PATH) $(CEED_PC_PATH)
	$(info CC      : $(CC))
	$(info CFLAGS  : $(CFLAGS))
	$(info CPPFLAGS: $(CPPFLAGS))
	$(info LDFLAGS : $(LDFLAGS))
	$(info LDLIBS  : $(LDLIBS))
	@true

clean:
	$(RM) -r $(OBJDIR) *.vtu area bps bpsraw bpssphere bpsswarm dmswarm multigrid libutils.a

$(PETSC_PC_PATH):
	$(if $(wildcard $@),,$(error \
	  PETSc config not found at $@. Please set PETSC_DIR and PETSC_ARCH))

.PHONY: all print clean

-include $(src.o:%.o=%.d)
