# Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
# the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
# reserved. See files LICENSE and NOTICE for details.
#
# This file is part of CEED, a collection of benchmarks, miniapps, software
# libraries and APIs for efficient high-order finite element and spectral
# element discretizations for exascale applications. For more information and
# source code availability see http://github.com/ceed
#
# The CEED research is supported by the Exascale Computing Project (17-SC-20-SC)
# a collaborative effort of two U.S. Department of Energy organizations (Office
# of Science and the National Nuclear Security Administration) responsible for
# the planning and preparation of a capable exascale ecosystem, including
# software, applications, hardware, advanced system engineering and early
# testbed platforms, in support of the nation's exascale computing imperative.

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

# Address Sanitizer Setup
# ASAN must be left empty if you don't want to use it
ASAN ?=
AFLAGS ?= -fsanitize=address
# Also: -fsanitize=undefined -fno-omit-frame-pointer
CFLAGS += $(if $(ASAN),$(AFLAGS))
FFLAGS += $(if $(ASAN),$(AFLAGS))
LDFLAGS += $(if $(ASAN),$(AFLAGS))
CPPFLAGS += -I./include

# Source Files
OBJDIR := build
SRCDIR := src
PROBLEMDIR := problems

src.c := navierstokes.c $(sort $(wildcard $(PROBLEMDIR)/*.c)) $(sort $(wildcard $(SRCDIR)/*.c))
src.o = $(src.c:%.c=$(OBJDIR)/%.o)

all: navierstokes

navierstokes: $(src.o) | $(PETSC_PC_PATH) $(CEED_PC_PATH)
	$(LINK.o) $(LDFLAGS) $^ $(LOADLIBES) $(LDLIBS) -o $@

.SECONDEXPANSION: # to expand $$(@D)/.DIR
%/.DIR :
	@mkdir -p $(@D)
	@touch $@

# Quiet, color output
quiet ?= $($(1))

$(OBJDIR)/%.o : %.c  Makefile | $$(@D)/.DIR
	$(call quiet,CC) $(CPPFLAGS) $(CFLAGS) -c -o $@ $(abspath $<)

info: $(PETSc.pc) $(ceed.pc)
	$(info CC      : $(CC))
	$(info CFLAGS  : $(CFLAGS))
	$(info CPPFLAGS: $(CPPFLAGS))
	$(info LDFLAGS : $(LDFLAGS))
	$(info LDLIBS  : $(LDLIBS))
	$(info OPT     : $(OPT))
	@true

clean:
	$(RM) -r $(OBJDIR) navierstokes *.vtu *.bin* *.csv *.png

$(PETSC_PC_PATH):
	$(if $(wildcard $@),,$(error \
	  PETSc config not found. Please set PETSC_DIR and PETSC_ARCH))

.PHONY: all info clean

-include $(src.o:%.o=%.d)
