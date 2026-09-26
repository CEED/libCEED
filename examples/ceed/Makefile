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
SED ?= sed
pkgconf   = $(shell pkg-config $1 | $(SED) -e 's/^"//g' -e 's/"$$//g')
ifneq ($(wildcard $(CEED_DIR)/lib/libceed.a),)
  pc_flags := --static
endif

CEED_PC = $(abspath $(CEED_DIR))/lib/pkgconfig/ceed.pc
CEED_FLAGS = $(call pkgconf, $(pc_flags) --cflags $(CEED_PC)) $(call pkgconf, $(pc_flags) --variable=cflags_extra $(CEED_PC))
CEED_LIBS = $(call pkgconf, $(pc_flags) --libs-only-l $(CEED_PC))
CEED_LDPATHS = $(call pkgconf, $(pc_flags) --libs-only-L $(CEED_PC))
CEED_LDFLAGS = $(CEED_LDPATHS) $(CEED_LDPATHS:-L%=-Wl,-rpath,%) \
	$(call pkgconf, $(pc_flags) --libs-only-other $(CEED_PC))
LDFLAGS ?=
LDFLAGS += $(CEED_LDFLAGS)

ifeq (,$(filter-out undefined default,$(origin CC)))
  CC = gcc
endif
ifeq (,$(filter-out undefined default,$(origin LINK)))
  LINK = $(CC)
endif

EXAMPLES.c = $(wildcard ex*.c)
EXAMPLES = $(EXAMPLES.c:%.c=%)
CEED_LIBS += -lm

.SUFFIXES:
.SUFFIXES: .c
.PHONY: all clean

all: $(EXAMPLES)

# Remove built-in rules
%: %.c

# Rules for building the examples
%: %.c
	$(call quiet,LINK.c) -DCEED_EXAMPLES_SOURCE_ROOT="\"$(abspath .)/\"" $(CEED_FLAGS) $(CEED_LDFLAGS) $< -o $@ $(CEED_LIBS)

clean:
	rm -f *~ $(EXAMPLES)
	rm -rf *.dSYM *.TVD.*breakpoints
