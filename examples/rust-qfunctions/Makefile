# Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and other CEED contributors
# All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
#
# SPDX-License-Identifier: BSD-2-Clause
#
# This file is part of CEED:  http://github.com/ceed

OPT ?= -O -g

# Ceed directory
CEED_DIR ?= ../..
SED ?= sed
PKGCONF ?= pkgconf
pkgconf = $(shell env PKG_CONFIG_PATH=$(PKG_CONFIG_PATH) $(PKGCONF) $1 | $(SED) -e 's/^"//g' -e 's/"$$//g')

ifneq ($(wildcard $(CEED_DIR)/lib/pkgconfig/ceed.pc),)
  export PKG_CONFIG_PATH := $(abspath $(CEED_DIR))/lib/pkgconfig$(if $(PKG_CONFIG_PATH),:$(PKG_CONFIG_PATH),)
endif
CEED_PC = ceed
ifneq ($(wildcard $(call pkgconf, --variable=prefix $(CEED_PC))/lib/libceed.a),)
  pc_flags := --static
endif
CEED_FLAGS = $(call pkgconf, --cflags $(CEED_PC)) $(call pkgconf, --variable=cflags_extra $(CEED_PC))
CEED_LDPATHS = $(call pkgconf, $(pc_flags) --libs-only-L $(CEED_PC))
CEED_LIBS = $(call pkgconf, -$(pc_flags) --libs-only-l $(CEED_PC))
CEED_LDFLAGS = $(CEED_LDPATHS) $(CEED_LDPATHS:-L%=-Wl,-rpath,%) $(CEED_LIBS) \
	$(call pkgconf, $(pc_flags) --libs-only-other $(CEED_PC))

CEED_LIBS += -lm

EXAMPLES.c = $(wildcard ex*.c)
EXAMPLES = $(EXAMPLES.c:%.c=%)

.SUFFIXES:
.SUFFIXES: .c
.PHONY: all clean

all: $(EXAMPLES)

# Remove built-in rules
%: %.c

# Special build rule for example 1 (rust)
ex1-volume: ex1-volume.c
	cargo +nightly build --release --manifest-path ex1-volume-rs/Cargo.toml --config ex1-volume-rs/.cargo/config.toml
	$(LINK.c) $(CEED_FLAGS) $(CEED_LDFLAGS) $(abspath $<) -o $@ $(CEED_LIBS) -L$(CEED_DIR)/examples/rust-qfunctions/ex1-volume-rs/target/release -lex1_volume_rs

clean:
	rm -f *~ $(EXAMPLES)
	rm -rf temp/
	rm -rf *.dSYM *.TVD.*breakpoints
