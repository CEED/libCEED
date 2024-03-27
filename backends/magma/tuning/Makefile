# Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors
# All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
#
# SPDX-License-Identifier: BSD-2-Clause
#
# This file is part of CEED:  http://github.com/ceed

COMMON ?= ../../../common.mk
-include $(COMMON)

CEED_DIR ?= ../../..

CXXFLAGS = $(OPT)
CPPFLAGS ?= -I$(CEED_DIR)/include
LDLIBS ?= -Wl,-rpath,$(abspath $(CEED_DIR)/lib) -L$(CEED_DIR)/lib -lceed -lm

# Verbose mode, V or VERBOSE
V ?= $(VERBOSE)

all: tuning

%: %.cpp $(ceed.pc)
	$(call quiet,CXX) $(CPPFLAGS) $(CXXFLAGS) $(abspath $<) -o $@ \
	  $(LDFLAGS) $(LDLIBS)

# Quiet, color output
quiet ?= $($(1))

print: $(ceed.pc)
	$(info CXX     : $(CXX))
	$(info CXXFLAGS: $(CXXFLAGS))
	$(info CPPFLAGS: $(CPPFLAGS))
	$(info LDFLAGS : $(LDFLAGS))
	$(info LDLIBS  : $(LDLIBS))
	@true

clean:
	$(RM) -r tuning output-nb-*.txt

.PHONY: all clean
