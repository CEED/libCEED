# Copyright (c) 2017-2018, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory. LLNL-CODE-734707.
# All Rights reserved. See files LICENSE and NOTICE for details.
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

# Output using the 216-color rules mode
rule_file = $(notdir $(1))
rule_path = $(patsubst %/,%,$(dir $(1)))
last_path = $(notdir $(patsubst %/,%,$(dir $(1))))
ansicolor = $(shell echo $(call last_path,$(1)) | cksum | cut -b1-2 | xargs -IS expr 2 \* S + 17)
emacs_out = @printf "  %10s %s/%s\n" $(1) $(call rule_path,$(2)) $(call rule_file,$(2))
color_out = @if [ -t 1 ]; then \
				printf "  %10s \033[38;5;%d;1m%s\033[m/%s\n" \
					$(1) $(call ansicolor,$(2)) \
					$(call rule_path,$(2)) $(call rule_file,$(2)); else \
				printf "  %10s %s\n" $(1) $(2); fi
# if TERM=dumb, use it, otherwise switch to the term one
output = $(if $(TERM:dumb=),$(call color_out,$1,$2),$(call emacs_out,$1,$2))

# if V is set to non-nil, turn the verbose mode
quiet = $(if $(V),$($(1)),$(call output,$1,$@);$($(1)))

# make-4.3 allows string literals like "#include" in variables, but older versions need "\#include". Specifically, the following code:
#
# X := $(shell echo "#foo")
#
# works with make-4.3, but fails with previous versions:
#
# Makefile:1: *** unterminated call to function 'shell': missing ')'.  Stop.
#
# Older versions work if you spell it "\#foo", but 4.3 will include the backslash. We define $(HASH), which works consistently across versions.
HASH := \#
