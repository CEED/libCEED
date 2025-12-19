# Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and other CEED contributors.
# All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
#
# SPDX-License-Identifier: BSD-2-Clause
#
# This file is part of CEED:  http://github.com/ceed

PYTHON ?= python3

clean:
	rm -rf build __pycache__ .pytest_cache *.so

setup:
	$(PYTHON) setup_qfunctions.py build

TEST_OPTS ?= --ceed /cpu/self/ref/serial
test: setup
	$(PYTHON) -m pytest ex_test.py $(TEST_OPTS)

.PHONY: clean setup test
