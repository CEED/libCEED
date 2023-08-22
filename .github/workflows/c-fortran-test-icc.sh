#!/bin/bash

# setup oneapi env
source /opt/intel/oneapi/setvars.sh

# run tests
export CC=icc CXX=icc FC=ifort
export OPENMP=1
make info
make -j2
PROVE_OPTS=-v make prove -j2
