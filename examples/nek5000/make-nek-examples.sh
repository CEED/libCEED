#!/bin/bash

# Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
# the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
# reserved. See files LICENSE and NOTICE for details.
#
# This file is part of CEED, a collection of benchmarks, miniapps, software
# libraries and APIs for efficient high-order finite element and spectral
# element discretizations for exascale applications. For more information and
# source code availability see http://github.com/ceed.
#
# The CEED research is supported by the Exascale Computing Project (17-SC-20-SC)
# a collaborative effort of two U.S. Department of Energy organizations (Office
# of Science and the National Nuclear Security Administration) responsible for
# the planning and preparation of a capable exascale ecosystem, including
# software, applications, hardware, advanced system engineering and early
# testbed platforms, in support of the nation's exascale computing imperative.

###############################################################################
# Make script for Nek5000 examples
###############################################################################
## Nek5000 path
#NEK5K_DIR=

## CEED path
#CEED_DIR=

## Fortran compiler
#FC=

## C compiler
#CC=

## list of examples to make
EXAMPLES=(bp1 bp3)

###############################################################################
# DONT'T TOUCH WHAT FOLLOWS !!!
###############################################################################

# See if its just cleaning and if yes, clean and exit
if [[ "$#" -eq 1 && "$1" -eq "clean" ]]; then
  if [[ -f ./makenek ]]; then
    printf "y\n" | ./makenek clean 2>&1 >> /dev/null
  fi
  rm makenek* SESSION.NAME 2> /dev/null
  for i in `seq 1 6`; do
    rm -f bp$i bp$i.f bp$i*log*              2> /dev/null
  done
  rm -f build.log
  rm -f .state
  find ./boxes -type d -regex ".*/b[0-9]+" -exec rm -rf "{}" \; 2>/dev/null
  exit 0
fi

# Set defaults for the parameters
: ${NEK5K_DIR:=`cd "../../../Nek5000"; pwd`}
: ${CEED_DIR:=`cd "../../"; pwd`}
: ${FC:="mpif77"}
: ${CC:="mpicc"}

# Exit if being sourced
if [[ "${#BASH_SOURCE[@]}" -gt 1 ]]; then
  return 0
fi

# Copy makenek from NEK5K_DIR/bin/
cp $NEK5K_DIR/bin/makenek .

FFLAGS="-g -std=legacy -I${CEED_DIR}/include"
USR_LFLAGS="-g -L${CEED_DIR}/lib -Wl,-rpath,${CEED_DIR}/lib -lceed"

# Build examples
for ex in "${EXAMPLES[@]}"; do
  echo "Building example: $ex ..."

  # makenek appends generated lines in SIZE, which we don't want versioned
  # So we copy SIZE.in to SIZE and use that with Nek5000. Once copied,
  # user can reuse the SIZE file until we clean the examples directory.
  if [[ ! -f SIZE ]]; then
    cp SIZE.in SIZE
  fi

  CC=$CC FC=$FC NEK_SOURCE_ROOT="${NEK5K_DIR}" FFLAGS="$FFLAGS" \
    USR_LFLAGS="$USR_LFLAGS" ./makenek $ex >> $ex.build.log 2>&1

  if [[ ! -f ./nek5000 ]]; then
    echo "  Building $ex failed. See $ex.build.log for details."
  else
    mv ./nek5000 $ex
    echo "  Built $ex successfully. See $ex.build.log for details."
  fi
done
