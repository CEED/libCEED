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

#!/bin/bash

###############################################################################
# Make script for Nek5000 examples
###############################################################################
# Nek5000 path
NEK5K_DIR=`cd "../../../Nek5000"; pwd`

# list of examples to make
EXAMPLES=ex1

###############################################################################
# DONT'T TOUCH WHAT FOLLOWS !!!
###############################################################################
# Exit if being sourced
if [[ "${#BASH_SOURCE[@]}" -gt 1 ]]; then
  return 0
fi

# Build examples
for ex in $EXAMPLES; do
  echo "Building example: $ex ..."
  NEK5K_DIR="$NEK5K_DIR" ./makenek ex1 2>&1 >> $ex.build.log

  if [[ ! -f ./nek5000 ]]; then
    echo "  Building $ex failed. See $ex.build.log for details."
  else
    mv ./nek5000 $ex
    echo "  Built $ex successfully. See $ex.build.log for details."
  fi
done
