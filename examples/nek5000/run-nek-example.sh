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
# DONT'T TOUCH WHAT FOLLOWS !!!
###############################################################################
if [[ $# -ne 4 ]]; then
  echo "Error: Number of inputs does not equal to 4. Please use the syntax below."
  echo "./run-nek-example <example_name> <#mpi_ranks> <rea_name> <rea_and_map_path>"
  echo "Example ./run-nek-example bp1 4 b3 ./boxes/b3"
  exit 1
fi

export LD_LIBRARY_PATH=`cd ../../lib; pwd`:${LD_LIBRARY_PATH}

ex=$1
np=$2
rea=$3
reapath=$4

if [[ ! -f $ex ]]; then
  echo "Example $ex does not exist. Build it with make-nek-examples.sh"
  exit 1
fi
if [[ ! -f $reapath/$rea.rea || ! -f $reapath/$rea.map ]]; then
  echo ".rea/.map file $reapath/$rea does not exist."
  exit 1
fi

echo $rea                   >  SESSION.NAME
echo `cd $reapath; pwd`'/' >>  SESSION.NAME
rm -f logfile
rm -f ioinfo
mv $ex.log.$np.$rea $ex.log1.$np.$rea 2>/dev/null

mpiexec -np $np ./$ex > $ex.log.$np.$rea
wait $!

echo "Run finished. Output was written to $ex.log.$np.$rea"
