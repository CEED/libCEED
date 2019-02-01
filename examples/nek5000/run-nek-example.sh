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

## Set libCEED directory
#CEED_DIR=

###############################################################################
# DONT'T TOUCH WHAT FOLLOWS !!!
###############################################################################
if [[ "${#BASH_ARGV[@]}" -ne "$#" ]]; then
   NEK_EXIT_CMD=return
else
   NEK_EXIT_CMD=exit
fi

# Set defaults for the parameters
: ${CEED_DIR:=`cd ../../; pwd`}
nek_examples=("bp1" "bp3")
nek_spec=/cpu/self
nek_np=1
nek_box=
NEK_BOX_DIR=./boxes

# Set constants
NEK_THIS_FILE="${BASH_SOURCE[0]}"
NEK_HELP_MSG="
$NEK_THIS_FILE [options]

options:
   -h|-help     Print this usage information and exit
   -c|-ceed     Ceed backend to be used for the run (optional, default: /cpu/self)
   -e|-example  Example name (optional, default: bp1)
   -n|-np       Specify number of MPI ranks for the run (optional, default: 1)
   -t|-test     Run in test mode
   -b|-box      Specify the box geometry to be found in ./boxes/ directory (Mandatory)

Example:
  ./run-nek-example -c /cpu/self -e bp1 -n 4 -b 3
"

# Read in parameter values
nek_test="notest"
nek_test_rst="PASS"
while [ $# -gt 0 ]; do
  case "$1" in
    -h|-help)
       echo "$NEK_HELP_MSG"
       $NEK_EXIT_CMD
       ;;
    -e|-example)
       shift
       nek_examples=("$1")
       ;;
    -c|-ceed)
       shift
       nek_spec="$1"
       ;;
    -n|-np)
       shift
       nek_np="$1"
       ;;
    -b|-box)
       shift
       nek_box="$1"
       ;;
    -t|-test)
       nek_test="test"
       ;;
  esac
  shift
done

if [[ -z "${nek_box}" ]]; then
    echo "$0: You must specify option -b <number of boxes>."
    echo "$NEK_HELP_MSG"
    ${NEK_EXIT_CMD} 1
fi

for nek_ex in "${nek_examples[@]}"; do
  if [ ${nek_test} == "test" ]; then
    nek_ex="build/"${nek_ex}
    NEK_BOX_DIR="build/boxes"
  else
    echo "Running Nek5000 Example: $nek_ex"    
  fi
  if [[ ! -f ${nek_ex} ]]; then
    echo "  Example ${nek_ex} does not exist. Build it with make-nek-examples.sh"
    ${NEK_EXIT_CMD} 1
  fi
  if [[ ! -f ${NEK_BOX_DIR}/b${nek_box}/b${nek_box}.rea || \
	  ! -f ${NEK_BOX_DIR}/b${nek_box}/b${nek_box}.map ]]; then
    ./generate-boxes.sh ${nek_box} ${nek_box}
  fi

  echo b${nek_box}                              > SESSION.NAME
  echo `cd ${NEK_BOX_DIR}/b${nek_box}; pwd`'/' >> SESSION.NAME
  rm -f logfile
  rm -f ioinfo
  mv ${nek_ex}.log.${nek_np}.b${nek_box} ${nek_ex}.log1.${nek_np}.b${nek_box} 2>/dev/null

  nek_spec_short=${nek_spec//[\/]}

  if [ ${nek_test} == "test" ]; then
      ./${nek_ex} ${nek_spec} ${nek_test} > ${nek_ex}.${nek_spec_short}.log.${nek_np}.b${nek_box}
    wait $!
  else
    ${MPIEXEC:-mpiexec} -np ${nek_np} ./${nek_ex} ${nek_spec} ${nek_test} > ${nek_ex}.${nek_spec_short}.log.${nek_np}.b${nek_box}
    wait $!
  fi

  if [ ! ${nek_test} == "test" ]; then
    echo "  Run finished. Output was written to ${nek_ex}.${nek_spec_short}.log.${nek_np}.b${nek_box}"
  fi
  if [ ${nek_test} == "test" ]; then
    if [[ $(grep "ERROR IS TOO LARGE" ${nek_ex}.${nek_spec_short}.log*) ]]; then
      nek_test_rst="FAIL"
    else
      rm -f ${nek_ex}.${nek_spec_short}.log*
    fi
  fi
done

exit 0
