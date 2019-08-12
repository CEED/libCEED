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
# Script for Building and Running Nek5000 examples
###############################################################################
## Nek5000 path
#NEK5K_DIR=

## NEKTOOLS path
#NEK5K_TOOLS_DIR=

## CEED path
#CEED_DIR=

## Fortran compiler
#FC=

## C compiler
#CC=

###############################################################################
# DONT'T TOUCH WHAT FOLLOWS !!!
###############################################################################
# Set defaults for the parameters
: ${NEK5K_DIR:=`cd "../../../Nek5000"; pwd`}
: ${NEK5K_TOOLS_DIR:=`cd "${NEK5K_DIR}/bin"; pwd`}
: ${CEED_DIR:=`cd "../../"; pwd`}
: ${FC:="mpif77"}
: ${CC:="mpicc"}
: ${MPI:=1}

# Exit if being sourced
if [[ "${#BASH_ARGV[@]}" -ne "$#" ]]; then
   nek_exit_cmd=return
else
   nek_exit_cmd=exit
fi

# Read in parameter values
nek_examples=("bp1")
nek_spec=/cpu/self
nek_np=1
nek_box=
nek_clean="false"
nek_make="false"
nek_run="true"
nek_test="notest"
# Won't work if there is a symlink.
nek_box_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )/boxes"

# Set constants
nek_this_file="${BASH_SOURCE[0]}"
nek_help_msg="
$nek_this_file [options]

options:
   -h|-help     Print this usage information and exit
   -c|-ceed     Ceed backend to be used for the run (optional, default: /cpu/self)
   -e|-example  Example name (optional, default: bp1)
   -n|-np       Specify number of MPI ranks for the run (optional, default: 1)
   -t|-test     Run in test mode (not on by default)
   -b|-box      Box case in boxes sub-directory found along with this script (default: 2x2x2)
   -clean       clean the examples directory
   -m|-make     Make the examples

Example:
  Build examples with:
    ./nek-examples.sh -m -e \"bp1 bp3\"
  Run them with:
    ./nek-examples.sh -c /cpu/self -e \"bp1 bp3\" -n 4 -b 3
  Clean the examples directory with:
    ./nek-examples.sh -clean
"

nek_verbose="true"
nek_mpi="true"

while [ $# -gt 0 ]; do
  case "$1" in
    -h|-help)
       echo "$nek_help_msg"
       $nek_exit_cmd
       ;;
    -e|-example)
       shift
       nek_examples=($1)
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
       nek_verbose="false"
       nek_mpi="false"
       nek_test="test"
       ;;
    -clean)
      nek_clean="true"
      nek_run="false"
      ;;
    -m|-make)
      nek_make="true"
      nek_run="false"
      ;;
  esac
  shift
done

function make() {
  # Set flags
  FFLAGS="-g -std=legacy -I${CEED_DIR}/include"
  USR_LFLAGS="-g -L${CEED_DIR}/lib -Wl,-rpath,${CEED_DIR}/lib -lceed"

  # Build examples
  echo "Building examples:"

  # BP dir
  if [[ ! -d build ]]; then
    mkdir build
  fi

  # Copy makenek from NEK5K_DIR/bin/
  if [[ ! -f build/makenek ]]; then
    cp $NEK5K_DIR/bin/makenek build/
  fi

  # Copy BP files
  cp bps/bps.* build/

  # Copy SIZE file
  if [[ ! -f build/SIZE ]]; then
    cp SIZE.in build/SIZE
  fi

  # Change to build directory
  cd build

  # Attempt build
  CC=$CC FC=$FC MPI=$MPI NEK_SOURCE_ROOT="${NEK5K_DIR}" FFLAGS="$FFLAGS" \
    USR_LFLAGS="$USR_LFLAGS" ./makenek bps >> bps.build.log 2>&1

    # Check and report
  if [ ! -f ./nek5000 ]; then
    echo "  Building examples failed. See build/bps.build.log for details."
    cd ../..
    ${nek_exit_cmd} 1
  elif [ ${nek_verbose} = "true" ]; then
    mv nek5000 bps
    cd ../
    echo "  Built examples successfully. See build/bps.build.log for details."
  fi
}

# Function to clean
function clean() {
  if [ ${nek_verbose} = "true" ]; then
    echo "Cleaning ..."
  fi

  # Run clean
  if [[ -f ./build/makenek ]]; then
    cd build
    printf "y\n" | NEK_SOURCE_ROOT=${NEK5K_DIR} ./makenek clean 2>&1 >> /dev/null
    cd ..
  fi

  # Remove build dir
  rm -rf build

  # Remove BPs
  rm -f bps  *log* SESSION.NAME 2> /dev/null

  # Clean box dir
  find ${nek_box_dir} -type d -regex ".*/b[0-9]+" -exec rm -rf "{}" \; 2>/dev/null
}

# Functions needed for creating box meshes
function xyz()
{
  prod=$1
  split=$((prod/3))

  nex=$split
  nez=$split
  ney=$split

  if [ $((prod%3)) -ne 0 ]; then
    nex=$((split + 1))
  fi
  if [ $((prod%3)) -eq 2 ]; then
    ney=$((split + 1))
  fi

  nex=$((2**nex))
  ney=$((2**ney))
  nez=$((2**nez))

  echo "$nex $ney $nez"
}

function genbb()
{
  cp $1.box ttt.box
  if [ ${nek_verbose} = "true" ]; then
    echo "Running genbox ..."
  fi

  if [ -z ${NEK5K_TOOLS_DIR} ]; then
    echo "Required variable NEKTOOLS_DIR not found."
    ${nek_exit_cmd} 1
  fi

  printf "ttt.box\n" | $NEK5K_TOOLS_DIR/genbox 2>&1 1>>box.log || return 1

  if [ ${nek_verbose} = "true" ]; then
    echo "Running genmap ..."
  fi
  printf "box\n.1\n" | $NEK5K_TOOLS_DIR/genmap 2>&1 1>>box.log || return 1

  if [ ${nek_verbose} = "true" ]; then
    echo "Running reatore2 ..."
  fi
  printf "box\n$1\n" | $NEK5K_TOOLS_DIR/reatore2 2>&1 1>>box.log || return 1

  rm ttt.box 2>/dev/null
  rm box.rea 2>/dev/null
  rm box.tmp 2>/dev/null
  mv box.map $1.map 2>/dev/null
}

function generate_boxes()
{
  if [ $# -ne 2 ]; then
    echo "Error: should be called with two parameters. See syntax below."
    echo "Syntax: generate_boxes log_2(<min_elem>) log_2(<max_elem>)."
    echo "Example: generate-boxes 2 4"
    ${nek_exit_cmd} 1
  fi
  local min_elem=$1
  local max_elem=$2
  local pwd_=`pwd`

  mkdir -p ${nek_box_dir} && cd ${nek_box_dir}
  # Run thorugh the box sizes
  for i in `seq $min_elem 1 $max_elem`; do
    # Generate the boxes only if they have not
    # been generated before.
    if [ ! -f b$i/b$i.map ]; then
      # Set the number of elements in box file.
      xyz=$(xyz $i)
      nex=$( echo $xyz | cut -f 1 -d ' ' )
      ney=$( echo $xyz | cut -f 2 -d ' ' )
      nez=$( echo $xyz | cut -f 3 -d ' ' )

      mkdir -p b$i
      sed "5s/.*/-$nex -$ney -$nez/" ${CEED_DIR}/examples/nek/boxes/b.box > b$i/b$i.box
      cp ${CEED_DIR}/examples/nek/boxes/b1e.rea b$i/

      cd b$i
      genbb b$i
      genbb b$i &> log || {
        echo "Error generating box. See $PWD/log for details."
        return 1
      }
      cd ..
    fi
  done
  cd $pwd_
}

function run() {
  for nek_ex in "${nek_examples[@]}"; do
    if [ ${nek_verbose} = "true" ]; then
      echo "Running Nek5000 Example: $nek_ex"
    fi

    # Check for build
    if [ ! -f $bps ]; then
      echo "  Examples file does not exist. Build it with nek-examples.sh -m"
      ${nek_exit_cmd} 1
    fi

    # Generate boxes, if needed
    if [ ! -f ${nek_box_dir}/b${nek_box}/b${nek_box}.rea ] || \
  	 [ ! -f ${nek_box_dir}/b${nek_box}/b${nek_box}.map ]; then
       if [ -z ${nek_box} ]; then
         nek_box=3
       fi
      generate_boxes ${nek_box} ${nek_box}
    fi

    # Get CEED spec
    nek_spec_short=${nek_spec//[\/]}

    # Logging
    echo b${nek_box}                              > SESSION.NAME
    echo `cd ${nek_box_dir}/b${nek_box}; pwd`'/' >> SESSION.NAME
    rm -f logfile
    rm -f ioinfo
    mv ${nek_ex}.${nek_spec_short}.log.${nek_np}.b${nek_box} ${nek_ex}.${nek_spec_short}.log1.${nek_np}.b${nek_box} 2>/dev/null

    # Run example
    if [ ${nek_mpi} = "false" ]; then
        ./build/bps ${nek_ex} ${nek_spec} ${nek_test} > ${nek_ex}.${nek_spec_short}.log.${nek_np}.b${nek_box}
      wait $!
    else
      ${MPIEXEC:-mpiexec} -np ${nek_np} ./build/bps ${nek_ex} ${nek_spec} ${nek_test} > \
        ${nek_ex}.${nek_spec_short}.log.${nek_np}.b${nek_box}
      wait $!
    fi

    # Log output location
    if [ ${nek_verbose} = "true" ]; then
      echo "  Run finished. Output was written to ${nek_ex}.${nek_spec_short}.log.${nek_np}.b${nek_box}"
    fi

    # Check error
    if [[ $(grep 'ERROR IS TOO LARGE' ${nek_ex}.${nek_spec_short}.log*) ]]; then
      echo "ERROR IS TOO LARGE"
      ${nek_exit_cmd} 1
    elif [ ${nek_verbose} != "true" ]; then # Cleanup if test mode
      rm -f ${nek_ex}.${nek_spec_short}.log*
    fi
  done

  ${nek_exit_cmd} 0
}

if [ "${nek_clean}" = "true" ]; then
  clean
fi
if [ "${nek_make}" = "true" ]; then
  make
fi
if [ "${nek_run}" = "true" ]; then
  run
fi
${nek_exit_cmd} 0
