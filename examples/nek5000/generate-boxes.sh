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

min_elem=$1
max_elem=$2

###############################################################################
# DONT'T TOUCH WHAT FOLLOWS !!!
###############################################################################
if [ -z ${NEK5K_DIR} ]; then
  echo "ERROR: Must set NEK5K_DIR" 1>&2
  exit 1
fi

# Functions needed for creating box meshes
function xyz()
{
  prod=$1
  split=$((prod/3))

  nex=$split
  nez=$split
  ney=$split

  if [[ $((prod%3)) -ne 0 ]]; then
    nex=$((split + 1))
  fi
  if [[ $((prod%3)) -eq 2 ]]; then
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
  echo
  echo "Running genbox ..."
  if [ -z ${NEKTOOLS_DIR} ]; then
    echo "Required variable NEKTOOLS_DIR not found."
    exit 1
  fi

  printf "ttt.box\n" | $NEKTOOLS_DIR/genbox || return 1
  echo
  echo "Running genmap ..."
  printf "box\n.1\n" | $NEKTOOLS_DIR/genmap || return 1
  echo
  echo "Running reatore2 ..."
  printf "box\n$1\n" | $NEKTOOLS_DIR/reatore2 || return 1
  rm ttt.box
  rm box.rea
  rm box.tmp
  mv box.map $1.map
}

function generate_boxes()
{
  if [[ $# -ne 2 ]]; then
    echo "Error: should be called with two parameters. See syntax below."
    echo "Syntax: generate_boxes log_2(<min_elem>) log_2(<max_elem>)."
    echo "Example: generate-boxes 2 4"
    exit 1
  fi
  local min_elem=$1
  local max_elem=$2

  cd boxes
  # Run thorugh the box sizes
  for i in `seq $min_elem 1 $max_elem`
  do
    # Generate the boxes only if they have not
    # been generated before.
    if [[ ! -f b$i/b$i.map ]]; then
      # Set the number of elements in box file.
      xyz=$(xyz $i)
      nex=$( echo $xyz | cut -f 1 -d ' ' )
      ney=$( echo $xyz | cut -f 2 -d ' ' )
      nez=$( echo $xyz | cut -f 3 -d ' ' )

      echo "Generating a box with 2^$i elements : $nex x $ney x $nez ..."
      mkdir -p b$i
      sed "5s/.*/-$nex -$ney -$nez/" b.box > b$i/b$i.box
      cp b1e.rea b$i

      cd b$i
      genbb b$i &> log || {
        echo "Error generating box. See $PWD/log for details."
        return 1
      }
      cd ..
    fi
  done
  cd ..
}

generate_boxes $min_elem $max_elem
