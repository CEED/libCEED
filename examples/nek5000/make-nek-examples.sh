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

## Fortran compiler
#FC=

## C compiler
#CC=

## list of examples to make
EXAMPLES=ex1

###############################################################################
# DONT'T TOUCH WHAT FOLLOWS !!!
###############################################################################
# Set defaults for the parameters
: ${NEK5K_DIR:=`cd "../../../Nek5000"; pwd`}
: ${FC:="mpif77"}
: ${CC:="mpicc"}

# See if its just cleaning and if yes, clean and exit
if [[ "$#" -eq 1 && "$1" -eq "clean" ]]; then
  if [[ -f ./makenek ]]; then
    printf "y\n" | ./makenek clean 2>&1 >> /dev/null
  fi
  rm makenek* ex1 ex1*log* SESSION.NAME 2> /dev/null
  find ./boxes -type d -regex ".*/b[0-9]+" -exec rm -rf "{}" \; 2>/dev/null
  exit
fi

# Exit if being sourced
if [[ "${#BASH_SOURCE[@]}" -gt 1 ]]; then
  return 0
fi

# Copy makenek from NEK5K_DIR/bin/
cp $NEK5K_DIR/bin/makenek .

sed -i.bak -e "s|^#FC=.*|FC=\"$FC\"|" \
    -e "s|^#CC=.*|CC=\"$CC\"|" \
    -e "s|^#SOURCE_ROOT=.*|SOURCE_ROOT=\"$NEK5K_DIR\"|" \
    -e "s|^#FFLAGS=.*|FFLAGS+=\"-I../../include\"|" \
    -e "s|^#USR_LFLAGS+=.*|USR_LFLAGS+=\"-L../../lib -lceed -fopenmp -L/$OCCA_DIR/lib -locca -lm -lcuda\"|" makenek


# Build examples
for ex in $EXAMPLES; do
  echo "Building example: $ex ..."

  # makenek appends generated lines in SIZE, which we don't want versioned
  # So we copy SIZE.in to SIZE and use that with Nek5000. Once copied,
  # user can reuse the SIZE file till he clean the examples directory.
  if [[ ! -f SIZE ]]; then
    cp SIZE.in SIZE
  fi

  ./makenek ex1 2>&1 >> $ex.build.log

  if [[ ! -f ./nek5000 ]]; then
    echo "  Building $ex failed. See $ex.build.log for details."
  else
    mv ./nek5000 $ex
    echo "  Built $ex successfully. See $ex.build.log for details."
  fi
done
