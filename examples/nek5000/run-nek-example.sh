#!/bin/bash

if [[ $# -ne 4 ]]; then
  echo "Error: Number of inputs does not equal to 4. Please use the syntax below."
  echo "./run-nek-example <example_name> <#mpi_ranks> <rea_name> <rea_and_map_path>"
  echo "Example ./run-nek-example ex1 4 b10 ./boxes/b10"
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
