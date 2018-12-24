# Copyright (c) 2017-2018, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory. LLNL-CODE-734707.
# All Rights reserved. See files LICENSE and NOTICE for details.
#
# This file is part of CEED, a collection of benchmarks, miniapps, software
# libraries and APIs for efficient high-order finite element and spectral
# element discretizations for exascale applications. For more information and
# source code availability see http://github.com/ceed.
#
# The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
# a collaborative effort of two U.S. Department of Energy organizations (Office
# of Science and the National Nuclear Security Administration) responsible for
# the planning and preparation of a capable exascale ecosystem, including
# software, applications, hardware, advanced system engineering and early
# testbed platforms, in support of the nation's exascale computing imperative.


# --- Linux ---

# Number of processors to use for building packages and tests:
num_proc_build=8
# Default number of processors and processors per node for running tests:
num_proc_run=${num_proc_run:-4}
num_proc_node=${num_proc_run}
# Total memory per node:
memory_per_node=8

# Optional (default): MPIEXEC (mpirun), MPIEXEC_OPTS (), MPIEXEC_NP (-np)


# --- Mac ---

# Number of processors to use for building packages and tests:
#num_proc_build=${num_proc_build:-$num_proc_detect}
# Default number of processors and processors per node for running tests:
#num_proc_run=${num_proc_run:-2}
#num_proc_node=${num_proc_run}
# Total memory per node:
#memory_per_node=8

# Optional (default): MPIEXEC (mpirun), MPIEXEC_OPTS (), MPIEXEC_NP (-np)


# --- Pascal ---
# Configuration for LLNL's Pascal system

# Number of processors to use for building packages and tests:
#num_proc_build=${num_proc_build:-36}
# Default number of processors and processors per node for running tests:
#num_proc_run=${num_proc_run:-36}
#num_proc_node=${num_proc_node:-36}
# Total memory per node:
#memory_per_node=256
# node_virt_mem_lim=

# Optional (default): MPIEXEC (mpirun), MPIEXEC_OPTS (), MPIEXEC_NP (-np)
#if [[ -z "${SLURM_JOB_CPUS_PER_NODE}" ]]; then
#   local account="${account:-ceed}"
#   local partition="${partition:-pbatch}"
#   MPIEXEC_OPTS="-A ${account} -p ${partition}"
#   MPIEXEC_OPTS+=" --ntasks-per-node $num_proc_node"
#   if [[ "$num_proc_node" -gt "36" ]]; then
#      MPIEXEC_OPTS+=" --overcommit"
#   fi
#else
#   local job_num_nodes=$SLURM_NNODES
#   if (( job_num_nodes < num_nodes )); then
#      echo "Insufficient number of nodes in the job allocation:"
#      echo "   ($job_num_nodes < $num_nodes)"
#      exit 1
#   fi
#   MPIEXEC_OPTS="--ntasks-per-node $num_proc_node"
#   if [[ "$num_proc_node" -gt "36" ]]; then
#      MPIEXEC_OPTS+=" --overcommit"
#   fi
#fi
#MPIEXEC=srun
#MPIEXEC_NP=-n


# --- Quartz ---
# Configuration for LLNL's Quartz system

# Number of processors to use for building packages and tests:
#num_proc_build=${num_proc_build:-36}
# Default number of processors and processors per node for running tests:
#num_proc_run=${num_proc_run:-36}
#num_proc_node=${num_proc_node:-36}
# Total memory per node:
#memory_per_node=128
# node_virt_mem_lim=

# Optional (default): MPIEXEC (mpirun), MPIEXEC_OPTS (), MPIEXEC_NP (-np)
#local account="${account:-ceed}"
#local partition="${partition:-pdebug}"
#MPIEXEC_OPTS="-A ${account} -p ${partition}"
#MPIEXEC_OPTS+=" --ntasks-per-node $num_proc_node"
#if [[ "$num_proc_node" -gt "36" ]]; then
#   MPIEXEC_OPTS+=" --overcommit"
#fi
#MPIEXEC=srun
#MPIEXEC_NP=-n


# --- Ray ---

# Number of processors to use for building packages and tests:
#num_proc_build=${num_proc_build:-40}
# Default number of processors and processors per node for running tests:
#num_proc_run=${num_proc_run:-20}
#num_proc_node=${num_proc_node:-20}
# Total memory per node:
#memory_per_node=256

# Optional (default): MPIEXEC (mpirun), MPIEXEC_OPTS (), MPIEXEC_NP (-np)
#local BSUB_OPTS=
#MPIEXEC_OPTS="-npernode $num_proc_node"
#MPIEXEC_OPTS+=" -prot"
#MPIEXEC_OPTS+=" -gpu"
#if [ "$num_proc_node" -gt "20" ]; then
#   MPIEXEC_OPTS="-oversubscribe $MPIEXEC_OPTS"
#fi
# Autodetect if running inside a job
#if [[ -n "${LSB_JOBID}" ]]; then
#   bind_sh="mpibind"
#   MPIEXEC="mpirun"
#   MPIEXEC_NP="-n"
#else
#   BSUB_OPTS="-q pbatch -G guests -n $((num_nodes*20)) -I"
#   MPIEXEC_OPTS="$BSUB_OPTS mpirun $MPIEXEC_OPTS"
#   bind_sh="mpibind"
#   MPIEXEC="bsub"
#   MPIEXEC_NP="-n"
#fi


# --- Sierra ---

# Number of processors to use for building packages and tests:
#num_proc_build=${num_proc_build:-16}
# Default number of processors and processors per node for running tests:
#num_proc_run=${num_proc_run:-4}
#num_proc_node=${num_proc_node:-4}
# Total memory per node:
#memory_per_node=256

# Optional (default): MPIEXEC (mpirun), MPIEXEC_OPTS (), MPIEXEC_NP (-np)
#if [[ -n "$no_gpu" ]]; then
#   bind_sh=
#   LRUN_GPU_OPT=
#else
#   bind_sh=${OUT_DIR}/bin/cvd.sh
#   LRUN_GPU_OPT="-M -gpu"
#fi
#if [[ -n "$bind_sh" ]] && [[ ! -e "$bind_sh" ||
#                             "$config" -nt "$bind_sh" ]]; then
#   echo "Creating/updating $bind_sh ..."
#   mkdir -p "${OUT_DIR}/bin"
#   {
#      cat <<'EOF'
##!/bin/bash

#NGPUS=4
#if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
#  if [ "$OMPI_COMM_WORLD_RANK" == 0 ]; then
#    echo "CUDA_VISIBLE_DEVICES is not set."
#  fi
#  exit 1
#fi

#cvds=($CUDA_VISIBLE_DEVICES)
#cvd=${cvds[0]}
#for i in ${cvds[@]:1}; do
#  cvd=$cvd,$i
#done
#cvds_pat=" ${cvds[@]} "
#for ((i=0; i<NGPUS; i++)); do
#  if [ -n "${cvds_pat##* $i *}" ]; then
#    cvd=$cvd,$i
#  fi
#done
# printf "[$HOSTNAME,rank=$OMPI_COMM_WORLD_RANK]: "
# printf "CUDA_VISIBLE_DEVICES: '$CUDA_VISIBLE_DEVICES' --> '$cvd'\n"
#export CUDA_VISIBLE_DEVICES=$cvd
#exec "$@"
#EOF
#   } > "$bind_sh"
#   chmod u+x "$bind_sh"
#fi

# Autodetect if running inside a job
#if [[ -z "${LSB_JOBID}" ]]; then
#   local account="${account:-guests}"
#   local partition="${partition:-pbatch}"
   # Time limit in minutes
#   local TIME_LIMIT=${time_limit:-30}
#   local BSUB_OPTS="-q ${partition} -G ${account} -nnodes ${num_nodes}"
#   BSUB_OPTS="${BSUB_OPTS} -W ${TIME_LIMIT}"
#   MPIEXEC_OPTS="-N $num_nodes"
#   MPIEXEC_OPTS="${BSUB_OPTS} lrun ${MPIEXEC_OPTS}"
#   MPIEXEC_POST_OPTS="${LRUN_GPU_OPT}"
#   MPIEXEC="bsub"
#   MPIEXEC_NP="-n"
#else
      # LSB_DJOB_NUMPROC=num
      #   - The number of processors (slots) allocated to the job.
      # LSB_MCPU_HOSTS="hostA num_processors1 hostB num_processors2..."
#   local job_nodes_list=($LSB_MCPU_HOSTS)
#   local job_num_nodes=$(( ${#job_nodes_list[@]} / 2 ))
#   if (( job_num_nodes < num_nodes )); then
#      echo "Insufficient number of nodes in the job allocation:"
#      echo "   ($job_num_nodes < $num_nodes)"
#      exit 1
#   fi
#   if (( LSB_DJOB_NUMPROC < num_proc_run )); then
#      echo "Insufficient number of processors in the job allocation:"
#      echo "   ($LSB_DJOB_NUMPROC < $num_proc_run)"
#      exit 1
#   fi
#   MPIEXEC_OPTS="-N $num_nodes"
#   MPIEXEC_POST_OPTS="${LRUN_GPU_OPT}"
#   MPIEXEC="lrun"
#   MPIEXEC_NP="-n"
#fi


# --- Vulcan ---

# Number of processors to use for building packages and tests:
#num_proc_build=${num_proc_build:-16}
# Default number of processors and processors per node for running tests:
#num_proc_run=${num_proc_run:-16}
#num_proc_node=${num_proc_node:-16}
# Total memory per node:
#memory_per_node=16
#node_virt_mem_lim=16

# Optional (default): MPIEXEC (mpirun), MPIEXEC_OPTS (), MPIEXEC_NP (-np)
#local account="${account:-ceed}"
#local partition="${partition:-pdebug}"
# pdebug (<= 1K nodes & <= 1h)
# psmall (<= 1K nodes & <= 12h)
# pbatch (> 1K nodes & <= 8K nodes & <= 12h)
#MPIEXEC_OPTS="-A ${account} -p ${partition}"
# MPIEXEC_OPTS="-A ceed -p psmall"
#MPIEXEC_OPTS+=" --ntasks-per-node $num_proc_node"
#if [[ "$num_proc_node" -gt "16" ]]; then
#   MPIEXEC_OPTS+=" --overcommit"
#fi
compose_mpi_run_command
MPIEXEC=srun
MPIEXEC_NP=-n
