#!/bin/bash

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

this_file="${BASH_SOURCE[0]}"
if [[ "${#BASH_ARGV[@]}" -ne "$#" ]]; then
   script_is_sourced="yes"
   exit_cmd=return
else
   script_is_sourced=""
   exit_cmd=exit
fi
test_file=""
backend_list="/cpu/self"
config=""
build=""
build_list=""
remove_list=""
run=""
post_process=""
profiler=""
num_proc_build=${num_proc_build:-""}
num_proc_run=${num_proc_run:-""}
num_proc_node=${num_proc_node:-""}
dry_run="" # empty string = NO
start_shell=""
verbose=""
cur_dir="$PWD"

function abspath()
{
   local outvar="$1" path="$2" cur_dir="$PWD"
   cd "$path" && path="$PWD" && cd "$cur_dir" && eval "$outvar=\"$path\""
}

abspath root_dir ".." || $exit_cmd 1
build_root="$root_dir/build"
configs_dir="$root_dir/benchmarks/machine-configs"

if [[ -t 1 ]]; then
   # ANSI color codes
   none=$'\E[0m'
   red=$'\E[0;31m'
   green=$'\E[0;32m'
   yellow=$'\E[0;33m'
   blue=$'\E[0;34m'
   bblue=$'\E[1;34m'
   magenta=$'\E[0;35m'
   cyan=$'\E[0;36m'
   clear="$(tput sgr0)"
fi

help_msg="
$this_file [options]

Options:
   -h|--help                print this usage information and exit
   -c|--ceed \"list\"         choose the libCEED backends to benchmark
   -u|--update              update the package sources before (re)building
   -r|--run <name>          run the tests in the script <name>
   -n|--num-proc \"list\"     total number of MPI tasks to use in the tests
   -p|--proc-node \"list\"    number of MPI tasks per node to use in the tests
  -pp|--post-process <name> post process the results using script <name>
   -d|--dry-run             show (but do not run) the commands for the tests
   -s|--shell               execute bash shell commands before running the test
   -v|--verbose             print additional messages
   -x                       enable script tracing with 'set -x'
   var=value                define shell variables; evaluated with 'eval'

This script builds and/or runs a set of tests using specified configuration
and compiler.

Example usage:
  $this_file  --run petsc-bp1.sh
"


function set_build_dirs()
{
   # Setup separate build directories inside $build_root based on $config
   # and $compiler.
   [[ -d "$build_root" ]] || mkdir -p "$build_root" || return 1
   OUT_DIR="$build_root"
   echo "Using OUT_DIR = $OUT_DIR"
   echo
}


function build_examples()
{
   cd ".."
   if [[ ! -f build/$test_required_packages ]]; then
     make benchmark-examples
   fi
   cd "$cur_dir"
}


function compose_mpi_run_command()
{
   mpi_run="${MPIEXEC:-mpirun} ${MPIEXEC_OPTS}"
   mpi_run+=" ${MPIEXEC_NP:--np} ${num_proc_run} ${MPIEXEC_POST_OPTS} $bind_sh"
   if [[ -n "$profiler" ]]; then
      mpi_run+=" $profiler"
   fi
}


function check_memory_req()
{
   local total_mem=""
   if [[ -n "$num_nodes" && -n "$memory_per_node" && \
         -n "$total_memory_required" ]]; then
      ((total_mem = memory_per_node * num_nodes))
      # echo "Total memory available: $total_mem GiB"
      # echo "Total memory required : $total_memory_required GiB"
      if [[ "$total_memory_required" -gt "$total_mem" ]]; then
         printf " *** Insufficient total memory: $total_mem GiB, "
         printf "this test requires: $total_memory_required GiB. "
         echo "Skipping test."
         return 1
      fi
   else
      echo " *** Warning: unable to check memory requirement."
   fi
   return 0
}


function quoted_echo()
{
   local arg= string=
   for arg; do
      if [[ -z "${arg##* *}" ]]; then
         string+=" \"${arg//\"/\\\"}\""
      else
         string+=" $arg"
      fi
   done
   printf "%s\n" "${string# }"
}


function set_num_nodes()
{
   if [[ -n "$num_proc_node" ]]; then
      ((num_proc_run % num_proc_node != 0)) && {
         echo "The total number of tasks ($num_proc_run) must be a multiple of"
         echo "the number of tasks per node ($num_proc_node). Stop."
         return 1
      }
      ((num_nodes = num_proc_run / num_proc_node))
   else
      num_proc_node="unknown number of"
      num_nodes=""
   fi
   echo "Running the tests using a total of $num_proc_run MPI tasks ..."
   echo "... with $num_proc_node tasks per node ..."
   echo
}


### Process command line parameters

while [ $# -gt 0 ]; do

case "$1" in
   -h|--help)
      # Echo usage information
      echo "$help_msg"
      $exit_cmd
      ;;
   -c|--ceed)
      shift
      [ $# -gt 0 ] || {
      echo "Missing \"list\" in --ceed \"list\""; $exit_cmd 1; }
      backend_list="$1"
      ;;
   -r|--run)
      run=on
      shift
      [ $# -gt 0 ] || { echo "Missing <name> in --run <name>"; $exit_cmd 1; }
      test_file="$1"
      [[ -r "$test_file" ]] || {
         echo "Test script not found: '$1'"; $exit_cmd 1
      }
      ;;
   -n|--num-proc)
      shift
      [ $# -gt 0 ] || {
      echo "Missing \"list\" in --num-proc \"list\""; $exit_cmd 1; }
      num_proc_run="$1"
      ;;
   -p|--proc-node)
      shift
      [ $# -gt 0 ] || {
      echo "Missing \"list\" in --proc-node \"list\""; $exit_cmd 1; }
      num_proc_node="$1"
      ;;
   -pp|--post-process)
      post_process=on
      shift
      [ $# -gt 0 ] || { echo "Missing <name> in --post-process <name>"; $exit_cmd 1; }
      pp_file="$1"
      [[ -r "$pp_file" ]] || {
         echo "Post process script not found: '$1'"; $exit_cmd 1
      }
      ;;
   -d|--dry-run)
      dry_run="quoted_echo"
      ;;
   -s|--shell)
      start_shell="yes"
      ;;
   -v|--verbose)
      verbose="yes"
      ;;
   -x)
      set -x
      ;;
   *=*)
      eval "$1" || { echo "Error evaluating argument: $1"; $exit_cmd 1; }
      ;;
   *)
      echo "Unknown option: '$1'"
      $exit_cmd 1
      ;;
esac

shift
done # while ...
# Done processing command line parameters


### Read configuration file
config="config.sh"
echo "Reading configuration $config ..."
. "$config" || $exit_cmd 1

abspath config_dir "$(dirname "$config")" || $exit_cmd 1
short_config="$(basename "$config")"
config="${config_dir}/${short_config}"
short_config="${short_config#config_}"
short_config="${short_config%.sh}"

num_proc_list=(${num_proc_run:-4})
num_proc_list_size=${#num_proc_list[@]}
num_proc_node_list=(${num_proc_node:-4})
num_proc_node_list_size=${#num_proc_node_list[@]}
(( num_proc_list_size != num_proc_node_list_size )) && {
   echo "
The size of the number-of-processors list (option --num-proc) must be the same
as the size of the number-of-processors-per-node list (option --proc-node)."
   echo
   $exit_cmd 1
}


### Loop over backends

for backend in $backend_list; do
(  ## Run each backend in its own environment

### Setup the environment based on $backend

echo "Using backend $backend ..."
short_backend=${backend//[\/]}
set_build_dirs || $exit_cmd 1

### Run the tests (building and running $test_file)

[ -n "$run" ] && {

cd "$cur_dir"
abspath test_dir "$(dirname "$test_file")" || $exit_cmd 1
test_basename="$(basename "$test_file")"
test_file="${test_dir}/${test_basename}"

[[ "$verbose" = "yes" ]] && {
   echo "Config file, $(basename "$config"):"
   echo "------------------------------------------------"
   cat $config
   echo "------------------------------------------------"
   echo

   echo "Test problem file, $test_basename:"
   echo "------------------------------------------------"
   cat $test_file
   echo "------------------------------------------------"
   echo
}

test_exe_dir="$OUT_DIR"

trap 'printf "\nScript interrupted.\n"; '$exit_cmd' 33' INT

## Source the test script file.
echo "Reading test file: $test_file"
echo
test_required_packages=""
. "$test_file" || $exit_cmd 1

## Setup output
output_file="${test_file%%.*}-$short_backend-output.txt"
rm -rf output_file

## Build files required by the test
echo "Files required by the test: $test_required_packages"
num_proc_build=${num_proc_build:-4}
echo "Building examples using $num_proc_build processors."
build_examples $test_required_packages || $exit_cmd 1
echo

## Loop over the number-of-processors list.
for (( num_proc_idx = 0; num_proc_idx < num_proc_list_size; num_proc_idx++ ))
do

num_proc_run="${num_proc_list[$num_proc_idx]}"
num_proc_node="${num_proc_node_list[$num_proc_idx]}"

set_num_nodes || $exit_cmd 1

if [[ "$start_shell" = "yes" ]]; then
   if [[ ! -t 1 ]]; then
      echo "Standard output is not a terminal. Stop."
      $exit_cmd 1
   fi
   echo "Reading shell commands, type 'c' to continue, 'exit' to stop ..."
   echo
   cd "$cur_dir"
   set -o emacs
   PS1='$ '
   [[ -r $HOME/.bashrc ]] && source $HOME/.bashrc
   HISTFILE="$root_dir/.bash_history"
   history -c
   history -r
   # bind '"\\C-i": menu-complete'
   alias c='break'
   while cwd="$PWD/" cwd="${cwd#${root_dir}/}" cwd="${cwd%/}" \
         prompt="[${cyan}benchmarks$none:$blue$cwd$clear]\$ " && \
         read -p "$prompt" -e line; do
      history -s "$line"
      history -w
      shopt -q -s expand_aliases
      eval "$line"
      shopt -q -u expand_aliases
   done
   [[ "${#line}" -eq 0 ]] && { echo; $exit_cmd 0; }
   shopt -q -u expand_aliases
   echo "Continuing ..."
fi

ceed=$backend
run_tests >> $output_file
echo

done ## End of loop over processor numbers

trap - INT

} ## run is on

### Post process the results

[[ -n "$post_process" ]] && {

. "$pp_file" || $exit_cmd 1

abspath test_dir "$(dirname "$pp_file")" || $exit_cmd 1
test_exe_dir="$OUT_DIR"

postprocess

} ## post-process on

$exit_cmd 0

) || {
   echo "Sub-shell for compiler '$backend' returned error code $?. Stop."
   $exit_cmd 1
}
done ## Loop over $compiler_list


$exit_cmd 0

