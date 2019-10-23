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
bp_list="bp1 bp3"
run=""
num_proc_run=${num_proc_run:-""}
num_proc_node=${num_proc_node:-""}
dry_run="" # empty string = NO
start_shell=""
verbose=""
cur_dir="$PWD"

mpiexec="mpirun"
mpiexec_np="-np"
mpiexec_opts=""
mpiexec_post_opts=""
profiler=""

function abspath()
{
   local outvar="$1" path="$2" cur_dir="$PWD"
   cd "$path" && path="$PWD" && cd "$cur_dir" && eval "$outvar=\"$path\""
}

abspath root_dir ".." || $exit_cmd 1
build_root="$root_dir/build"

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
   -b|--bp \"list\"           choose the benchmark problems to run
   -c|--ceed \"list\"         choose the libCEED backends to benchmark
   -r|--run <name>          run the tests in the script <name>
   -n|--num-proc \"list\"     total number of MPI tasks to use in the tests
   -p|--proc-node \"list\"    number of MPI tasks per node to use in the tests
   -d|--dry-run             show (but do not run) the commands for the tests
   -s|--shell               execute bash shell commands before running the test
   -v|--verbose             print additional messages
   -x                       enable script tracing with 'set -x'
   var=value                define shell variables; evaluated with 'eval'

This script builds and runs a set of benchmarks for a list of specified
backends.

Example usage:
  $this_file  --run petsc-bpsraw.sh
"


function build_examples()
{
   for example; do
      # We require the examples to be already built because we do not know what
      # options to use when building the library + examples.
      if [ ! -e $build_root/$example ]; then
         echo "Error: example is not built: $example"
         return 1
      fi
   done
}


function compose_mpi_run_command()
{
   mpi_run="${mpiexec:-mpirun} ${mpiexec_opts}"
   mpi_run+=" ${mpiexec_np:--np} ${num_proc_run} ${mpiexec_post_opts}"
   if [[ -n "$profiler" ]]; then
      mpi_run+=" $profiler"
   fi
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
   echo "Running the tests using a total of $num_proc_run MPI tasks ..." | tee -a $output_file
   echo "... with $num_proc_node tasks per node ..." | tee -a $output_file
   echo | tee -a $output_file
}


### Process command line parameters

while [ $# -gt 0 ]; do

case "$1" in
   -h|--help)
      # Echo usage information
      echo "$help_msg"
      $exit_cmd
      ;;
   -b|--bp)
      shift
      [ $# -gt 0 ] || {
      echo "Missing \"list\" in --bp \"list\""; $exit_cmd 1; }
      bp_list="$1"
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

### Loop over BPs

for bp in $bp_list; do

### Loop over backends

for backend in $backend_list; do
(  ## Run each backend in its own environment

### Setup output
### Test name
cd "$cur_dir"
abspath test_dir "$(dirname "$test_file")" || $exit_cmd 1
test_basename="$(basename "$test_file")"
test_file="${test_dir}/${test_basename}"
### Backend name
short_backend=${backend//[\/]}
### Output file
output_file="${test_file%%.*}-$bp-$short_backend-output.txt"
rm -rf output_file

### Setup the environment based on $backend

echo
echo "Using backend $backend ..." | tee $output_file

### Run the tests (building and running $test_file)

[ -n "$run" ] && {

[[ "$verbose" = "yes" ]] && {
   echo "Test problem file, $test_basename:" | tee -a $output_file
   echo "------------------------------------------------" | tee -a $output_file
   cat $test_file | tee -a $output_file
   echo "------------------------------------------------" | tee -a $output_file
   echo | tee -a $output_file
}

test_exe_dir="$build_root"

trap 'printf "\nScript interrupted.\n"; '$exit_cmd' 33' INT

## Source the test script file.
echo "Reading test file: $test_file" | tee -a $output_file
echo | tee -a $output_file
test_required_examples=""
. "$test_file" || $exit_cmd 1

## Build files required by the test
echo "Example(s) required by the test: $test_required_examples" | tee -a $output_file
build_examples $test_required_examples || $exit_cmd 1
echo | tee -a $output_file

## Loop over the number-of-processors list.
for (( num_proc_idx = 0; num_proc_idx < num_proc_list_size; num_proc_idx++ ))
do

num_proc_run="${num_proc_list[$num_proc_idx]}"
num_proc_node="${num_proc_node_list[$num_proc_idx]}"

set_num_nodes || $exit_cmd 1
compose_mpi_run_command

if [[ "$start_shell" = "yes" ]]; then
   if [[ ! -t 1 ]]; then
      echo "Standard output is not a terminal. Stop." | tee -a $output_file
      $exit_cmd 1
   fi
   echo "Reading shell commands, type 'c' to continue, 'exit' to stop ..." | tee -a $output_file
   echo | tee -a $output_file
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
   echo "Continuing ..." | tee -a $output_file
fi

# Call the function run_tests defined inside the $test_file
ceed=$backend
if [ -z "$dry_run" ]; then
   run_tests >> $output_file
else
   run_tests
fi
echo

done ## End of loop over processor numbers

trap - INT

} ## run is on

$exit_cmd 0

) || {
   echo "Sub-shell for backend '$backend' returned error code $?. Stop."
   $exit_cmd 1
}
done ## Loop over $backend_list

done ## Loop over $bp_list


$exit_cmd 0

