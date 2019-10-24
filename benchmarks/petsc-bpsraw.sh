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


function run_tests()
{
   $dry_run cd "$test_exe_dir"

   # Some of the available options are:
   # -degree <1>: Polynomial degree of tensor product basis
   # -qextra <2>: Number of extra quadrature points
   # -ceed </cpu/self>: CEED resource specifier
   # -local <1000>: Target number of locally (per rank) owned degrees of freedom

   # The variables 'max_dofs_node', and 'max_p' can be set on the command line
   # invoking the 'benchmark.sh' script.
   local ceed="${ceed:-/cpu/self}"
   local common_args=(-ceed $ceed -pc_type none -benchmark)
   local max_dofs_node_def=$((3*2**20))
   local max_dofs_node=${max_dofs_node:-$max_dofs_node_def}
   local max_loc_dofs=$((max_dofs_node/num_proc_node))
   local max_p=${max_p:-8}
   local sol_p=
   for ((sol_p = 1; sol_p <= max_p; sol_p++)); do
      local loc_el=
      for ((loc_el = 1; loc_el*sol_p**3 <= max_loc_dofs; loc_el = 2*loc_el)); do
         local loc_dofs=$((loc_el*sol_p**3))
         local all_args=("${common_args[@]}" -degree $sol_p -local $loc_dofs -problem $bp)
         if [ -z "$dry_run" ]; then
            echo
            echo "Running test:"
            quoted_echo $mpi_run ./petsc-bpsraw "${all_args[@]}"
            $mpi_run ./petsc-bpsraw "${all_args[@]}" || \
               printf "\nError in the test, error code: $?\n\n"
         else
            $dry_run $mpi_run ./petsc-bpsraw "${all_args[@]}"
         fi
      done
   done
}

test_required_examples="petsc-bpsraw"
