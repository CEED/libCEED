# Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
# All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
#
# SPDX-License-Identifier: BSD-2-Clause
#
# This file is part of CEED:  http://github.com/ceed
function run_tests()
{
   $dry_run cd "$test_exe_dir"

   # Some of the available options are:
   # -degree <1>: Polynomial degree of tensor product basis
   # -q_extra <1>: Number of extra quadrature points
   # -ceed </cpu/self>: CEED resource specifier
   # -local_nodes <1000>: Target number of locally (per rank) owned nodes

   # The variables 'max_dofs_node', and 'max_p' can be set on the command line
   # invoking the 'benchmark.sh' script.
   local ceed="${ceed:-/cpu/self}"
   local common_args=(-ceed $ceed -pc_type none -benchmark)
   local max_dofs_node_def=$((3*2**20))
   local max_dofs_node=${max_dofs_node:-$max_dofs_node_def}
   local max_loc_nodes=$((max_dofs_node/num_proc_node))
   local max_p=${max_p:-8}
   local sol_p=
   for ((sol_p = 1; sol_p <= max_p; sol_p++)); do
      local loc_el=
      for ((loc_el = 1; loc_el*sol_p**3 <= max_loc_nodes; loc_el = 2*loc_el)); do
         local loc_nodes=$((loc_el*sol_p**3))
         local all_args=("${common_args[@]}" -degree $sol_p -local_nodes $loc_nodes -problem $bp)
         if [ -z "$dry_run" ]; then
            echo
            echo "Running test:"
            quoted_echo $mpi_run ./petsc-bps "${all_args[@]}"
            $mpi_run ./petsc-bps "${all_args[@]}" || \
               printf "\nError in the test, error code: $?\n\n"
         else
            $dry_run $mpi_run ./petsc-bps "${all_args[@]}"
         fi
      done
   done
}

test_required_examples="petsc-bps"
