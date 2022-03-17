#!/usr/bin/env python3
# Copyright (c) 2017-2018, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory. LLNL-CODE-734707.
# All Rights reserved. See files LICENSE and NOTICE for details.
#
# This file is part of CEED, a collection of benchmarks, miniapps, software
# libraries and APIs for efficient high-order finite element and spectral
# element discretizations for exascale applications. For more information and
# source code availability see http://github.com/ceed
#
# The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
# a collaborative effort of two U.S. Department of Energy organizations (Office
# of Science and the National Nuclear Security Administration) responsible for
# the planning and preparation of a capable exascale ecosystem, including
# software, applications, hardware, advanced system engineering and early
# testbed platforms, in support of the nation's exascale computing imperative.


# Adjustable plot parameters:
from pylab import *
from matplotlib import use
from postprocess_base import read_logs
import pandas as pd
log_y = 0               # use log scale on the y-axis?
x_range = (1e1, 4e6)     # plot range for the x-axis; comment out for auto
y_range = (0, 2e9)       # plot range for the y-axis; comment out for auto
draw_iter_lines = 0     # draw the "iter/s" lines?
ymin_iter_lines = 3e5   # minimal y value for the "iter/s" lines
ymax_iter_lines = 8e8   # maximal y value for the "iter/s" lines
legend_ncol = (2 if log_y else 1)   # number of columns in the legend
write_figures = 1       # save the figures to files?
show_figures = 1        # display the figures on the screen?


# Load the data

runs = read_logs()

# Sample plot output
if not show_figures:
    use('pdf')

rcParams['font.sans-serif'].insert(0, 'Noto Sans')
rcParams['font.sans-serif'].insert(1, 'Open Sans')
rcParams['figure.figsize'] = [10, 8]  # default: 8 x 6

cm_size = 16
colors = ['dimgrey', 'black', 'saddlebrown', 'firebrick', 'red', 'orange',
          'gold', 'lightgreen', 'green', 'cyan', 'teal', 'blue', 'navy',
          'purple', 'magenta', 'pink']

# Get test names
sel_runs = runs
tests = list(sel_runs.test.unique())
test = tests[0]

# Run information
print('Using test:', test)

if 'CEED Benchmark Problem' in test:
    test_short = test.strip().split()[0] + ' BP' + test.strip().split()[-1]

# Plot same BP
sel_runs = sel_runs.loc[sel_runs['test'] == test]

# Plot same case (scalar vs vector)
cases = list(sel_runs.case.unique())
case = cases[0]
vdim = 1 if case == 'scalar' else 3
print('Using case:', case)
sel_runs = sel_runs.loc[sel_runs['case'] == case]

# Plot same 'code'
codes = list(sel_runs.code.unique())
code = codes[0]
sel_runs = sel_runs.loc[sel_runs['code'] == code]

# Group plots by backend and number of processes
pl_set = sel_runs[['backend',
                   'backend_memtype',
                   'num_procs',
                   'num_procs_node']]
pl_set = pl_set.drop_duplicates()

# Plotting
for index, row in pl_set.iterrows():
    backend = row['backend']
    backend_memtype = row['backend_memtype']
    num_procs = float(row['num_procs'])
    num_procs_node = float(row['num_procs_node'])
    num_nodes = num_procs / num_procs_node
    pl_runs = sel_runs[(sel_runs.backend == backend) |
                       (sel_runs.num_procs == num_procs) |
                       (sel_runs.num_procs_node == num_procs_node)]
    if len(pl_runs.index) == 0:
        continue

    print('backend: %s, compute nodes: %i, number of MPI tasks = %i' % (
        backend, num_nodes, num_procs))

    figure()
    i = 0
    sol_p_set = sel_runs['degree'].drop_duplicates()
    sol_p_set = sol_p_set.sort_values()
    # Iterate over P
    for sol_p in sol_p_set:
        qpts = sel_runs['quadrature_pts'].loc[pl_runs['degree'] == sol_p]
        qpts = qpts.drop_duplicates().sort_values(ascending=False)
        qpts = qpts.reset_index(drop=True)
        print('Degree: %i, quadrature points:' % sol_p, qpts[0])
        # Generate plot data
        d = [[run['degree'], run['num_elem'], 1. * run['num_unknowns'] / num_nodes / vdim,
              run['cg_iteration_dps'] / num_nodes]
             for index, run in
             pl_runs.loc[(pl_runs['degree'] == sol_p) |
                         (pl_runs['quadrature_pts'] == qpts[0])].iterrows()]
        d = [[e[2], e[3]] for e in d if e[0] == sol_p]
        # (DOFs/[sec/iter]/node)/(DOFs/node) = iter/sec
        d = [[nun,
              min([e[1] for e in d if e[0] == nun]),
              max([e[1] for e in d if e[0] == nun])]
             for nun in set([e[0] for e in d])]
        d = asarray(sorted(d))
        # Plot
        plot(d[:, 0], d[:, 2], 'o-', color=colors[i % cm_size],
             label='p=%i' % sol_p)
        if list(d[:, 1]) != list(d[:, 2]):
            plot(d[:, 0], d[:, 1], 'o-', color=colors[i])
            fill_between(d[:, 0], d[:, 1], d[:, 2],
                         facecolor=colors[i], alpha=0.2)
        # Continue if only 1 set of qpts
        if len(qpts) == 1:
            i = i + 1
            continue
        # Second set of qpts
        d = [[run['degree'], run['num_elem'], 1. * run['num_unknowns'] / num_nodes / vdim,
              run['cg_iteration_dps'] / num_nodes]
             for index, run in
             pl_runs.loc[(pl_runs['degree'] == sol_p) |
                         (pl_runs['quadrature_pts'] == qpts[1])].iterrows()]
        d = [[e[2], e[3]] for e in d if e[0] == sol_p]
        if len(d) == 0:
            i = i + 1
            continue
        d = [[nun,
              min([e[1] for e in d if e[0] == nun]),
              max([e[1] for e in d if e[0] == nun])]
             for nun in set([e[0] for e in d])]
        d = asarray(sorted(d))
        plot(d[:, 0], d[:, 2], 's--', color=colors[i],
             label='p=%i' % sol_p)
        if list(d[:, 1]) != list(d[:, 2]):
            plot(d[:, 0], d[:, 1], 's--', color=colors[i])
        ##
        i = i + 1
    ##
    if draw_iter_lines:
        y0, y1 = ymin_iter_lines, ymax_iter_lines
        y = asarray([y0, y1]) if log_y else exp(linspace(log(y0), log(y1)))
        slope1 = 600.
        slope2 = 6000.
        plot(y / slope1, y, 'k--', label='%g iter/s' % (slope1 / vdim))
        plot(y / slope2, y, 'k-', label='%g iter/s' % (slope2 / vdim))

    # Plot information
    title(r'%i node%s $\times$ %i ranks, %s, %s, %s' % (
          num_nodes, '' if num_nodes == 1 else 's',
          num_procs_node, backend, backend_memtype, test_short), fontsize=16)
    xscale('log')  # subsx=[2,4,6,8]
    if log_y:
        yscale('log')
    if 'x_range' in vars() and len(x_range) == 2:
        xlim(x_range)
    if 'y_range' in vars() and len(y_range) == 2:
        ylim(y_range)
    grid('on', color='gray', ls='dotted')
    grid('on', axis='both', which='minor', color='gray', ls='dotted')
    plt.tick_params(labelsize=14)
    exptext = gca().yaxis.get_offset_text()
    exptext.set_size(14)
    gca().set_axisbelow(True)
    xlabel('Points per compute node', fontsize=14)
    ylabel('[DOFs x CG iterations] / [compute nodes x seconds]', fontsize=14)
    legend(ncol=legend_ncol, loc='best', fontsize=13)

    # Write
    if write_figures:  # write .pdf file?
        short_backend = backend.replace('/', '')
        test_short_save = test_short.replace(' ', '')
        pdf_file = 'plot_%s_%s_%s_%s_N%03i_pn%i.pdf' % (
            code, test_short_save, short_backend, backend_memtype, num_nodes, num_procs_node)
        print('\nsaving figure --> %s' % pdf_file)
        savefig(pdf_file, format='pdf', bbox_inches='tight')

if show_figures:  # show the figures?
    print('\nShowing figures ...')
    show()
