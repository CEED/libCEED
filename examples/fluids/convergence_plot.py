#!/usr/bin/env python3
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

import numpy as np
import pandas as pd
from pylab import *
from matplotlib import use


def plot():
    # Load the data
    runs = pd.read_csv("conv_test_result.csv")
    colors = ['orange', 'red', 'navy', 'green', 'magenta',
              'gray', 'blue', 'purple', 'pink', 'black']
    res = 'mesh_res'
    fig, ax = plt.subplots()
    # Arbitrary coefficients
    C = [2.2e-2, .24e0, .22e0, .7e0, 2.5e0,
        3e0, 3.5e0, 4e0, 4.5e0, 5e0]
    i = 0
    for group in runs.groupby('degree'):
        data = group[1]
        data = data.sort_values('rel_error')
        p = data['degree'].values[0]
        h = 1/data[res]
        H = C[i] * h**p # H = C h^p
        E = data['rel_error']
        log_h = np.log10(h)
        log_H = np.log10(H)
        ax.loglog(h, E, 'o', color=colors[i])
        m, b = np.polyfit(log_h, log_H, 1)
        ax.loglog(h, 10**b * h**m, '--', color=colors[i], label='O(h^' + str(p) + ')')
        i = i + 1

    ax.legend(loc='best')
    ax.set_xlabel('h')
    ax.set_ylabel('Relative Error')
    ax.set_title('Convergence by h Refinement')
    xlim(.03, .3)
    fig.tight_layout()
    plt.savefig('h_conv_plt.png', bbox_inches='tight')


if __name__ == "__main__":
    plot()
