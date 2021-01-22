import fileinput
import numpy as np
import pandas as pd
from pylab import *
from matplotlib import use


def read_logs(files=None):
    """Read all input files and return pandas DataFrame"""
    data_default = dict(
        problem='unknown',
        num_procs=0,
        mesh_res=0,
        degree=0,
        quadrature_pts=0,
        num_unknowns=0,
        max_error=0
    )
    data = data_default.copy()
    runs = []
    for line in fileinput.input(files):
        # Number of MPI tasks
        if 'rank(s)' in line:
            data = data_default.copy()
            data['num_procs'] = int(line.split(': ', 1)[1])
        # New Problem
        elif "Problem Name" in line:
            # Starting a new block
            data = data.copy()
            runs.append(data)
            data['problem'] = line.split(': ')[1].strip()
        # Mesh resolution
        elif "Box Faces" in line:
            res = line.split(': ')[1]
            data['mesh_res'] = int(line.split(',')[1])
        # P
        elif 'Basis Nodes' in line:
            data['degree'] = int(line.split(': ')[1]) - 1
        # Q
        elif 'Quadrature Points' in line:
            data['quadrature_pts'] = int(line.split(': ')[1])
        # Total DOFs
        elif 'Global DoFs' in line:
            data['num_unknowns'] = int(line.split(': ')[1])
        # Max Error
        elif 'Max Error' in line:
            data['max_error'] = float(line.split(': ')[1])
        # End of output
    return pd.DataFrame(runs)


### Plotting
def plot():
    # Load the data
    runs = read_logs()
    colors = ['orange', 'red', 'navy', 'green', 'magenta',
              'teal', 'blue', 'purple', 'pink', 'cyan']
    res = 'mesh_res'
    fig, ax = plt.subplots()
    i = 0
    HH = [0.4*1e1, 2*1e2, 1e4, 5e5]
    for group in runs.groupby('degree'):
        data = group[1]
        data = data.sort_values('max_error')
        p = data['degree'].values[0]
        h = 1/data[res]
        H = HH[i] * h**p # H = h^p
        E = data['max_error']
        log_h = np.log10(h)
        log_H = np.log10(H)
        log_E = np.log10(E)
        ax.loglog(h, E, 'o', color=colors[i])
        m, b = np.polyfit(log_h, log_H, 1)
        n, c = np.polyfit(log_h, log_E, 1)
        ax.loglog(h, 10**b * h**m, '--', color=colors[i], label='O(h^' + str(p) + ')')
        ax.loglog(h, 10**c * h**n, '-', color=colors[i], label='p=' + str(p))
        i = i + 1

    ax.legend(loc='best')
    ax.set_xlabel('h')
    ax.set_ylabel('Max Error')
    ax.set_title('Convergence by h Refinement')
    xlim(.005, .05)
    fig.tight_layout()
    plt.savefig('h_ref_conv_test.png', bbox_inches='tight')


if __name__ == "__main__":
    runs = read_logs()
    plot()
    print(runs)
