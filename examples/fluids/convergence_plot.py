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
        # Problem name
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
        # Relative Error
        elif 'Relative Error' in line:
            data['max_error'] = float(line.split(': ')[1])
        # End of output
    return pd.DataFrame(runs)


### Plotting
def plot():
    # Load the data
    runs = read_logs()
    colors = ['orange', 'red', 'navy', 'green', 'magenta',
              'gray', 'blue', 'purple', 'pink', 'black']
    res = 'mesh_res'
    fig, ax = plt.subplots()
    i = 0
    HH = [2.2e-2, .24e0, .22e0, .7e0, 2.5e0,
          3e0, 3.5e0, 4e0, 4.5e0, 5e0]
    for group in runs.groupby('degree'):
        data = group[1]
        data = data.sort_values('max_error')
        p = data['degree'].values[0]
        h = 1/data[res]
        H = HH[i] * h**p # H = C h^p
        E = data['max_error']
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
    plt.savefig('h_convergence_plot.png', bbox_inches='tight')


if __name__ == "__main__":
    runs = read_logs()
    plot()
    print(runs)
