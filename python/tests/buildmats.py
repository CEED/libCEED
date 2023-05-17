import numpy as np


def buildmats(qref, qweight, mat_dtype="float64"):
    P, Q, dim = 6, 4, 2
    interp = np.empty(P * Q, dtype=mat_dtype)
    grad = np.empty(dim * P * Q, dtype=mat_dtype)

    qref[0] = 0.2
    qref[1] = 0.6
    qref[2] = 1. / 3.
    qref[3] = 0.2
    qref[4] = 0.2
    qref[5] = 0.2
    qref[6] = 1. / 3.
    qref[7] = 0.6
    qweight[0] = 25. / 96.
    qweight[1] = 25. / 96.
    qweight[2] = -27. / 96.
    qweight[3] = 25. / 96.

    # Loop over quadrature points
    for i in range(Q):
        x1 = qref[0 * Q + i]
        x2 = qref[1 * Q + i]
        # Interp
        interp[i * P + 0] = 2. * (x1 + x2 - 1.) * (x1 + x2 - 1. / 2.)
        interp[i * P + 1] = -4. * x1 * (x1 + x2 - 1.)
        interp[i * P + 2] = 2. * x1 * (x1 - 1. / 2.)
        interp[i * P + 3] = -4. * x2 * (x1 + x2 - 1.)
        interp[i * P + 4] = 4. * x1 * x2
        interp[i * P + 5] = 2. * x2 * (x2 - 1. / 2.)
        # Grad
        grad[(i + 0) * P + 0] = 2. * \
            (1. * (x1 + x2 - 1. / 2.) + (x1 + x2 - 1.) * 1.)
        grad[(i + Q) * P + 0] = 2. * \
            (1. * (x1 + x2 - 1. / 2.) + (x1 + x2 - 1.) * 1.)
        grad[(i + 0) * P + 1] = -4. * (1. * (x1 + x2 - 1.) + x1 * 1.)
        grad[(i + Q) * P + 1] = -4. * (x1 * 1.)
        grad[(i + 0) * P + 2] = 2. * (1. * (x1 - 1. / 2.) + x1 * 1.)
        grad[(i + Q) * P + 2] = 2. * 0.
        grad[(i + 0) * P + 3] = -4. * (x2 * 1.)
        grad[(i + Q) * P + 3] = -4. * (1. * (x1 + x2 - 1.) + x2 * 1.)
        grad[(i + 0) * P + 4] = 4. * (1. * x2)
        grad[(i + Q) * P + 4] = 4. * (x1 * 1.)
        grad[(i + 0) * P + 5] = 2. * 0.
        grad[(i + Q) * P + 5] = 2. * (1. * (x2 - 1. / 2.) + x2 * 1.)

    return interp, grad


def buildmatshdiv(qref, qweight, mat_dtype="float64"):
    P, Q, dim = 4, 4, 2
    interp = np.empty(dim * P * Q, dtype=mat_dtype)
    div = np.empty(P * Q, dtype=mat_dtype)

    qref[0] = -1. / np.sqrt(3.)
    qref[1] = qref[0]
    qref[2] = qref[0]
    qref[3] = -qref[0]
    qref[4] = -qref[0]
    qref[5] = -qref[0]
    qref[6] = qref[0]
    qref[7] = qref[0]
    qweight[0] = 1.
    qweight[1] = 1.
    qweight[2] = 1.
    qweight[3] = 1.

    # Loop over quadrature points
    for i in range(Q):
        x1 = qref[0 * Q + i]
        x2 = qref[1 * Q + i]
        # Interp
        interp[(i + 0) * P + 0] = 0.
        interp[(i + Q) * P + 0] = 1. - x2
        interp[(i + 0) * P + 1] = x1 - 1.
        interp[(i + Q) * P + 1] = 0.
        interp[(i + 0) * P + 2] = -x1
        interp[(i + Q) * P + 2] = 0.
        interp[(i + 0) * P + 3] = 0.
        interp[(i + Q) * P + 3] = x2
        # Div
        div[i * P + 0] = -1.
        div[i * P + 1] = 1.
        div[i * P + 2] = -1.
        div[i * P + 3] = 1.

    return interp, div


def buildmatshcurl(qref, qweight, mat_dtype="float64"):
    P, Q, dim = 3, 4, 2
    interp = np.empty(dim * P * Q, dtype=mat_dtype)
    curl = np.empty(P * Q, dtype=mat_dtype)

    qref[0] = 0.2
    qref[1] = 0.6
    qref[2] = 1. / 3.
    qref[3] = 0.2
    qref[4] = 0.2
    qref[5] = 0.2
    qref[6] = 1. / 3.
    qref[7] = 0.6
    qweight[0] = 25. / 96.
    qweight[1] = 25. / 96.
    qweight[2] = -27. / 96.
    qweight[3] = 25. / 96.

    # Loop over quadrature points
    for i in range(Q):
        x1 = qref[0 * Q + i]
        x2 = qref[1 * Q + i]
        # Interp
        interp[(i + 0) * P + 0] = -x2
        interp[(i + Q) * P + 0] = x1
        interp[(i + 0) * P + 1] = x2
        interp[(i + Q) * P + 1] = 1. - x1
        interp[(i + 0) * P + 2] = 1. - x2
        interp[(i + Q) * P + 2] = x1
        # Curl
        curl[i * P + 0] = 2.
        curl[i * P + 1] = -2.
        curl[i * P + 2] = -2.

    return interp, curl
