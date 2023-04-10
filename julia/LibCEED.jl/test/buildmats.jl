function build_mats_hdiv(qref, qweight, ::Type{T}=Float64) where {T}
    P, Q, dim = 4, 4, 2
    interp = Array{T}(undef, dim, Q, P)
    div = Array{T}(undef, Q, P)

    qref[1, 1] = -1.0/sqrt(3.0)
    qref[1, 2] = qref[1, 1]
    qref[1, 3] = qref[1, 1]
    qref[1, 4] = -qref[1, 1]
    qref[2, 1] = -qref[1, 1]
    qref[2, 2] = -qref[1, 1]
    qref[2, 3] = qref[1, 1]
    qref[2, 4] = qref[1, 1]
    qweight[1] = 1.0
    qweight[2] = 1.0
    qweight[3] = 1.0
    qweight[4] = 1.0

    # Loop over quadrature points
    for i = 1:Q
        x1 = qref[1, i]
        x2 = qref[2, i]
        # Interp
        interp[1, i, 1] = 0.0
        interp[2, i, 1] = 1.0 - x2
        interp[1, i, 2] = x1 - 1.0
        interp[2, i, 2] = 0.0
        interp[1, i, 3] = -x1
        interp[2, i, 3] = 0.0
        interp[1, i, 4] = 0.0
        interp[2, i, 4] = x2
        # Div
        div[i, 1] = -1.0
        div[i, 2] = 1.0
        div[i, 3] = -1.0
        div[i, 4] = 1.0
    end

    return interp, div
end

function build_mats_hcurl(qref, qweight, ::Type{T}=Float64) where {T}
    P, Q, dim = 3, 4, 2
    interp = Array{T}(undef, dim, Q, P)
    curl = Array{T}(undef, 1, Q, P)

    qref[1, 1] = 0.2
    qref[1, 2] = 0.6
    qref[1, 3] = 1.0/3.0
    qref[1, 4] = 0.2
    qref[2, 1] = 0.2
    qref[2, 2] = 0.2
    qref[2, 3] = 1.0/3.0
    qref[2, 4] = 0.6
    qweight[1] = 25.0/96.0
    qweight[2] = 25.0/96.0
    qweight[3] = -27.0/96.0
    qweight[4] = 25.0/96.0

    # Loop over quadrature points
    for i = 1:Q
        x1 = qref[1, i]
        x2 = qref[2, i]
        # Interp
        interp[1, i, 1] = -x2
        interp[2, i, 1] = x1
        interp[1, i, 2] = x2
        interp[2, i, 2] = 1.0 - x1
        interp[1, i, 3] = 1.0 - x2
        interp[2, i, 3] = x1
        # Curl
        curl[1, i, 1] = 2.0
        curl[1, i, 2] = -2.0
        curl[1, i, 3] = -2.0
    end

    return interp, curl
end
