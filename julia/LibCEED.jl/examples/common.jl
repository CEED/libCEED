function get_cartesian_mesh_size(dim, order, prob_size)
    dims = zeros(CeedInt, dim)
    # Use the approximate formula:
    #    prob_size ~ num_elem * order^dim
    num_elem = div(prob_size, order^dim)
    s = 0 # find s: num_elem/2 < 2^s <= num_elem
    while num_elem > 1
        num_elem = div(num_elem, 2)
        s += 1
    end
    r = s%dim
    for d = 1:dim
        sd = div(s, dim)
        if r > 0
            sd += 1
            r -= 1
        end
        dims[d] = 2^sd
    end
    dims
end

struct FormRestrictionMode{T} end
const RestrictionOnly = FormRestrictionMode{:restr}()
const StridedOnly = FormRestrictionMode{:restr_i}()
const RestrictionAndStrided = FormRestrictionMode{:both}()

function build_cartesian_restriction(
    c::Ceed,
    dim,
    nxyz,
    order,
    ncomp,
    num_qpts;
    mode::Mode=RestrictionOnly,
) where {Mode}
    p::CeedInt = order
    pp1::CeedInt = p + 1
    nnodes::CeedInt = pp1^dim # number of scal. nodes per element
    elem_qpts::CeedInt = num_qpts^dim # number of qpts per element

    nd = CeedInt.(p*nxyz .+ 1)
    num_elem::CeedInt = prod(nxyz)
    scalar_size::CeedInt = prod(nd)
    size::CeedInt = scalar_size*ncomp

    form_restr = (Mode() != StridedOnly)
    form_strided = (Mode() != RestrictionOnly)

    # elem:         0             1                 n-1
    #        |---*-...-*---|---*-...-*---|- ... -|--...--|
    # nnodes:   0   1    p-1  p  p+1       2*p             n*p

    if form_restr
        el_nodes = zeros(CeedInt, num_elem*nnodes)
        exyz = zeros(CeedInt, dim)
        @inbounds @simd for e = 0:(num_elem-1)
            re::CeedInt = e
            for d::CeedInt = 1:dim
                exyz[d] = re%nxyz[d]
                re ÷= nxyz[d]
            end
            for lnodes::CeedInt = 0:(nnodes-1)
                gnodes::CeedInt = 0
                gnodes_stride::CeedInt = 1
                rnodes::CeedInt = lnodes
                for d::CeedInt = 1:dim
                    q = rnodes÷pp1
                    r = rnodes - pp1*q
                    gnodes::CeedInt += (exyz[d]*p + r)*gnodes_stride
                    gnodes_stride::CeedInt *= nd[d]
                    rnodes = q
                end
                el_nodes[e*nnodes+lnodes+1] = gnodes
            end
        end
    end

    restr =
        form_restr ?
        create_elem_restriction(
            c,
            num_elem,
            nnodes,
            ncomp,
            scalar_size,
            ncomp*scalar_size,
            el_nodes,
        ) : nothing
    restr_i =
        form_strided ?
        create_elem_restriction_strided(
            c,
            num_elem,
            elem_qpts,
            ncomp,
            ncomp*elem_qpts*num_elem,
            STRIDES_BACKEND,
        ) : nothing

    return size, restr, restr_i
end

function set_cartesian_mesh_coords!(dim, nxyz, mesh_order, mesh_coords)
    p = mesh_order
    nd = p*nxyz .+ 1
    num_elem::CeedInt = prod(nxyz)
    scalar_size::CeedInt = prod(nd)

    # The H1 basis uses Lobatto quadrature points as nodes.
    nodes::Vector{CeedScalar} = lobatto_quadrature(p + 1) # nodes are in [-1,1]
    nodes = 0.5 .+ 0.5*nodes

    @witharray coords = mesh_coords begin
        @inbounds @simd for gsnodes = 0:scalar_size-1
            rnodes = gsnodes
            for d = 1:dim
                ndd = nd[d]
                q = rnodes÷ndd
                r = rnodes - ndd*q

                d1d = r
                coords[gsnodes+scalar_size*(d-1)+1] = ((d1d÷p) + nodes[d1d%p+1])/nxyz[d]
                rnodes = q
            end
        end
    end
end
