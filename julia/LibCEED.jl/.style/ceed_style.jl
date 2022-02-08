using JuliaFormatter, CSTParser, Tokenize

for name in names(JuliaFormatter, all=true)
    if name != :include && name != :eval && name != Base.Docs.META
        @eval using JuliaFormatter: $name
    end
end

# Same as DefaultStyle, but no space in between operators with precedence CSTParser.TimesOp
struct CeedStyle <: AbstractStyle end
@inline JuliaFormatter.getstyle(s::CeedStyle) = s

function JuliaFormatter.p_binaryopcall(
    ds::CeedStyle,
    cst::CSTParser.EXPR,
    s::State;
    nonest=false,
    nospace=false,
)
    style = getstyle(ds)
    t = FST(Binary, cst, nspaces(s))
    op = cst[2]

    nonest = nonest || CSTParser.is_colon(op)

    if CSTParser.iscurly(cst.parent) &&
       (op.val == "<:" || op.val == ">:") &&
       !s.opts.whitespace_typedefs
        nospace = true
    elseif CSTParser.is_colon(op)
        nospace = true
    end
    nospace_args = s.opts.whitespace_ops_in_indices ? false : nospace

    if is_opcall(cst[1])
        n = pretty(style, cst[1], s, nonest=nonest, nospace=nospace_args)
    else
        n = pretty(style, cst[1], s)
    end

    if CSTParser.is_colon(op) &&
       s.opts.whitespace_ops_in_indices &&
       !is_leaf(cst[1]) &&
       !is_iterable(cst[1])
        paren = FST(PUNCTUATION, -1, n.startline, n.startline, "(")
        add_node!(t, paren, s)
        add_node!(t, n, s, join_lines=true)
        paren = FST(PUNCTUATION, -1, n.startline, n.startline, ")")
        add_node!(t, paren, s, join_lines=true)
    else
        add_node!(t, n, s)
    end

    nrhs = nest_rhs(cst)
    nrhs && (t.nest_behavior = AlwaysNest)
    nest = (is_binaryop_nestable(style, cst) && !nonest) || nrhs

    if op.fullspan == 0
        # Do nothing - represents a binary op with no textual representation.
        # For example: `2a`, which is equivalent to `2 * a`.
    elseif CSTParser.is_exor(op)
        add_node!(t, pretty(style, op, s), s, join_lines=true)
    elseif (CSTParser.isnumber(cst[1]) || is_circumflex_accent(op)) &&
           CSTParser.isdotted(op)
        add_node!(t, Whitespace(1), s)
        add_node!(t, pretty(style, op, s), s, join_lines=true)
        nest ? add_node!(t, Placeholder(1), s) : add_node!(t, Whitespace(1), s)
    elseif !(CSTParser.is_in(op) || CSTParser.is_elof(op)) && (
        nospace || (
            !CSTParser.is_anon_func(op) && precedence(op) in (
                CSTParser.PowerOp,
                CSTParser.DeclarationOp,
                CSTParser.DotOp,
                CSTParser.TimesOp,
            )
        )
    )
        add_node!(t, pretty(style, op, s), s, join_lines=true)
    elseif op.val in RADICAL_OPS
        add_node!(t, pretty(style, op, s), s, join_lines=true)
    else
        add_node!(t, Whitespace(1), s)
        add_node!(t, pretty(style, op, s), s, join_lines=true)
        nest ? add_node!(t, Placeholder(1), s) : add_node!(t, Whitespace(1), s)
    end

    if is_opcall(cst[3])
        n = pretty(style, cst[3], s, nonest=nonest, nospace=nospace_args)
    else
        n = pretty(style, cst[3], s)
    end

    if CSTParser.is_colon(op) &&
       s.opts.whitespace_ops_in_indices &&
       !is_leaf(cst[3]) &&
       !is_iterable(cst[3])
        paren = FST(PUNCTUATION, -1, n.startline, n.startline, "(")
        add_node!(t, paren, s, join_lines=true)
        add_node!(t, n, s, join_lines=true, override_join_lines_based_on_source=!nest)
        paren = FST(PUNCTUATION, -1, n.startline, n.startline, ")")
        add_node!(t, paren, s, join_lines=true)
    else
        add_node!(t, n, s, join_lines=true, override_join_lines_based_on_source=!nest)
    end

    if nest
        # for indent, will be converted to `indent` if needed
        insert!(t.nodes, length(t.nodes), Placeholder(0))
    end

    t
end

function JuliaFormatter.p_chainopcall(
    ds::CeedStyle,
    cst::CSTParser.EXPR,
    s::State;
    nonest=false,
    nospace=false,
)
    style = getstyle(ds)
    t = FST(Chain, cst, nspaces(s))

    # Check if there's a number literal on the LHS of a dot operator.
    # In this case we need to surround the dot operator with whitespace
    # in order to avoid ambiguity.
    for (i, a) in enumerate(cst)
        if CSTParser.isoperator(a) && CSTParser.isdotted(a) && CSTParser.isnumber(cst[i-1])
            nospace = false
            break
        end
    end

    nws = nospace ? 0 : 1
    for (i, a) in enumerate(cst)
        nws_op = precedence(a) == CSTParser.TimesOp ? 0 : nws
        if CSTParser.isoperator(a)
            add_node!(t, Whitespace(nws_op), s)
            add_node!(t, pretty(style, a, s), s, join_lines=true)
            if nonest
                add_node!(t, Whitespace(nws_op), s)
            else
                add_node!(t, Placeholder(nws_op), s)
            end
        elseif is_opcall(a)
            add_node!(
                t,
                pretty(style, a, s, nospace=nospace, nonest=nonest),
                s,
                join_lines=true,
            )
        elseif i == length(cst) - 1 && is_punc(a) && is_punc(cst[i+1])
            add_node!(t, pretty(style, a, s), s, join_lines=true)
        else
            add_node!(t, pretty(style, a, s), s, join_lines=true)
        end
    end
    t
end

prefix_path(fname) = joinpath(@__DIR__, "..", fname)
format(
    prefix_path.(["src", "test", "examples", ".style"]),
    style=CeedStyle(),
    indent=4,
    margin=92,
    remove_extra_newlines=true,
    whitespace_in_kwargs=false,
)
