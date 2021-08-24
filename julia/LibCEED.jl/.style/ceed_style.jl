using JuliaFormatter, CSTParser, Tokenize

for name in names(JuliaFormatter, all=true)
    if name != :include && name != :eval && name != Base.Docs.META
        @eval import JuliaFormatter: $name
    end
end

# Same as DefaultStyle, but no space in between operators with precedence CSTParser.TimesOp
struct CeedStyle <: AbstractStyle end

getstyle(s::CeedStyle) = s

function p_binaryopcall(
    style::CeedStyle,
    cst::CSTParser.EXPR,
    s::State;
    nonest=false,
    nospace=false,
)
    t = FST(cst, nspaces(s))
    op = cst[2]
    nonest = nonest || op.kind === Tokens.COLON
    if cst.parent.typ === CSTParser.Curly &&
       op.kind in (Tokens.ISSUBTYPE, Tokens.ISSUPERTYPE) &&
       !s.opts.whitespace_typedefs
        nospace = true
    elseif op.kind === Tokens.COLON
        nospace = true
    end
    nospace_args = s.opts.whitespace_ops_in_indices ? false : nospace

    if is_opcall(cst[1])
        n = pretty(style, cst[1], s, nonest=nonest, nospace=nospace_args)
    else
        n = pretty(style, cst[1], s)
    end

    if op.kind === Tokens.COLON &&
       s.opts.whitespace_ops_in_indices &&
       !is_leaf(cst[1]) &&
       !is_iterable(cst[1])
        paren = FST(CSTParser.PUNCTUATION, -1, n.startline, n.startline, "(")
        add_node!(t, paren, s)
        add_node!(t, n, s, join_lines=true)
        paren = FST(CSTParser.PUNCTUATION, -1, n.startline, n.startline, ")")
        add_node!(t, paren, s, join_lines=true)
    else
        add_node!(t, n, s)
    end

    nrhs = nest_rhs(cst)
    nrhs && (t.nest_behavior = AlwaysNest)
    nest = (nestable(style, cst) && !nonest) || nrhs

    if op.fullspan == 0
        # Do nothing - represents a binary op with no textual representation.
        # For example: `2a`, which is equivalent to `2 * a`.
    elseif op.kind === Tokens.EX_OR
        add_node!(t, Whitespace(1), s)
        add_node!(t, pretty(style, op, s), s, join_lines=true)
    elseif (is_number(cst[1]) || op.kind === Tokens.CIRCUMFLEX_ACCENT) && op.dot
        add_node!(t, Whitespace(1), s)
        add_node!(t, pretty(style, op, s), s, join_lines=true)
        nest ? add_node!(t, Placeholder(1), s) : add_node!(t, Whitespace(1), s)
    elseif op.kind !== Tokens.IN && (
        nospace || (
            op.kind !== Tokens.ANON_FUNC && CSTParser.precedence(op) in (
                CSTParser.ColonOp,
                CSTParser.PowerOp,
                CSTParser.DeclarationOp,
                CSTParser.DotOp,
                CSTParser.TimesOp,
            )
        )
    )
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

    if op.kind === Tokens.COLON &&
       s.opts.whitespace_ops_in_indices &&
       !is_leaf(cst[3]) &&
       !is_iterable(cst[3])
        paren = FST(CSTParser.PUNCTUATION, -1, n.startline, n.startline, "(")
        add_node!(t, paren, s, join_lines=true)
        add_node!(t, n, s, join_lines=true)
        paren = FST(CSTParser.PUNCTUATION, -1, n.startline, n.startline, ")")
        add_node!(t, paren, s, join_lines=true)
    else
        add_node!(t, n, s, join_lines=true)
    end

    if nest
        # for indent, will be converted to `indent` if needed
        insert!(t.nodes, length(t.nodes), Placeholder(0))
    end

    t
end

function p_chainopcall(
    style::CeedStyle,
    cst::CSTParser.EXPR,
    s::State;
    nonest=false,
    nospace=false,
)
    t = FST(cst, nspaces(s))

    # Check if there's a number literal on the LHS of a dot operator.
    # In this case we need to surround the dot operator with whitespace
    # in order to avoid ambiguity.
    for (i, a) in enumerate(cst)
        if a.typ === CSTParser.OPERATOR && a.dot && is_number(cst[i-1])
            nospace = false
            break
        end
    end

    nws = nospace ? 0 : 1
    for (i, a) in enumerate(cst)
        if a.typ === CSTParser.OPERATOR
            nws_op = (CSTParser.precedence(a) == CSTParser.TimesOp) ? 0 : nws
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
