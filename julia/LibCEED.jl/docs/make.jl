using Documenter, LibCEED, LinearAlgebra

makedocs(sitename="LibCEED.jl Docs",
         format=Documenter.HTML(prettyurls=false),
         pages=[
             "Home" => "index.md",
             "Ceed Objects" => [
                "Ceed.md",
                "CeedVector.md",
                "ElemRestriction.md",
                "Basis.md",
                "QFunction.md",
                "Operator.md",
             ],
             "Utilities" => [
                "Misc.md",
                "Globals.md",
                "Quadrature.md",
             ],
             "LibCEED.md",
             "C.md",
             "UserQFunctions.md",
             "Examples.md",
         ])

deploydocs(
    repo="github.com/CEED/libCEED-julia-docs.git",
    devbranch="main",
    push_preview=true,
)
