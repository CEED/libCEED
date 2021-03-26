using Clang
using Clang.LibClang.Clang_jll

"""
    generate_ceed_wrapper(ceed_path)

Generate Julia bindings for the libCEED library, where `ceed_path` is the path
of the libCEED directory. The generated bindings are used (with some manual
modifications) to create the low-level C interface LibCEED.C for the LibCEED.jl
package.
"""
function generate_ceed_wrapper(ceed_path)
    headers = ["ceed/ceed.h", "ceed/cuda.h", "ceed/backend.h"]
    ceed_include = joinpath(ceed_path, "include")
    ceed_headers = [joinpath(ceed_include, header) for header in headers]

    wc = init(;
        headers=ceed_headers,
        output_file=joinpath(@__DIR__, "libceed_api_gen.jl"),
        common_file=joinpath(@__DIR__, "libceed_common_gen.jl"),
        clang_includes=[ceed_include, CLANG_INCLUDE],
        clang_args=map(x -> "-I"*x, find_std_headers()),
        header_wrapped=(root, current) -> root == current,
        header_library=x -> "libceed",
        clang_diagnostics=true,
    )
    run(wc, false)
end
