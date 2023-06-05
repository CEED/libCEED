using Clang.Generators

function generate_ceed_bindings(ceed_path)
    # libCEED include dir and header files
    include_dir = joinpath(ceed_path, "include") |> normpath
    header_files = ["ceed.h", "ceed/cuda.h", "ceed/backend.h"]
    headers = [joinpath(include_dir, header) for header in header_files]

    # load options from generator TOML file
    options = load_options(joinpath(@__DIR__, "generator.toml"))

    # add compiler flags
    args = get_default_args()
    push!(args, "-I$include_dir")

    # create context
    ctx = create_context(headers, args, options)

    # run generator
    build!(ctx)

    # remove trailing newline from output file
    outfile = options["general"]["output_file_path"]
    write(outfile, readchomp(outfile))
    return
end
