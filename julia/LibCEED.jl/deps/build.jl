const libceed_envvar = "JULIA_LIBCEED_LIB"

if isfile("deps.jl")
   rm("deps.jl")
end

if haskey(ENV, libceed_envvar)
   ceedpath = ENV[libceed_envvar]
   if !isfile(ceedpath)
      error("No library file found at $libceed_envvar, given by $ceedpath")
   end
   @info "Using libCEED library specified by $libceed_envvar at $ceedpath"
   open("deps.jl", write=true) do f
      println(f, "const libceed = \"$(escape_string(ceedpath))\"")
   end
else
   @info "Using prebuilt libCEED binaries provided by libCEED_jll"
   open("deps.jl", write=true) do f
      println(f, "using libCEED_jll")
   end
end
