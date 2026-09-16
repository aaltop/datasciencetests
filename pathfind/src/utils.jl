module Profiling
using Profile

mutable struct _Env
    output_filename::String

    _Env(output_filename::String) = new(output_filename)
end

const _env = _Env("output/profile.txt")

export @profile_to_file
"""
Profile passed code like [`@Profile.profile`](@ref). Clears the buffer
beforehand and writes automatically to file if the profile buffer has
content after profiling has finished.
"""
macro profile_to_file(ex)
    ex = quote
        Profile.clear()
        @profile $(esc(ex))
        if length(Profile.fetch()) > 0
            write_profile()
        end
    end
    return ex
end

function write_profile(filename::String)
    open(filename, "w") do f
        Profile.print(
            IOContext(f, :displaysize => (24, 500)),
            noisefloor=2.0,
        )
    end
end

function write_profile()
    write_profile(_env.output_filename)
end

end