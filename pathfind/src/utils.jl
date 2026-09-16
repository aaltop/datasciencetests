module Profiling
using Profile

mutable struct _Env
    profile_filename::String
    alloc_filename::String

    _Env(profile_filename::String, alloc_filename::String) = new(profile_filename, alloc_filename)
end

const _env = _Env("output/profile.txt", "output/alloc.txt")

export @profile_to_file
"""
Profile passed code like [`Profile.@profile`](@ref). Clears the buffer
beforehand and writes automatically to file if the profile buffer has
content after profiling has finished.
"""
macro profile_to_file(ex)
    ex = quote
        Profile.clear()
        ret = @profile $(esc(ex))
        if length(Profile.fetch()) > 0
            write_profile()
        end
        return ret
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
    write_profile(_env.profile_filename)
end

export @alloc_to_file
"""
    @alloc_to_file [sample_rate=0.1] expr

Profile the allocation behaviour of the passed code like
[`Profile.Allocs.@profile`](@ref). Clears the buffer beforehand and writes
automatically to file if any allocations were recorded.
"""
macro alloc_to_file(ex)
    return quote
        @alloc_to_file sample_rate = 0.1 $(esc(ex))
    end
end

macro alloc_to_file(sample_rate, ex)

    return quote
        Profile.Allocs.clear()
        Profile.Allocs.start(; sample_rate=$sample_rate)
        ret = nothing
        try
            ret = $(esc(ex))
        finally
            Profile.Allocs.stop()
        end

        if length(Profile.Allocs.fetch().allocs) > 0
            write_alloc()
        end
        return ret
    end

end

function write_alloc(filename::String)
    open(filename, "w") do f
        Profile.Allocs.print(
            IOContext(f, :displaysize => (24, 500)),
            noisefloor=2.0
        )
    end
end

function write_alloc()
    write_alloc(_env.alloc_filename)
end

end