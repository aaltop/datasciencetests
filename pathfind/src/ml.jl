

module metrics

function area_under_curve(borders::Vector{N1}, heights::Vector{N2}; normalize=true) where {N1<:Real,N2<:Real}

    widths = borders[2:end] .- borders[1:end-1]

    avg_heights = (heights[1:end-1] .+ heights[2:end]) ./ 2

    height_extrema = extrema(heights)
    width_extrema = extrema(borders)

    total_width = width_extrema[2] - width_extrema[1]
    total_height = height_extrema[2]

    ret = (widths .* avg_heights)
    # without this, it's effectively just numerical intergration
    if normalize
        ret ./= (total_height * total_width)
    end

    return sum(ret)

end

end