"""
Utilities for linear algebra.
"""

using LinearAlgebra
import GeoInterface as GI

"""
Compute the cosine similarity between the two vectors.
"""
function cosine_similarity(vec1::T, vec2::T) where T<:Vector{N} where N<:Number
    return normalize(vec1) ⋅ normalize(vec2)
end

"""
Compute the cosine similarity between the two points, using `origin` as
origin.
"""
function cosine_similarity(point1::T, point2::T, origin::Vector{Float64}) where T<:GI.Point
    return cosine_similarity(point1.geom - origin, point2.geom - origin)
end

"""
Compute the cosine similarity between the two points, using `origin` as
origin.
"""
function cosine_similarity(point1::T, point2::T, origin::T) where T<:GI.Point
    return cosine_similarity(point1.geom - origin.geom, point2.geom - origin.geom)
end