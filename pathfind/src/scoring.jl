"""
Scoring functions for pathfinding algorithm.
"""

"""
Calculate scores for roads based on start and destination and previous
road. Higher score is better.
"""
module scoring

import DataFrames as DF
import GeometryOps as GO

include("linalg.jl")
include("geom.jl")

"""
Default scoring function.
"""
function score(start::DF.DataFrameRow, destination::DF.DataFrameRow, prev_road::DF.DataFrame, candidate_roads::DF.DataFrame)
    # the optimal road, in terms of pure distance, would be the one
    # that gets us closest to our destination with the shortest
    # road length. If the road length is L, and it gets us S closer
    # to our destination, 1 <= L/S < ∞ for S > 0 and -∞ < L/S <= -1 for
    # S < 0. Generally, L/S == 1 is optimal, with S > 0 and L/S -> ∞
    # being worse, while for S < 0, L/S == -1 is worst (as we want
    # to go as little in the wrong direction), with L/S -> -∞ being
    # better. However, it would clearly also be useful to encode
    # the actual distance S on its own in the score, as L/S -> -∞
    # could still mean a large negative S.

    # TODO: add penalty for changing roads too frequently? Making turns would generally
    # mean having to slow down; short distances between intersections could
    # be penalised.

    # make sure that these (L and S) are actually the same scale (same units)
    S = GO.distance(destination.geometry, prev_road.intersection) .- GO.distance.([destination.geometry], candidate_roads.intersection)
    L = GO.distance.(prev_road.intersection, candidate_roads.intersection)

    # TODO: add start -> destination direction somehow? Reward paths
    # that stay close to that beeline path; only checking current
    # road -> next road might suffer from the locality.

    # use logarithm?
    _score = L ./ S

    _score_positive_mask = _score .> 0
    _score_negative_mask = .!_score_positive_mask
    # flip around the negative values. L/S closer to -1 is worse, so
    # should use the below to flip it correctly.
    _score[_score_negative_mask] .= (minimum(_score[_score_negative_mask], init=-1.0) - 1) .- _score[_score_negative_mask]

    # limit the score a little. L/S >= 1000 is already a massive
    # difference, so fair enough to limit, arguably.
    _score = ifelse.(abs.(_score) .> 1e3, sign.(_score) * 1e3, _score)

    similarity_mult = score_cosine_similarity(start, destination, prev_road, candidate_roads)
    # cosine_similarity is -1 <= x <= 1, transform so 1 becomes 0 and -1 becomes 2
    # multiply the score, leading to high similarity directions being emphasised
    # more
    similarity_mult .= (abs.(similarity_mult .- 1)) .^ 0.5

    # TODO: might be able to incorporate some of these calculations
    # directly, so no need to calculate all this separately, which could
    # save computation?
    # multiplier for how much closer the path gets (encoding the value of S)
    approach_mult = S ./ maximum(S, init=1.0)
    # multiplying by this is the roughly the same as
    # (k*S/S_max + 1.0)*(L/S)
    # (k*L/S_max + *L/S)
    # where for S = S_max,
    # (L/S_max)*(k + 1)
    # and S = S_min = j*S_max where j <= 1,
    # (L/S_max)*(k + j) <= L/S_max.
    #
    # emphasis on S depends on k < 0, here -0.25: larger absolute value
    # means higher emphasis. The use of the constant term 1.0 causes
    # a separation around S = 0: when S = 0, the multiplier is 1.0.
    # When S < 0, The multiplier is greater than 1.0, causing the negative
    # score to move further away from zero; when, S > 0, the negative score
    # moves closer to zero.
    approach_mult .= -0.25 .* approach_mult .+ 1.0

    _score .= approach_mult .* _score .* similarity_mult
    # ensure that negative scores are all lower than positive ones,
    # and that positive but large scores are lower than positive but
    # small scores:

    # max(neg) - max(pos) < -max(pos) for x > 0 because global max(neg) == -1,
    # so max(neg) < -max(pos) + max(pos) == 0, which is identically true.
    # Further, min(neg) <= max(neg), and -max(pos) <= -min(pos).
    # (neg = negative scores, pos = positive scores)
    _score[_score_negative_mask] .-= maximum(_score[_score_positive_mask], init=1.0)
    _score[_score_positive_mask] .*= -1


    return _score
end

function score_intersection_distance(start::DF.DataFrameRow, destination::DF.DataFrameRow, prev_road::DF.DataFrame, candidate_roads::DF.DataFrame)
    S = -GO.distance.([destination.geometry], candidate_roads.intersection)
    return S
end

"""
Calculate score based on how well current direction matches the direction
of the destination.
"""
function score_cosine_similarity(start::DF.DataFrameRow, destination::DF.DataFrameRow, prev_road::DF.DataFrame, candidate_roads::DF.DataFrame)
    return cosine_similarity.([destination.geometry], candidate_roads.intersection, prev_road.intersection)
end

"""
Calculate score based on how closely a road direction matches with
the start-destination direction.
"""
function score_cosine_similarity_start_destination(
    start::DF.DataFrameRow,
    destination::DF.DataFrameRow,
    prev_road::DF.DataFrame,
    candidate_roads::DF.DataFrame
)
    start_destination = getcoord(destination.geometry) .- getcoord(start.geometry)
    prev_next_road = getcoord.(candidate_roads.intersection) .- getcoord.(prev_road.intersection)
    return cosine_similarity.([start_destination], prev_next_road)
end

end
