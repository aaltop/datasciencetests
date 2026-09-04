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

"""
Default scoring function.
"""
function score(start::DF.DataFrameRow, destination::DF.DataFrameRow, prev_road::DF.DataFrameRow, candidate_roads::DF.DataFrame)
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


    # Distance is *probably* the distance between the closest points
    # of the two geometries, need to check this out. if so, simply
    # calculating it like this won't work: if a road curves, say a
    # overall u-turn, the closest it might be could be in the middle
    # of the road, whereas the actual exit from that road might
    # be further away:
    #   
    #   v-- entrance
    #   -------
    #         |
    #         |<-- "closest" point (no intersecting road) ### destination --> x
    #         |
    #   -------
    #   ^-- exit
    #
    # Equally, a road can have multiple exits, and maybe the road
    # *does* have an exit at the middle of the u-turn. What would
    # actually need to be done is to find the intersection points
    # of each road with its intersecting roads, and find the closest
    # of these to the destination. Still, this should be enough
    # for now, for purposes of testing.

    # make sure that these (L and S) are actually the same scale (same units)
    S = GO.distance(destination.geometry, prev_road.geometry) .- GO.distance.([destination.geometry], candidate_roads.geometry)
    # the distance here is presumably the actual "total length of
    # road" on this road, meaning that it wouldn't obviously always
    # be just the distance travelled in a specific direction; for
    # example, a round-about would be longer than most of the distances
    # travelled on it. So, actually not that ideal. Especially highways
    # might have a lot of extra road, which would skew.
    L = candidate_roads.AJR_PITUUS

    # TODO: add start -> destination direction somehow? Reward paths
    # that stay close to that beeline path; only checking current
    # road -> next road might suffer from the locality.

    # use logarithm?
    _score = L ./ S
    # limit the score a little. L/S >= 1000 is already a massive
    # difference, so fair enough to limit, arguably.
    _score = ifelse.(abs.(_score) .> 1e3, sign.(_score) * 1e3, _score)
    _score_positive_mask = _score .> 0
    # ensure that negative scores are all lower than positive ones,
    # and that positive but large scores are lower than positive but
    # small scores:
    # 
    # max(neg) - max(pos) < -max(pos) for x > 0 because global max(neg) == -1,
    # so max(neg) < -max(pos) + max(pos) == 0, which is identically true.
    # Further, min(neg) <= max(neg), and -max(pos) <= -min(pos).
    # (neg = negative scores, pos = positive scores)
    _score[_score.<0] .-= maximum(_score[_score_positive_mask], init=1.0)
    _score[_score_positive_mask] .*= -1

    return _score
end

"""
Score based on distance of candidate roads from destination.
"""
function score_road_distance(start::DF.DataFrameRow, destination::DF.DataFrameRow, prev_road::DF.DataFrameRow, candidate_roads::DF.DataFrame)

    S = -GO.distance.([destination.geometry], candidate_roads.geometry)
    return S

end

end
