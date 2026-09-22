using Statistics

import XML
import DataFrames as DF
import GeoDataFrames as GDF
import GeometryOps as GO
import GeoInterface as GI
using GLMakie
using Shapefile
using EnumX

include("src/data.jl")
include("src/geom.jl")
include("src/scoring.jl")
include("src/linalg.jl")

const TURKU_ID = Int32(40294758)
const HELSINKI_ID = Int32(40342733)

_data = _Data()

function plot_route!(
    ax::Axis,
    start,
    destination;
    score::Vector{Pair{String,Function}}=Pair{String,Function}["default"=>scoring.score],
    pathfind_kwargs=(;),
    get_path_kwargs=(;),
    plot_kwargs=(;),
    clear_plot=true,
)


    if clear_plot
        _plot(ax, options=[])
        plot!(ax, start.geometry)
        plot!(ax, destination.geometry)
    end
    remove_legends!(ax.parent)
    for (func_name, func) in score

        route_info = get_path(pathfind(start, destination; score=func, pathfind_kwargs...); get_path_kwargs...)
        if length(score) == 1
            _plot_route!(ax, route_info, func_name; color_segments=true, plot_kwargs...)
        else
            _plot_route!(ax, route_info, func_name; plot_kwargs...)
        end




    end
    axislegend(ax)
    return nothing
end

const PathInfo = @NamedTuple{path::Vector{Int}, length::Float64, score::Float64}

function _plot_route!(ax::Axis, route_info::PathInfo, legend_name::String; nearby_roads::Bool=false, color_segments::Bool=false)

    route = subset_by_id(roads(), :OBJECTID => route_info.path, order=:right)


    # find and plot also some roads that intersect with the found path
    # to get an idea of how good the path might be
    if nearby_roads
        route_intersects_ids = subset_by_id(road_intersections(), :id1 => route_info.path).id2
        route_intersects_ids = vcat(route_intersects_ids, subset_by_id(road_intersections(), :id1 => route_intersects_ids).id2)
        route_intersects_ids = setdiff(unique(route_intersects_ids), route.OBJECTID)
        route_intersects = subset_by_id(roads(), :OBJECTID => route_intersects_ids)
        plot!(ax, route_intersects.geometry; color="#0f03")
    end
    label = "$legend_name, length: $(round(route_info.length, digits=2))"
    if color_segments
        # looping so easy to colour each road segment differently
        for (i, rou) in enumerate(route.geometry)
            if i == 1
                plot!(
                    ax,
                    rou,
                    label=label
                )
            end
            plot!(
                ax,
                rou,
            )
        end
    else
        plot!(ax, route.geometry, label=label)
    end

    axislegend()

end

function plot_comparison!(ax::Axis, start, destination; max_iter=500, score=nothing)

    if isnothing(score)
        score = scoring.score
    end

    plot_kwargs = (; color_segments=false)
    plot_route!(ax, start, destination, score=Pair{String,Function}["fast return"=>scoring.score], clear_plot=true, pathfind_kwargs=(; return_fast=true), plot_kwargs=plot_kwargs)
    plot_route!(ax, start, destination, score=Pair{String,Function}["max_iter: $max_iter"=>scoring.score], pathfind_kwargs=(; max_iter=max_iter), clear_plot=false, plot_kwargs=plot_kwargs)

end

function plot_path_length_improvements!(ax::Axis, start, destination, names::Tuple{String,String}, info::String; pathfind_kwargs=(;), save_figure::Bool=false)

    max_iter = get(pathfind_kwargs, :max_iter, 10_000)

    result = pathfind(start, destination; pathfind_kwargs...)
    improvements = DF.DataFrame(result.improvements, [:iter, :length])
    empty!(ax)
    remove_legends!(ax.parent)
    max_length = round(maximum(improvements.length), digits=2)
    min_length = round(minimum(improvements.length), digits=2)
    ax.title = "$(names[1]) -> $(names[2]), max_iter = $max_iter)\nmax $max_length min $min_length\n$info"
    lines!(ax, improvements.iter, improvements.length)
    autolimits!(ax)

    if save_figure

        i = 0
        filename = ""
        while true
            i += 1
            idx = lpad("$i", 4, "0")
            filename = "output/path_performance_$(names[1])_$(names[2])$(idx).png"
            if Base.Filesystem.ispath(filename)
                continue
            end
            break
        end

        save(filename, ax.parent)
    end
end

const PathfindResult = @NamedTuple{path::DF.DataFrame, destination_id::Int32, found::Bool, improvements::Vector{Tuple{Int,Float64}}}

"""
Find a path between `start` and `destination`. These should be rows as
returned by [`search_places`](@ref).

See also: [`get_path`](@ref).
"""
function pathfind(
    start::DF.DataFrameRow,
    destination::DF.DataFrameRow;
    score::Function=scoring.score,
    max_iter::Union{Int,Nothing}=nothing,
    return_fast::Bool=false)#::PathfindResult

    road_chain = DF.DataFrame(
        id=Int[],
        parent_id=Union{Int,Nothing}[],
        # the score for this road segment
        score=Float64[],
        # the sum of the scores on the path up to this road segment
        path_score=Float64[],
        # at each point, the remaining distance to the destination
        distance=Float64[],
        # total length of the path up to this point
        path_length=Float64[],
        # intersection from which this road is entered
        intersection=GI.Point[],
        intersection_id=Union{Int,Nothing}[],
        not_checked=Bool[]
    )

    """
    Calculate an approximate distance from the destination.
    """
    function approx_distance(distance_from_destination::Float64, path_length::Float64)
        return distance_from_destination * 2 + path_length
    end

    function path_length(current_path_length::Float64, current_intersection::GI.Point, next_intersection::GI.Point)
        return GO.distance(current_intersection, next_intersection) + current_path_length
    end

    # Using just the closest road to our start point, because ensuring
    # that the path actually makes it back to the road closest to the
    # start point is more difficult if all roads close to the starting point
    # are considered as starting roads. In most cases, probably fine,
    # but could also be that that road actually doesn't connect to
    # the wider road network, so a path cannot be found at all. It might
    # also be wiser to find the road that is closest to the staring point
    # while also being in the direction of the destination point.
    start_roads = closest_roads(start)[[1], :]
    start_roads.score = [0]

    starting_distance = GO.distance(start_roads.geometry, destination.geometry)
    road_chain = vcat(
        road_chain,
        DF.DataFrame([
            :id => start_roads.OBJECTID,
            :parent_id => nothing,
            :score => start_roads.score,
            :path_score => start_roads.score,
            :distance => starting_distance,
            :path_length => GO.distance.([start.geometry], start_roads.geometry),
            :intersection => start.geometry,
            :intersection_id => nothing,
            :not_checked => true
        ])
    )
    destination_road = closest_roads(destination)[1, :]
    rd_inter = DF.innerjoin(
        DF.select(roads(), :geometry, :OBJECTID),
        DF.rename(road_intersection_points(), :geometry => :intersection),
        on=:OBJECTID => :id1
    )
    # add an "intersection" for the destination and the road closest
    # to it, to be used as the actual target
    rd_inter = vcat(rd_inter, DF.DataFrame([
        :geometry => [destination.geometry, destination_road.geometry],
        :OBJECTID => [0, destination_road.OBJECTID],
        :id2 => [destination_road.OBJECTID, 0],
        # TODO: calculate the actual point on the destination road that
        # is closest to the destination
        :intersection => [destination.geometry, destination.geometry]
    ]))
    rd_inter = hcat(rd_inter, DF.DataFrame([:intersection_id => 1:DF.nrow(rd_inter), :not_checked => true]))

    # not entirely sure why rd_inter loses the typing information above,
    # just do these to speed up computations later on. Would be nicer
    # to just have it retain the typing info in the first place, but
    # not that big of an issue, as this is relatively a small set of operations,
    # timing-wise.
    rd_inter.OBJECTID = Int.(rd_inter.OBJECTID)
    rd_inter.id2 = Int.(rd_inter.id2)
    rd_inter.intersection = GI.Point.(rd_inter.intersection)


    destination_intersection = rd_inter[end-1, :]

    id2_rows = Dict{Int,Vector{Int}}(
        g.id2[1] => Int.(g.row) for g in
        DF.groupby(DF.DataFrame([:id2 => rd_inter.id2, :row => 1:DF.nrow(rd_inter)]), :id2)
    )

    """
    Get rows indices for `rd_inter` where IDs in `id2s` match those of the
    `id2` column in `rd_inter`.
    """
    function get_id2_rows(id2s::Vector{Int})::Vector{Int}
        return vcat(get.([id2_rows], unique(id2s), [Int[]])...)
    end

    if isnothing(max_iter)
        max_iter = 1000
    end

    current_road = road_chain[end, :]
    shortest_path_length = Inf64
    path_found = false
    exhausted = false
    curr_iter = 0
    path_length_change = Tuple{Int,Float64}[]
    found_destination_intersection::Union{DF.DataFrame,Nothing} = nothing
    # 1. Find nearest road to starting point (above)
    # 2. Find intersections for that road
    # 3. Score the roads that those intersections are for: which
    # of the roads has a further intersection that, in whatever way,
    # is best (determined by choice of score function) for getting
    # towards our destination
    # 4. Choose the intersection matching the road with the best score
    # 5. Repeat 2.-4. until the nearest road to destination is reached 
    #
    # With intersections, there can be a large number of possibilities.
    # In the most naive case, given one intersection of a certain road,
    # all other intersections should be considered for which is best.
    # This is also different for each intersection, and not always
    # symmetric in practice (e.g. if we used directions for the roads).
    # Possibly only one road-to-road intersection (between road x -> road y)
    # need be considered if simple pathfinding is the goal. For the purposes
    # here, though it can take more calculation, it may be best to consider
    # all intersection pairs because limiting the pairs would be difficult.
    while curr_iter < max_iter
        curr_iter += 1
        print("\r")
        print("iter $curr_iter/$max_iter")
        found_destination_intersection = nothing


        # find next road on path
        # -------------------------------------------


        if path_found && (curr_iter % 100 == 0) && (DF.nrow(road_chain) > 3000)
            # Remove path points from chosen intersections if they appear to
            # be worse than the shortest found path. Main intention is to
            # improve performance for larger iteration counts, as without
            # this, road_chain might end up with a lot of rows, most of
            # which go unused.
            #
            # With infinite computational resources, this would of course
            # not be done, as it might still remove parts of good paths
            # as well.

            # could use subset!, but not sure if it's really better?
            # seems the recompilation time of the nameless function tends
            # to slow it down?
            road_chain = road_chain[(road_chain.distance.+road_chain.path_length).<1.02*shortest_path_length, :]
        end

        n = DF.nrow(road_chain)
        road_chain_rows = collect(1:n)
        to_check = road_chain_rows[road_chain.not_checked]
        if path_found
            # once a path is found, use gathered info to pick other paths
            # to test: with this heuristic, the closer a road is and
            # the shorter its path thus far, the better the candidate

            road_chain_not_checked = road_chain[road_chain.not_checked, :]
            possibly_shorter = approx_distance.(road_chain_not_checked.distance, road_chain_not_checked.path_length) .* (-road_chain_not_checked.score)
            possibly_shorter_order = sortperm(possibly_shorter, rev=true)
            to_check = to_check[possibly_shorter_order]


        end

        for i in length(to_check):-1:1
            exhausted = i == i
            current_road = road_chain[to_check[i], :]
            # This road's children have already been checked, but with
            # the road itself having a different parent during that previous
            # check. Therefore, the path leading up to this road would
            # be different in this case, and its childrens' path lengths
            # and scores might need to be reassigned, so do that.
            if current_road.id in road_chain.parent_id

                reassigned_ids = Set{Int}(current_road.intersection_id)
                reassign_parents = DF.DataFrameRow[current_road]
                while length(reassign_parents) > 0
                    parent_intersection = pop!(reassign_parents)

                    reassign_children_row = road_chain_rows[road_chain.parent_id.==parent_intersection.id]
                    for row_num in reassign_children_row
                        row = road_chain[row_num, :]

                        # assign the shorter path length, and with
                        # it the corresponding path score 
                        # (whether better or not)
                        new_path_length = path_length(parent_intersection.path_length, parent_intersection.intersection, row.intersection)
                        if row.path_length > new_path_length
                            row.path_length = new_path_length
                            row.path_score = parent_intersection.path_score + row.score
                            if row.id == destination_intersection.OBJECTID
                                shortest_path_length = min(shortest_path_length, row.path_length)
                                push!(path_length_change, (curr_iter, shortest_path_length))
                            end

                            # this child intersection has already been checked,
                            # meaning that it's not the "leaf" in a path that
                            # has been tested — meaning that it has already
                            # passed a given path's length and total score
                            # to further road sections. Therefore, find also
                            # its children and reset the path length and score
                            # for those too
                            if !row.not_checked && !(row.intersection_id in reassigned_ids)
                                push!(reassign_parents, row)
                                # prevent infinite reassign loops
                                push!(reassigned_ids, row.intersection_id)
                            end

                        end

                    end

                end



                # set the road segment as checked, but because its children
                # have already been processed, don't use it as the
                # next road segment
                current_road.not_checked = false
                continue
            end

            # as the road segment was not checked at all, all road
            # segments have not been exhausted yet
            exhausted = false
            current_road.not_checked = false
            # this particular road segment's children have not been checked
            # yet, so do that next
            break
        end

        if exhausted
            break
        end

        # find next road on path
        # ===========================================


        # find possible and suitable road connections
        # -------------------------------------------

        current_intersecting = rd_inter[get_id2_rows([current_road.id]), :]
        if destination_intersection.OBJECTID in current_intersecting.OBJECTID
            found_destination_intersection = subset_by_id(current_intersecting, :OBJECTID => [destination_intersection.OBJECTID])[[1], :]
        end

        current_intersecting.distance = GO.distance.([current_road.intersection], current_intersecting.intersection)
        # find closest intersections for each intersecting road:
        # sort so that closest distances are highest, take
        # first of each OBJECTID group (which should then be the closest)
        current_intersecting = unique(DF.sort(current_intersecting, :distance), :OBJECTID)[:, DF.Not(:distance)]

        if DF.nrow(current_intersecting) == 0
            continue
        end

        next_intersecting = rd_inter[
            get_id2_rows(current_intersecting.OBJECTID),
            :
        ]

        # match rows of current_intersecting to ones in next_intersecting
        # to speed up finding matching intersections later on
        next_intersecting = DF.innerjoin(
            next_intersecting,
            DF.DataFrame([:id2 => current_intersecting.OBJECTID, :current_intersecting_row => 1:DF.nrow(current_intersecting)]),
            on=:id2
        )


        # calculate distance to each further intersection
        next_intersecting.distance = GO.distance.(
            current_intersecting[next_intersecting.current_intersecting_row, :intersection],
            next_intersecting.intersection
        )

        # find the intersections which are closest for each further
        # road connection
        next_intersecting = unique(DF.sort(next_intersecting, :distance), :OBJECTID)
        # remove roads whose intersection is very close for now, mostly
        # because these just make the scoring difficult
        next_intersecting = next_intersecting[.!isapprox.(next_intersecting.distance, 0.0), DF.Not(:distance)]


        # find possible and suitable road connections
        # ===========================================

        if DF.nrow(next_intersecting) == 0
            continue
        end


        # find scores with regard to joining roads
        next_intersecting.score = score(
            start,
            destination,
            # the parents of the next intersecting. Should match the number
            # and order of paths in `next_intersecting`, naturally.
            current_intersecting[next_intersecting.current_intersecting_row, :],
            next_intersecting
        )

        best_scores = unique(DF.sort(next_intersecting, [:current_intersecting_row, DF.order(:score, rev=true)]), :id2)
        # remove current_intersecting intersections that lost
        # to another intersection (had only common intersections, but worse
        # scores for those intersections)
        current_intersecting = current_intersecting[best_scores.current_intersecting_row, :]
        # score based on the most optimal further connection for each immediate
        # road connection
        current_intersecting.score = best_scores.score

        # the algorithm above might not allow the destination to
        # pass through, particularly as it only intersects the one road,
        # so add it back by force
        if !isnothing(found_destination_intersection)
            # TODO: calculate a proper score?
            found_destination_intersection.score = [0.0]
            append!(
                current_intersecting,
                found_destination_intersection
            )
        end
        # highest score at the end
        DF.sort!(current_intersecting, :score)

        append!(road_chain, DF.DataFrame([
            :id => current_intersecting.OBJECTID,
            :parent_id => current_road.id,
            :score => current_intersecting.score,
            :path_score => current_intersecting.score .+ current_road.path_score,
            :distance => GO.distance.([destination.geometry], current_intersecting.intersection),
            :path_length => path_length.([current_road.path_length], [current_road.intersection], current_intersecting.intersection),
            :intersection => current_intersecting.intersection,
            :intersection_id => current_intersecting.intersection_id,
            :not_checked => true
        ]))

        if destination_intersection.OBJECTID in current_intersecting.OBJECTID
            destination_loc = (road_chain.id .== destination_intersection.OBJECTID)
            road_chain[destination_loc, :not_checked] .= false
            path_found = true

            shortest_path_length = min(shortest_path_length, road_chain[destination_loc, :path_length]...)
            push!(path_length_change, (curr_iter, shortest_path_length))
        end

        if return_fast && path_found
            break
        end

    end
    return PathfindResult((;
        path=road_chain,
        destination_id=destination_intersection.OBJECTID,
        found=path_found,
        improvements=unique(x -> x[1], sort(push!(path_length_change, (curr_iter, shortest_path_length))))
    ))
end

"""
Get the paths (sequences of road IDs) from `pathfind_result` if a path
was found.

`sort_by` should be function that returns a sortperm output. It is passed
a dataframe matching the columns of `pathfind_result.path`, and should sort
such that the most optimal roads come first.

See also: [`pathfind`](@ref).

## Examples

```julia
# get possible paths (see `pathfind` for more)
result = pathfind(location1, location2)
# use default sorting method
get_path(result)
# use path score instead for finding "best" path
get_path(result, sort_by = x -> sortperm(x.path_score, rev=true))
# ...or with DataFrames-style sortperm
get_path(result, sort_by = x -> sortperm(x [DF.order(:path_score, rev=true)]))
```

"""
function get_path(pathfind_result::PathfindResult; sort_by::Union{Function,Nothing}=nothing)::PathInfo

    not_found_result = PathInfo((; path=Int[], length=NaN64, score=NaN64))
    if !pathfind_result.found
        return not_found_result
    end

    if isnothing(sort_by)
        sort_by = x -> sortperm(x, [:path_length, DF.order(:path_score, rev=true)])
    end

    paths, destination_id = pathfind_result.path, pathfind_result.destination_id
    paths = hcat(paths[:, DF.Not(:intersection_id)], DF.DataFrame([:intersection_id => 1:DF.nrow(paths)]))

    # get the length and score first
    next_roads = paths[paths.id.==destination_id, :]
    next_roads = next_roads[sort_by(next_roads), :]
    length, score = next_roads[1, :path_length], next_roads[1, :path_score]
    added_intersections = Set()
    path_road_ids = [destination_id]

    # TODO: find all possible paths
    while true

        next_roads = paths[paths.id.==path_road_ids[end], :]
        next_roads = next_roads[sort_by(next_roads), :]

        next_id = -1
        for row in eachrow(next_roads)
            if row.intersection_id in added_intersections
                continue
            end
            next_id = row.parent_id
            push!(added_intersections, row.intersection_id)
            break
        end

        if next_id == -1
            return not_found_result
        end

        if isnothing(next_id)
            break
        end
        push!(path_road_ids, next_id)
    end
    return PathInfo((; path=reverse(path_road_ids), length=length, score=score))
end

"""
Search for places based on placename, region, subregion, and municipality, or on IDs.


If `ids` is passed, only considers those for search. When searching by ID, the
order of rows returned is the same as the passed IDs, IDs not found represented
by empty rows.
"""
function search_places(; placename::Regex=r"", region::Regex=r"", subregion::Regex=r"", municipality::Regex=r"", ids::Vector{Int32}=Int32[])

    # maybe a better way to do it than this, but using subset() took
    # much longer, funnily
    #
    # the search terms are ordered here in such a way that earlier
    # ones should limit the number of remaining rows the most, which
    # should in turn limit how many rows need to be searched further
    # down.

    df = placenames()

    if length(ids) > 0
        return DF.rightjoin(df, DF.DataFrame([:placeNameId => ids]), on=:placeNameId, order=:right)
    end

    if municipality != r""
        df = df[occursin.(municipality, df.municipality_fin), :]
    end

    if subregion != r""
        df = df[occursin.(subregion, df.subregion_fin), :]
    end

    if region != r""
        df = df[occursin.(region, df.region_fin), :]
    end

    if placename != r""
        df = df[occursin.(placename, df.spelling), :]
    end

    return df
end

"""
Calculate which geometries in `candidates` are closest to those in `to`.

Returns the index ordering of the values in `candidates`, from closest
to farthest.
"""
function closest(to, candidates)
    return sortperm(GO.distance.([to], candidates))
end


"""
Find the roads that are closest to `location`.

`location` should be a row such as returned by [`search_places`](@ref).
"""
function closest_roads(location)
    muni_roads = municipality_roads(location)
    return muni_roads[closest(location.geometry, muni_roads.geometry), :]
end

"""
Find the roads that are in the same municipality as `location`.

`location` should be a row such as returned by [`search_places`](@ref).
"""
function municipality_roads(location)
    # TODO: allow including neighboring municipalities?
    muni_road_ids = subset_by_id(road_municipality_intersect(), :municipality_id => [location.municipality]).road_id
    muni_roads = DF.innerjoin(roads(), DF.DataFrame([:OBJECTID => muni_road_ids]), on=:OBJECTID)
    return muni_roads
end

"""
Subset the dataframe `df` by the given ids in `cols`.

`order` allows determining the row ordering of the result, as for
DataFrame joins.
"""
function subset_by_id(df, cols::Pair{Symbol,Vector{N}}...; order=:undefined) where N

    return DF.innerjoin(df, DF.DataFrame([cols...]), on=[col.first for col in cols], order=order)

end

"""
Calculate whether the passed geometries intersect. `with` is the geometry
against which the geometries in the iterable `geoms` are compared.
"""
function intersects(with, geoms)
    return GO.intersects.(geoms, [with])
end

"""
Calculate the intersection points of the passed geometries. `with` is
the geometry against which the geometries in the iterable `geoms` are
compared.

Returns a vector of vector of tuples, where each tuple represents an
intersection point.
"""
function intersection(with, geoms)
    return GO.intersection.([with], geoms, target=GI.PointTrait())
end


"""
Calculate the intersections of geometries in the two dataframes. Both
dataframes are expected to contain a column `geometry` (of geometries),
and a column `id` of IDs per geometry.

Returns DataFrame with columns `id1`, `id2` where `id1` matches
IDs in `df1` and `id2` matches IDs in `df2`. Each row indicates an
intersection between the geometries that match `id1` and `id2`.
"""
function intersects(df1::DF.DataFrame, df2::DF.DataFrame)

    intersections::Vector{DF.DataFrame} = []
    for i in 1:size(df1)[1]
        g = df1[i, :]
        temp_df = DF.DataFrame([:id1 => g.id, :id2 => df2[intersects(g.geometry, df2.geometry), :id]])
        push!(intersections, temp_df)
    end
    return DF.vcat(intersections...)
end

function road_region_intersect()
    parquet_file = "data/road_region_intersect.parquet"
    if isnothing(_data.road_region_intersect)
        if !isfile(parquet_file)

            regi = regions_geom()
            road_per_region = []
            n = size(regi)[1]
            for i in 1:n
                print("\r")
                print("road-region intersection calculation $i/$n")
                reg = regi[i, :]
                ids = roads()[intersects(reg.geometry, roads().geometry), :OBJECTID]
                push!(road_per_region, DF.DataFrame([:region_id => reg.NATCODE, :road_id => ids]))
            end

            println()
            PQ.writefile(parquet_file, DF.vcat(road_per_region...))
        end

        _data.road_region_intersect = DF.DataFrame(PQ.Dataset(parquet_file); copycols=false)
    end

    return _data.road_region_intersect
end

function road_municipality_intersect()
    parquet_file = "data/road_municipality_intersect.parquet"
    if isnothing(_data.road_municipality_intersect)
        if !isfile(parquet_file)

            regi = municipality_geom()
            road_per_muni = []

            roads_per_region = DF.innerjoin(roads(), road_region_intersect(), on=:OBJECTID => :road_id)
            total_municipalities = size(regi)[1]
            current_muni = 1
            for region_id in unique(road_region_intersect().region_id)
                region_roads = DF.innerjoin(roads_per_region, DF.DataFrame([:region_id => region_id]), on=:region_id)

                # find the municipalities that match the current region
                municipalities = DF.innerjoin(regi, DF.DataFrame([:region_id => region_id]), on=:region_id)
                n = size(municipalities)[1]
                # compare the region's municipalities against only that region's roads
                for i in 1:n
                    print("\r")
                    print("road-municipality intersection calculation $current_muni/$total_municipalities")
                    reg = municipalities[i, :]
                    ids = region_roads[intersects(reg.geometry, region_roads.geometry), :OBJECTID]
                    push!(road_per_muni, DF.DataFrame([:municipality_id => reg.NATCODE, :road_id => ids]))
                    current_muni += 1
                end
            end

            println()
            PQ.writefile(parquet_file, DF.vcat(road_per_muni...))
        end

        _data.road_municipality_intersect = DF.DataFrame(PQ.Dataset(parquet_file); copycols=false)
    end

    return _data.road_municipality_intersect
end

function road_intersection_points()
    parquet_file = "data/road_intersection_points.parquet"
    if !isfile(parquet_file)
        # TODO: figure out bidirectionality
        GDF.write(parquet_file, _road_intersection_points(roads(), road_intersections()))
    end
    if !isnothing(_data.road_intersection_points)
        return _data.road_intersection_points
    end
    df = GDF.read(parquet_file)
    # GDF REALLY likes transforming everything into WellKnownBinary,
    # transform back into points here; using getcoord returns the actual
    # Float Vector of the point, rather than some UInt8 Vector.
    df.geometry = df.geometry .|> getcoord .|> GI.Point
    _data.road_intersection_points = df
    return df

end

function _road_intersection_points(road_df, road_intersection_df)

    roads_to_consider = intersect(unique(road_intersection_df.id1), road_df.OBJECTID)

    road_lines = [GI.LineString([(r.points[i].x, r.points[i].y, r.zvalues[i]) for i in 1:length(r.points)]) for r in road_df.geometry]
    lines_df = DF.DataFrame([:id => road_df.OBJECTID, :lines => road_lines])
    all_lines_df = DF.select(DF.innerjoin(road_intersection_df, lines_df, on=:id1 => :id), :id1, :id2, :lines => :lines1)
    all_lines_df = DF.select(DF.innerjoin(all_lines_df, lines_df, on=:id2 => :id), :id1, :id2, :lines1, :lines => :lines2)
    intersection_points = DF.DataFrame([:id1 => Int64[], :id2 => Int64[], :geometry => GI.Point[]])

    n = length(roads_to_consider)

    id_groups = DF.groupby(all_lines_df, :id1)
    println("Calculating road intersections points...")
    for (i, gr) in enumerate(id_groups)
        id1 = gr.id1[1]
        if (i % 10 == 0)
            print("\r")
            print("road $i/$n")
        end

        # don't recalculate intersections
        current_roads = subset_by_id(gr, :id2 => setdiff(gr.id2, intersection_points.id1))

        if size(current_roads)[1] < 1
            continue
        end

        inter = intersection(current_roads.lines1[1], current_roads.lines2)
        for (id2, inter_points) in zip(current_roads.id2, inter)
            intersection_points = vcat(intersection_points, DF.DataFrame([:id1 => id1, :id2 => id2, :geometry => GI.Point.(inter_points)]))
        end
    end

    intersection_points.id1 = Int64.(intersection_points.id1)
    intersection_points.id2 = Int64.(intersection_points.id2)

    # make bidirectional so that the intersections of a given road
    # can be found by simply subsetting one of the ID columns
    intersection_points = vcat(intersection_points, DF.rename(intersection_points, :id1 => :id2, :id2 => :id1))

    return intersection_points

end

"""
Return a dataframe of road intersections. Column `id1` is a road id,
and column `id2` has the IDs of roads it intersects with.
"""
function road_intersections()
    parquet_file = "data/road_intersections.parquet"
    if !isfile(parquet_file)
        df = _road_intersections(DF.select(roads(), :OBJECTID => :id, :geometry))
        PQ.writefile(parquet_file, bidirectional_associative_table(df, :id1, :id2))
    end
    return DF.DataFrame(PQ.readfile(parquet_file); copycols=false)
end

"""
Calculate intersections of roads. `road_df` should contain columns of
`geometry` and `id`.

Returns a dataframe with columns `id1` and `id2` representing the
intersecting roads.
"""
function _road_intersections(road_df::DF.DataFrame)

    println("Calculating convex hulls for road intersection calculation...")
    # using convex hulls to quickly calculate viable candidates
    # for intersection. Bounding boxes might be faster, but insanely enough,
    # there's no ready-made calculation for that in GO (don't know if
    # BBoxes are much a thing when working non-euclidean systems, but to
    # my understanding one country would still generally be approximated
    # as a manifold), and I'm not spending
    # time sorting that out myself. There IS a minimum bounding circle calculation
    # in GO, but it's excrutiatingly slow for some reason; I guess it
    # might just be a slow calculation in general, don't really know.
    # Anyway, a close-to-minimum bounding circle would be fastest for
    # this comparison, I think, but this'll do for now.
    convex_hulls = DF.transform(road_df, :geometry => (x -> GO.convex_hull.(x)) => :geometry)

    n = size(road_df)[1]
    intersections = []

    muni = DF.innerjoin(road_municipality_intersect(), DF.select(road_df, :id), on=:road_id => :id)
    roads_calculated = 1
    total_roads = size(muni)[1]
    for muni_id in unique(muni.municipality_id)

        # only consider the roads that are in the same municipality
        same_muni_road_ids = DF.innerjoin(muni, DF.DataFrame([:municipality_id => muni_id]), on=:municipality_id)
        same_muni_roads = DF.innerjoin(road_df, same_muni_road_ids, on=:id => :road_id)
        same_muni_hulls = DF.innerjoin(convex_hulls, same_muni_road_ids, on=:id => :road_id)

        n = size(same_muni_hulls)[1]
        for i in 1:(n-1)
            print("\r")
            # a *rough* idea of how many left to calculate
            print("Road intersection calculation $roads_calculated/$total_roads")

            # don't repeat calculations; intersection is symmetric in
            # its operands, so geom2 intersects with geom1 if and only
            # if geom1 intersects with geom2
            compare_road = same_muni_roads[i, :]
            other_slice = i+1:n
            # only compare against roads whose convex hull overlap that
            # of the compared road
            other_road = same_muni_roads[other_slice, :][intersects(same_muni_hulls[i, :geometry], same_muni_hulls[other_slice, :geometry]), :]

            push!(
                intersections,
                DF.DataFrame(
                    [:id1 => compare_road.id, :id2 => other_road.id[intersects(compare_road.geometry, other_road.geometry)]]
                )
            )
            roads_calculated += 1
        end


    end
    println()
    return DF.vcat(intersections...)

end

"""
Calculate a bidirectional associative table. The table is assumed to
be an associative one, where IDs in column 1 (id1) are mapped to IDs
in column 2 (id2).

"Bidirectional" here means that any relation of a particular
ID in id1 and id2 is is added to id1, such that to find the associations
of that ID, it is only necessary to filter based on that ID in either of
the two columns id1 and id2.

## Examples

```
Non-bidirectional:
---
id1 | id2
1 | 2
3 | 1
---

Bidirectional:
---
id1 | id2
1 | 2
2 | 1
3 | 1
1 | 3
---
```

"""
function bidirectional_associative_table(df::DF.DataFrame, id1::Symbol, id2::Symbol)

    col1, col2 = df[:, id1], df[:, id2]
    DF.rename(DF.DataFrame(Set(Pair(p...) for p in vcat(zip(col1, col2)..., zip(col2, col1)...))), :first => id1, :second => id2)

end

function get_fig_ax()
    fig = Figure()
    ax = Axis(fig[1, 1])
    return fig, ax
end


function plot_country(ax::Axis)
    plot!(ax, country().geometry; color="#f5deb366", strokecolor="#000F", strokewidth=1)
    _set_lims!(ax)
end

@enumx _PlotOption begin
    municipality
    region
end

function _plot(ax::Axis; options::Vector=[instances(_PlotOption.T)...])

    empty!(ax)
    remove_legends!(ax.parent)

    plot_country(ax)

    if _PlotOption.municipality in options
        plot!(ax, municipality_geom().geometry; color="#0000", strokecolor="#0a0f", strokewidth=1)
    end
    if _PlotOption.region in options
        plot!(ax, regions_geom().geometry; color="#0000", strokecolor="#00fa", strokewidth=1)
    end
end

function _set_lims!(ax)
    ylims!(ax, [6.55e6, 7.8e6])
    xlims!(ax, [-1e6, 1.5e6])
end

function remove_legends!(fig::Figure)
    foreach(delete!, filter(x -> isa(x, Legend), fig.content))
end