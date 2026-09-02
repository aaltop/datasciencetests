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

const TURKU_ID = Int32(40294758)
const HELSINKI_ID = Int32(40342733)

_data = _Data()

function plot_route!(ax::Axis, start, destination)

    _plot(ax, options=[])
    plot!(ax, start.geometry)
    plot!(ax, destination.geometry)
    route = subset_by_id(roads(), :OBJECTID => get_path(pathfind(start, destination)), order=:right)
    plot!(ax, route.geometry)

end

const PathfindResult = @NamedTuple{path::DF.DataFrame, destination_id::Int32, found::Bool}

# add roads with own id, parent id, and checked
# status. For the lowest non-checked, find all intersecting, add to end
# of dataframe (in descending order based on how close road is to destination,
# such that the closest road is lowest), repeat until hopefully finding destination road. If
# destination road is found, go back up its ID chain to top.
"""
Find a path between `start` and `destination`. These should rows as
returned by [`search_places`](@ref).

See also: [`get_path`](@ref).
"""
function pathfind(start::DF.DataFrameRow, destination::DF.DataFrameRow)::PathfindResult

    road_chain = DF.DataFrame(id=Int[], parent_id=Union{Int,Nothing}[], checked=Bool[])

    start_roads = closest_roads(start)
    road_chain = vcat(road_chain, DF.DataFrame([:id => start_roads[end:-1:1, :OBJECTID], :parent_id => nothing, :checked => false]))
    destination_road = closest_roads(destination)[1, :]
    max_iter = 1000
    curr_iter = 1
    rd_inter = DF.innerjoin(roads(), road_intersections(), on=:OBJECTID => :id1)

    path_found = false
    while curr_iter <= max_iter
        curr_iter += 1
        not_checked = road_chain[.!road_chain.checked, :]
        if size(not_checked)[1] == 0
            break
        end
        current_road = not_checked[end, :]
        # set ALL roads with the current ID to checked, regardless of what
        # their parent_id is (checked meaning that this road's intersections
        # will have been introduced to the list)
        road_chain[road_chain.id.==current_road.id, :checked] .= true
        current_intersecting = subset_by_id(rd_inter, :id2 => [current_road.id])

        if size(current_intersecting)[1] == 0
            continue
        end

        # add in reverse so that closest can be found at the end
        current_intersecting = current_intersecting[closest(destination.geometry, current_intersecting.geometry)[end:-1:1], :]
        # find the check status of each road; some roads might have already
        # been checked, but all roads from this intersection should also
        # be added because they would create a different path based on
        # their parent_id.
        current_intersecting_checked = DF.leftjoin(current_intersecting, unique(DF.select(road_chain, :id, :checked)), on=:OBJECTID => :id, order=:left).checked
        # println(current_intersecting, current_intersecting_checked)
        # println(current_road.id, subset_by_id(unique(DF.select(road_chain, :id, :checked)), :id => [32]))
        # if it's missing, wasn't in `road_chain`, so checked status is false
        current_intersecting_checked = coalesce.(current_intersecting_checked, false)

        road_chain = vcat(road_chain, DF.DataFrame([
            :id => current_intersecting.OBJECTID,
            :parent_id => current_road.id,
            :checked => current_intersecting_checked
        ]))

        if destination_road.OBJECTID in current_intersecting.OBJECTID
            road_chain[road_chain.id.==destination_road.OBJECTID, :checked] .= true
            path_found = true
        end
    end
    return PathfindResult((; path=road_chain, destination_id=destination_road.OBJECTID, found=path_found))
end

"""
Get the paths (sequences of road IDs) from `pathfind_result` if a path
was found.

See also: [`pathfind`](@ref).
"""
function get_path(pathfind_result::PathfindResult)::Vector{Int}

    if !pathfind_result.found
        return Int[]
    end

    paths, destination_id = pathfind_result.path, pathfind_result.destination_id
    path_road_ids = [destination_id]
    while true

        next_id = paths[paths.id.==path_road_ids[end], :parent_id][1]
        if isnothing(next_id)
            break
        end
        push!(path_road_ids, next_id)

    end
    return reverse(path_road_ids)
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
    # TODO: change to also including roads that surround current municipality
    muni_road_ids = subset_by_id(road_municipality_intersect(), :municipality_id => [location.municipality]).road_id
    muni_roads = DF.innerjoin(roads(), DF.DataFrame([:OBJECTID => muni_road_ids]), on=:OBJECTID)

    return muni_roads[closest(location.geometry, muni_roads.geometry), :]
end

"""
Subset the dataframe `df` by the given ids in `cols`.

`order` allows determining the row ordering of the result, as for
DataFrame joins.
"""
function subset_by_id(df::DF.DataFrame, cols::Pair{Symbol,Vector{N}}...; order=:undefined) where N

    return DF.innerjoin(df, DF.DataFrame([cols...]), on=[col.first for col in cols], order=order)

end

"""
Calculate the intersections of the geometries passed. `with` is the geometry
against which the geometries in the iterable `geoms` are compared.
"""
function intersects(with, geoms)
    return GO.intersects.(geoms, [with])
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
in column 2 (id2). "Bidirectional" here means that any relation of a particular
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