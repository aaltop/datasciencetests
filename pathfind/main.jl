import XML
import DataFrames as DF
import GeoDataFrames as GDF
import GeometryOps as GO
import GeoParquet as GP
import GeoInterface as GI
import Parquet2 as PQ
using GLMakie
using Shapefile

const TURKU_ID = "PNS_40294758"
const HELSINKI_ID = "PNS_40342733"

"""
Parse .xsd files that come with placenames data.
"""
function parse_placenames_xsd(filename::String)

    doc = XML.read("$filename", XML.Node)
    enumerations = XML.xpath(doc, "//xs:enumeration")

    rows::Vector{Dict{String,Union{Int,String}}} = []
    for enumeration in enumerations

        push!(rows, Dict("id" => parse(Int, enumeration["value"])))
        for doc in XML.xpath(enumeration, "//xs:documentation")

            key = doc["xml:lang"]
            rows[end][key] = XML.value(XML.only(doc))

        end


    end

    return DF.DataFrame(rows)

end

mutable struct _Data
    roads::Union{DF.DataFrame,Nothing}
    country::Union{DF.DataFrame,Nothing}
    placenames::Union{DF.DataFrame,Nothing}
    municipality_geom::Union{DF.DataFrame,Nothing}
    road_region_intersect::Union{DF.DataFrame,Nothing}
    road_municipality_intersect::Union{DF.DataFrame,Nothing}

    _Data() = new(nothing, nothing, nothing, nothing, nothing, nothing)
end

_data = _Data()

function roads()
    parquet_file = "data/roads.parquet"
    if isnothing(_data.roads)
        if !isfile(parquet_file)
            df = GDF.read("data/Tieosoiteverkko_hall_lk-Elinvoimakeskus-2026-01-01/Tieosoiteverkko_hall_lk_elinvoimakeskus_01_01_2026.shp")
            GP.write(parquet_file, df)
        end
        _data.roads = GP.read(parquet_file)
    end
    return _data.roads
end

function country()
    if isnothing(_data.country)
        _data.country = GDF.read("data/TietoaKuntajaosta_2026_10k/SuomenValtakunta_2026_10k.shp")
    end
    return _data.country
end

function placenames()
    parquet_file = "data/placenames.parquet"
    if isnothing(_data.placenames)
        if !isfile(parquet_file)
            df = GDF.read("data/placenames_simple_2026_05/placenames_simple.xml")
            df.region = parse.(Int, df.region)
            df.subregion = parse.(Int, df.subregion)
            df.municipality = parse.(Int, df.municipality)

            # join with region, subregion, and municipality names
            df = DF.innerjoin(df, DF.select(regions(), :id, :fin => :region_fin), on=:region => :id)
            df = DF.innerjoin(df, DF.select(subregions(), :id, :fin => :subregion_fin), on=:subregion => :id)
            df = DF.innerjoin(df, DF.select(municipality(), :id, :fin => :municipality_fin), on=:municipality => :id)

            # GeoParquet for some reason requires it to be 'geometry'
            df = DF.rename(df, :placeLocation => :geometry)

            GP.write(parquet_file, df)
        end

        _data.placenames = GP.read(parquet_file)
    end
    return _data.placenames
end

"""Search for places based on placename, region, subregion, and municipality."""
function search_places(; placename::Regex=r"", region::Regex=r"", subregion::Regex=r"", municipality::Regex=r"")

    # maybe a better way to do it than this, but using subset() took
    # much longer, funnily
    #
    # the search terms are ordered here in such a way that earlier
    # ones should limit the number of remaining rows the most, which
    # should in turn limit how many rows need to be searched further
    # down.

    df = placenames()

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

function regions()
    return parse_placenames_xsd("data/placenames_simple_2026_05/region.xsd")
end

function regions_geom()
    reg = GDF.read("data/TietoaKuntajaosta_2026_4500k/SuomenMaakuntajako_2026_4500k.shp")
    return DF.transform(reg, :NATCODE => DF.ByRow(x -> parse(Int, x)) => :NATCODE)
end

function subregions()
    return parse_placenames_xsd("data/placenames_simple_2026_05/subregion.xsd")
end

function municipality()
    return parse_placenames_xsd("data/placenames_simple_2026_05/municipality.xsd")
end

function municipality_geom()
    parquet_file = "data/municipality_geom.parquet"
    if isnothing(_data.municipality_geom)
        if !isfile(parquet_file)

            muni = GDF.read("data/TietoaKuntajaosta_2026_4500k/SuomenKuntajako_2026_4500k.shp")
            muni = DF.transform(muni, :NATCODE => DF.ByRow(x -> parse(Int, x)) => :NATCODE)

            reg = DF.select(regions_geom(), :geometry, :NATCODE => :id)

            intersections = intersects(reg, DF.select(muni, :geometry => DF.ByRow(GO.centroid) => :geometry, :NATCODE => :id))

            intersections = DF.rename(intersections, :id1 => :region_id)

            GDF.write(parquet_file, DF.innerjoin(muni, intersections, on=:NATCODE => :id2))
        end

        _data.municipality_geom = GDF.read(parquet_file)
    end

    return _data.municipality_geom

end

"""
Calculate the intersections of the geometries passed. `with` is the geometry
against which the geometries in the iterable `geoms` are compared.
"""
function intersects(with, geoms)
    return [GO.intersects(g, with) for g in geoms]
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

function get_fig_ax()
    fig = Figure()
    ax = Axis(fig[1, 1])
    return fig, ax
end


function plot_country(ax::Axis)
    plot!(ax, country().geometry; color="#f5deb366", strokecolor="#000F", strokewidth=1)
end

function _plot(ax::Axis)

    empty!(ax)

    plot_country(ax)
    turku = placenames()[placenames().gml_id.==TURKU_ID, :geometry][1]
    hel = placenames()[placenames().gml_id.==HELSINKI_ID, :geometry][1]

    plot!(ax, turku)
    plot!(ax, hel)

    regi = municipality_geom()

    plot!(ax, regi.geometry; color="#0000", strokecolor="#0a0f", strokewidth=1)
    plot!(ax, regions_geom().geometry; color="#0000", strokecolor="#00fa", strokewidth=1)

end