import XML
import DataFrames as DF
import GeoDataFrames as GDF
import GeometryOps as GO
import GeoParquet as GP
using GLMakie
using Shapefile

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

    _Data() = new(nothing, nothing, nothing)
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

function subregions()
    return parse_placenames_xsd("data/placenames_simple_2026_05/subregion.xsd")
end

function municipality()
    return parse_placenames_xsd("data/placenames_simple_2026_05/municipality.xsd")
end

function get_fig_ax()
    fig = Figure()
    ax = Axis(fig[1, 1])
    return fig, ax
end

function plot_country(ax::Axis)
    plot!(ax, country().geometry; color="#f5deb366", strokecolor="#000F", strokewidth=1)
end