"""
Data utilities.
"""

import XML
import DataFrames as DF
import GeoDataFrames as GDF
import Parquet2 as PQ
import GeoParquet as GP

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

function roads()
    if isnothing(_data.roads)
        _data.roads = GDF.read("data/Tieosoiteverkko_hall_lk-Elinvoimakeskus-2026-01-01/Tieosoiteverkko_hall_lk_elinvoimakeskus_01_01_2026.shp")
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