
"""
    set!(kelp::GiantKelp; kwargs...)

Sets the properties of the `kelp` model. The keyword arguments kwargs... take the form name=data, where name refers to one of the properties of
`kelp`, and the data may be an array mathcing the size of the property for one individual (i.e. size(kelp.name[1])), or for all (i.e. size(kelp.name)).

Example
=======

```jldoctest
julia> using GiantKelpDynamics, Oceananigans

julia> grid = RectilinearGrid(size=(16, 16, 16), extent=(100, 100, 8));

julia> kelp = GiantKelp(; grid, number_nodes = 2, holdfast_x = [10., 20.], holdfast_y = [10., 20])
Giant kelp (Macrocystis pyrifera) model with 2 individuals of 2 nodes.
 Base positions:
 - x ∈ [10.0, 20.0]
 - y ∈ [10.0, 20.0]
 - z ∈ [-8.0, -8.0]

julia> set!(kelp, positions = (x = [0, 8], y = zeros(2), z = [0, 8]))

```
"""
function set!(kelp::GiantKelp; kwargs...)
    for (fldname, value) in kwargs
        ϕ = getproperty(kelp, fldname)
        set!(ϕ, value)
    end
end

function set!(ϕ::NamedTuple, value)
    set!(ϕ.x, value.x)
    set!(ϕ.y, value.y)
    set!(ϕ.z, value.z)
end

const NotAField = Union{Array, CuArray}

set!(ϕ::NotAField, value::Number) = ϕ .= value
set!(ϕ::AbstractArray{FT1, 1}, value::AbstractArray{FT2, 1}) where {FT1, FT2} = ϕ.= value
set!(ϕ::AbstractArray{FT1, 2}, value::AbstractArray{FT2, 2}) where {FT1, FT2} = ϕ.= value

function set!(ϕ, value)
    if length(size(value)) == 1
        set_1d!(ϕ, value)
    elseif size(value) == size(ϕ)
        set!(ϕ, on_architecture(architecture(ϕ), value))
    else
        error("Failed to set property with size $(size(ϕ)) to values with size $(size(value))")
    end
end

function set_1d!(ϕ, value)
    for n in eachindex(value)
        ϕ[:, n] .= value[n]
    end
end

# for output writer

const PropertyArray = Union{Array, CuArray}

fetch_output(output::Array, model) = output

fetch_output(output::CuArray, model) = on_architecture(CPU(), output)

function convert_output(output::CuArray, writer)
    output_array = writer.array_type(undef, size(output)...)
    copyto!(output_array, output)

    return output_array
end

"""
    NothingBGC()

An Oceananigans `AbstractContinuousFormBiogeochemistry` which specifies no biogeochemical
interactions to allow the giant kelp model to be run alone.

Example
=======

```jldoctest
julia> using GiantKelpDynamics, Oceananigans, OceanBioME

julia> grid = RectilinearGrid(size=(16, 16, 16), extent=(100, 100, 8));

julia> kelp = GiantKelp(; grid, number_nodes = 2, holdfast_x = [10., 20.], holdfast_y = [10., 20])
Giant kelp (Macrocystis pyrifera) model with 2 individuals of 2 nodes.
 Base positions:
 - x ∈ [10.0, 20.0]
 - y ∈ [10.0, 20.0]
 - z ∈ [-8.0, -8.0]

julia> biogeochemistry = Biogeochemistry(NothingBGC(); particles = kelp)
No biogeochemistry
 Light attenuation: Nothing
 Sediment: Nothing
 Particles: Giant kelp (Macrocystis pyrifera) model with 2 individuals of 2 nodes.
 Modifiers: Nothing

```
"""
struct NothingBGC <: AbstractContinuousFormBiogeochemistry end

summary(::NothingBGC) = string("No biogeochemistry")
show(io, ::NothingBGC) = print(io, string("No biogeochemistry"))
show(::NothingBGC) = string("No biogeochemistry") # show be removed when show for `Biogeochemistry` is corrected