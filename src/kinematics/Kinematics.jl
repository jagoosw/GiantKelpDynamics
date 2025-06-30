using LinearAlgebra

using OceanBioME.Particles: get_node, collapse_position

using Oceananigans.Fields: _fractional_indices, interpolator
using Oceananigans.Grids: AbstractGrid

include("field_functions.jl")

@inline tension(Δx, l₀, Aᶜ, k, α) = ifelse(Δx > l₀ && !(Δx == 0.0), k * (max(0, (Δx - l₀)) / l₀) ^ α * Aᶜ, 0.0)

include("utter_denny.jl")
include("utter_denny_speed.jl")