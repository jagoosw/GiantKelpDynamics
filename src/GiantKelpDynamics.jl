"""
A coupled model for the motion (and in the future growth and biogeochemical interactions) of giant kelp (Macrocystis pyrifera).

Based on the models proposed by [Utter1996](@citet) and [Rosman2013](@citet), and used in [StrongWright2023](@citet).

Implemented in the framework of OceanBioME.jl[StrongWright2023](@citep) and the coupled with the fluid dynamics of Oceananigans.jl[Ramadhan2020](@citep).
"""
module GiantKelpDynamics

export GiantKelp, NothingBGC, RK3, Euler, UtterDenny

using Adapt, CUDA

using KernelAbstractions: @kernel, @index, synchronize
using Oceananigans: CPU

using KernelAbstractions.Extras: @unroll
using OceanBioME.Particles: BiogeochemicalParticles, atomic_add!
using Oceananigans: Center, Face
using Oceananigans.Architectures: architecture, device, on_architecture
using Oceananigans.Biogeochemistry: AbstractContinuousFormBiogeochemistry
using Oceananigans.Fields: Field, CenterField, VelocityFields
using Oceananigans.Grids: znodes
using Oceananigans.Operators: volume

using OceanBioME.Particles: AbstractBiogeochemicalParticles

import Adapt: adapt_structure
import Base: size, length, show, summary
import Oceananigans: set!
import Oceananigans.Biogeochemistry: update_tendencies!
import Oceananigans.Models.LagrangianParticleTracking: update_lagrangian_particle_properties!, _advect_particles!
import Oceananigans.OutputWriters: fetch_output, convert_output

struct GiantKelp{FT, VT, MT, TM, KP, TS, DT, TF, CD} <: AbstractBiogeochemicalParticles
    scalefactor :: VT

    #information about nodes
               positions :: TM
              velocities :: TM 
         relaxed_lengths :: MT
             blade_areas :: MT

    # forces on nodes and force history
        accelerations :: TM
       old_velocities :: TM
    old_accelerations :: TM
          drag_forces :: TM

          stipe_radii :: FT
pneumatocyst_buoyancy :: FT

    kinematics :: KP

    timestepper :: TS
         max_Δt :: DT

     tracer_forcing :: TF
    custom_dynamics :: CD

    function GiantKelp(scalefactor::VT,
                       positions::TM,
                       velocities::TM,
                       relaxed_lengths::MT,
                       blade_areas::MT,
                       accelerations::TM,
                       old_velocities::TM,
                       old_accelerations::TM,
                       drag_forces::TM,
                       stipe_radii::FT,
                       pneumatocyst_buoyancy::FT,
                       kinematics::KP,
                       timestepper::TS,
                       max_Δt::DT,
                       tracer_forcing::TF,
                       custom_dynamics::CD) where {FT, VT, MT, TM, KP, TS, DT, TF, CD}

        return new{FT, VT, MT, TM, KP, TS, DT, TF, CD}(scalefactor,
                                                       positions,
                                                       velocities,
                                                       relaxed_lengths,
                                                       blade_areas,
                                                       accelerations,
                                                       old_velocities,
                                                       old_accelerations,
                                                       drag_forces,
                                                       stipe_radii,
                                                       pneumatocyst_buoyancy,
                                                       kinematics,
                                                       timestepper,
                                                       max_Δt,
                                                       tracer_forcing,
                                                       custom_dynamics)
    end
end

include("timesteppers.jl")
include("kinematics/Kinematics.jl")
include("drag_coupling.jl")

function segment_area_fraction(lengths)
    fractional_length = cumsum(lengths) ./ sum(lengths)

    # Jackson et al. 1985 (https://www.jstor.org/stable/24817427)
    cumulative_areas = -0.08 .+ 3.3 .* fractional_length .- 4.1 .* fractional_length .^ 2 .+ 1.9 .* fractional_length .^ 3

    return reverse(cumulative_areas .- [0.0, cumulative_areas[1:end-1]...])
end

"""
    nothingfunc(args...)

Returns nothing for `nothing(args...)`
"""
@inline nothingfunc(args...) = nothing

"""
    GiantKelp(; grid, 
                holdfast_x, holdfast_y, holdfast_z,
                scalefactor = ones(length(holdfast_x)),
                number_nodes = 8,
                segment_unstretched_length = 3.,
                initial_stipe_radii = 0.004,
                initial_blade_areas = 3.0 * (isa(segment_unstretched_length, Number) ? 
                                               ones(number_nodes) ./ number_nodes :
                                               segment_area_fraction(segment_unstretched_length)),
                initial_pneumatocyst_volume = (2.5 / (5 * 9.81)) .* (isa(segment_unstretched_length, Number) ?
                                                                       1 / number_nodes .* ones(number_nodes) :
                                                                       segment_unstretched_length ./ sum(segment_unstretched_length)),
                kinematics = UtterDenny(),
                timestepper = Euler(),
                max_Δt = Inf,
                tracer_forcing = NamedTuple(),
                custom_dynamics = nothingfunc)

Constructs a model of giant kelps with bases at `holdfast_x`, `_y`, `_z`.


Keyword Arguments
=================

- `grid`: (required) the geometry to build the model on
- `holdfast_x`, `holdfast_y`, `holdfast_z`: An array of the base/holdfast positions of the individuals
- `scalefactor`: array of the scalefactor for each plant (used to allow each plant model to represnt the effect of multiple individuals)
- `number_nodes`: the number of nodes to split each individual interior
- `segment_unstretched_length`: either a scalar specifying the unstretched length of all segments, 
   or an array of the length of each segment (at the moment each plant must have the same)
- `initial_stipe_radii`: either a scalar specifying the stipe radii of all segments, 
   or an array of the stipe radii of each segment (at the moment each plant must have the same)
- `initial_blade_areas`: an array of the blade area attatched to each segment
- `initial_pneumatocyst_volume`: an array of the volume of pneumatocyst attatched to each segment
- `kinematics`: the kinematics model specifying the individuals motion
- `timestepper`: the timestepper to integrate the motion with (at each substep)
- `max_Δt`: the maximum timestep for integrating the motion
- `tracer_forcing`: a `NamedTuple` of `Oceananigans.Forcings(func; field_dependencies, parameters)` with for discrete form forcing only. Functions 
  must be of the form `func(i, j, k, p, n, grid, clock, tracers, particles, parameters)` where `field_dependencies` can be particle properties or 
  fields from the underlying model (tracers or velocities)
- `custom_dynamics`: function of the form `func(particles, model, bgc, Δt)` to be executed at every timestep after the kelp model properties are updated.

Example
=======

```jldoctest
julia> using GiantKelpDynamics, Oceananigans

julia> grid = RectilinearGrid(size=(16, 16, 16), extent=(100, 100, 8));

julia> kelp = GiantKelp(; grid, holdfast_x = [10., 20.], holdfast_y = [10., 20], holdfast_z = [-8., -8.])
Giant kelp (Macrocystis pyrifera) model with 2 individuals of 8 nodes. 
 Base positions:
 - x ∈ [10.0, 20.0]
 - y ∈ [10.0, 20.0]
 - z ∈ [-8.0, -8.0]

```
"""
function GiantKelp(; grid, 
                     holdfast_x, holdfast_y,
                     scalefactor = ones(length(holdfast_x)),
                     number_nodes = 8,
                     segment_unstretched_length = 3.,
                     stipe_radii = 0.004,
                     initial_blade_areas = sum(segment_unstretched_length)^0.995*0.297 * (isa(segment_unstretched_length, Number) ? 
                                                    ones(number_nodes) ./ number_nodes :
                                                    segment_area_fraction(segment_unstretched_length)),
                     pneumatocyst_buoyancy = 2.5, #kg m/s²
                     kinematics = UtterDennySpeed(),
                     timestepper = Euler(),
                     max_Δt = nothing,
                     tracer_forcing = NamedTuple(),
                     custom_dynamics = nothingfunc)

    number_kelp = length(holdfast_x)

    arch = architecture(grid)

    scalefactor = on_architecture(arch, scalefactor)
    
    positions = threeD_array(number_kelp, number_nodes+1, arch; z0 = znodes(grid, Center(), Center(), Face())[1])

    CUDA.@allowscalar begin
        positions.x .= holdfast_x
        positions.y .= holdfast_y
    end

    velocities = threeD_array(number_kelp, number_nodes+1, arch)

    relaxed_lengths = on_architecture(arch, ones(number_kelp, number_nodes))
    blade_areas = on_architecture(arch, ones(number_kelp, number_nodes))

    set!(relaxed_lengths, segment_unstretched_length)
    set!(blade_areas, initial_blade_areas)

    accelerations = threeD_array(number_kelp, number_nodes+1, arch)
    old_velocities = threeD_array(number_kelp, number_nodes+1, arch)
    old_accelerations = threeD_array(number_kelp, number_nodes+1, arch)
    drag_forces = threeD_array(number_kelp, number_nodes+1, arch)

    if isnothing(max_Δt)
        max_Δt = on_architecture(arch, ones(number_kelp) * 0.001)
    end

    return GiantKelp(scalefactor,
                     positions,
                     velocities,
                     relaxed_lengths,
                     blade_areas,
                     accelerations,
                     old_velocities, 
                     old_accelerations,
                     drag_forces,
                     stipe_radii,
                     pneumatocyst_buoyancy,
                     kinematics,
                     timestepper,
                     max_Δt,
                     tracer_forcing,
                     custom_dynamics)
end

threeD_array(d1, d2, arch; x0 = 0, y0 = 0, z0 = 0) = 
    (x = on_architecture(arch, ones(d1, d2) * x0),
     y = on_architecture(arch, ones(d1, d2) * y0),
     z = on_architecture(arch, ones(d1, d2) * z0))

adapt_structure(to, kelp::GiantKelp) = GiantKelp(adapt(to, kelp.scalefactor),
                                                 adapt(to, kelp.positions),
                                                 adapt(to, kelp.velocities),
                                                 adapt(to, kelp.relaxed_lengths),
                                                 adapt(to, kelp.blade_areas),
                                                 adapt(to, kelp.accelerations),
                                                 adapt(to, kelp.old_velocities),
                                                 adapt(to, kelp.old_accelerations),
                                                 adapt(to, kelp.drag_forces),
                                                 stipe_radii,
                                                 pneumatocyst_buoyancy,
                                                 adapt(to, kelp.kinematics),
                                                 nothing,
                                                 adapt(to, kelp.max_Δt),
                                                 nothing,
                                                 nothing)

size(particles::GiantKelp) = size(particles.positions.x)
length(particles::GiantKelp) = length(particles.scalefactor)

size(particles::GiantKelp, dim::Int) = size(particles.positions.x, dim)

summary(particles::GiantKelp) = string("Giant kelp (Macrocystis pyrifera) model with $(length(particles)) individuals of $(size(particles.positions.x, 2) - 1) nodes.")
show(io::IO, particles::GiantKelp) = print(io, string(summary(particles), " \n",
                                                      " Base positions:\n", 
                                                      " - x ∈ [$(minimum(particles.positions.x)), $(maximum(particles.positions.x))]\n",
                                                      " - y ∈ [$(minimum(particles.positions.y)), $(maximum(particles.positions.y))]\n",
                                                      " - z ∈ [$(minimum(particles.positions.z)), $(maximum(particles.positions.z))]"))

function update_tendencies!(bgc, particles::GiantKelp, model)
    Gᵘ, Gᵛ, Gʷ = @inbounds model.timestepper.Gⁿ[(:u, :v, :w)]

    tracer_tendencies = @inbounds model.timestepper.Gⁿ[keys(particles.tracer_forcing)]

    n_particles = size(particles, 1)
    worksize = n_particles
    workgroup = min(256, worksize)

    #####
    ##### Apply the tracer tendencies from each particle
    ####
    update_tendencies_kernel! = _update_tendencies!(device(model.architecture), workgroup, worksize)

    update_tendencies_kernel!(particles, Gᵘ, Gᵛ, Gʷ, tracer_tendencies, model.grid, model.tracers, values(particles.tracer_forcing)) 

    synchronize(device(architecture(model)))
end

@kernel function _update_tendencies!(particles, Gᵘ, Gᵛ, Gʷ, tracer_tendencies, grid, tracers, tracer_forcings)
    p = @index(Global)

    sf = particles.scalefactor[p]
    positions = particles.positions

    x⃗₀ = @inbounds (x = positions.x[p, 1], y = positions.y[p, 1], z = positions.z[p, 1])
    x⃗₁ = @inbounds (x = positions.x[p, 2], y = positions.y[p, 2], z = positions.z[p, 2])
    x⃗₂ = @inbounds (x = positions.x[p, 3], y = positions.y[p, 3], z = positions.z[p, 3])

    i₀, j₀, k₀ = get_closest_ijk(grid, x⃗₀)
    i₁, j₁, k₁ = get_closest_ijk(grid, x⃗₁)
    i₂, j₂, k₂ = get_closest_ijk(grid, x⃗₂)

    k1₁ = min(k₀, k₁)
    k2₁ = max(k₀, k₁)

    k1₂ = min(k₁, k₂)
    k2₂ = max(k₁, k₂)

    vol1 = total_volume(grid, i₁, j₁, Val(k1₁), Val(k2₁))
    vol2 = total_volume(grid, i₂, j₂, Val(k1₂), Val(k2₂))

    # first node
    for k in k1₁:k2₁
        scaling = sf / vol1 /  particles.kinematics.water_density

        @inbounds atomic_add!(Gᵘ, i₁, j₁, k, - particles.drag_forces.x[p, 2] * scaling)
        @inbounds atomic_add!(Gᵛ, i₁, j₁, k, - particles.drag_forces.y[p, 2] * scaling)
        @inbounds atomic_add!(Gʷ, i₁, j₁, k, - particles.drag_forces.z[p, 2] * scaling)

        for (tracer_idx, forcing) in enumerate(tracer_forcings)
            tracer_tendency = @inbounds tracer_tendencies[tracer_idx]

            total_scaling = sf / vol1 * volume(i₁, j₁, k, grid, Center(), Center(), Center())
            atomic_add!(tracer_tendency, i₁, j₁, k, total_scaling * forcing.func(i₁, j₁, k, p, 1, grid, clock, particles, tracers, forcing.parameters))
        end
    end

    # second node
    for k in k1₂:k2₂
        scaling = sf / vol2 /  particles.kinematics.water_density

        @inbounds atomic_add!(Gᵘ, i₂, j₂, k, - particles.drag_forces.x[p, 3] * scaling)
        @inbounds atomic_add!(Gᵛ, i₂, j₂, k, - particles.drag_forces.y[p, 3] * scaling)
        @inbounds atomic_add!(Gʷ, i₂, j₂, k, - particles.drag_forces.z[p, 3] * scaling)

        for (tracer_idx, forcing) in enumerate(tracer_forcings)
            tracer_tendency = @inbounds tracer_tendencies[tracer_idx]

            total_scaling = sf / vol2 * volume(i₂, j₂, k, grid, Center(), Center(), Center())
            atomic_add!(tracer_tendency, i₂, j₂, k, total_scaling * forcing.func(i₂, j₂, k, p, 2, grid, clock, particles, tracers, forcing.parameters))
        end
    end
end

@inline total_volume(grid, i, j, ::Val{k1}, ::Val{k2}) where {k1, k2} = sum(
    ntuple(k0 -> volume(i, j, k0 + k1 - 1, grid, Center(), Center(), Center()), 
           Val(k2 - k1 + 1))
)

include("utils.jl")

end # module GiantKelpDynamics
