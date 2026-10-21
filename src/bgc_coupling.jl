using GiantKelpGrowth, OceanBioME, Oceananigans, KernelAbstractions

using Oceananigans.Architectures: architecture
using Oceananigans.Grids: znodes
using Oceananigans.Utils: launch!

import Oceananigans.TimeSteppers: step_lagrangian_particles!, update_state!
import OceanBioME: update_tendencies!
import OceanBioME.Particles: compute_particle_tendencies!, update_particle_state!, step_particle_biogeochemistry!
import GiantKelpGrowth: canopy_volume, subcanopy_volume, canopy_indices, subcanopy_indices

const GrowingGiantKelp = Tuple{GiantKelp, BiogeochemicalParticles}

compute_particle_tendencies!(particles::GrowingGiantKelp, model) =
    compute_particle_tendencies!(@inbounds particles[2], model)

update_particle_state!(particles::GrowingGiantKelp, model, Δt) =
    update_particle_state!(@inbounds particles[2], model, Δt)

function update_tendencies!(bgc, particles::GrowingGiantKelp, model)
    dynamics, growth = particles

    update_coupling!(dynamics, growth, model) # put the growth areas into the dynamics

    update_tendencies!(bgc, growth, model)
    update_tendencies!(bgc, dynamics, model)

    return nothing
end

# assumes only one growth model, also hopes and prayers that I've remembered to colocate them
function update_coupling!(dynamics, growth::BiogeochemicalParticles{<:Any, <:GiantKelpGrowth.GiantKelp{NC, NS}}, model) where {NC, NS}
    CUDA.@allowscalar begin # TODO: think of a better way todo this
        ΣAc = sum(growth.fields.A[1:NC]) * growth.scalefactors[]
        ΣAs = sum(growth.fields.A[NC+1:NC+NS]) * growth.scalefactors[]

        dynamics.blade_areas[:, 1] .= ΣAc / 10000
        dynamics.blade_areas[:, 2] .= ΣAs / 10000
    end

    canopy_indices = growth.biogeochemistry.tracer_values.canopy_indices
    subcanopy_indices = growth.biogeochemistry.tracer_values.subcanopy_indices

    grid = model.grid
    arch = architecture(grid)
    launch!(architecture(grid), grid, :xyz, update_kelp_location!, canopy_indices.indices, dynamics.drag.w, grid, Val(:surface), )
    launch!(architecture(grid), grid, :xyz, update_kelp_location!, subcanopy_indices.indices, dynamics.drag.w, grid, Val(:subsurface))

#    canopy_depth = 1.0 # todo: put this back somewhere

# this solution is not working for some reason
#    canopy_indices.indices .*= on_architecture(arch, reshape(Bool.(map(z->z >= -canopy_depth, znodes(canopy_indices.indices))), 1, 1, grid.Nz))
#    subcanopy_indices.indices .*= on_architecture(arch, reshape(Bool.(map(z->z < -canopy_depth, znodes(subcanopy_indices.indices))), 1, 1, grid.Nz))

    CUDA.@allowscalar begin
        canopy_indices.volume[] = interior(Field(Integral(canopy_indices.indices)), 1, 1, 1)[]
        subcanopy_indices.volume[] = interior(Field(Integral(subcanopy_indices.indices)), 1, 1, 1)[]
    end

    if canopy_indices.volume[] == 0
        canopy_indices.indices .= true
        canopy_indices.volume[] = Inf
    end

    if subcanopy_indices.volume[] == 0
        subcanopy_indices.indices .= true
        subcanopy_indices.volume[] = Inf
    end

    return nothing
end

using Oceananigans.Grids: znode

@kernel function update_kelp_location!(kelp_location, drag, grid, ::Val{:surface}, canopy_depth = 1)
    i, j, k = @index(Global, NTuple)

    z = znode(i, j, k, grid, Center(), Center(), Center())

    @inbounds begin
        kelp_location[i, j, k] = (abs(drag[i, j, k]) > 0) & (z >= - canopy_depth)
    end
end

@kernel function update_kelp_location!(kelp_location, drag, grid, ::Val{:subsurface}, canopy_depth = 1)
    i, j, k = @index(Global, NTuple)

    z = znode(i, j, k, grid, Center(), Center(), Center())

    @inbounds begin
        kelp_location[i, j, k] = (abs(drag[i, j, k]) > 0) & (z < - canopy_depth)
    end
end

@inline step_particle_biogeochemistry!(timestepper, particles::GrowingGiantKelp, model, Δt) =
    @inbounds step_particle_biogeochemistry!(timestepper, particles[2], model, Δt)

const NonhydrostaticWithCoupledBiogeochemicalParticles = NonhydrostaticModel{
    <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
    <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
    <:Any, <:Any, <:Any, <:Any, <:OceanBioME.DiscreteBiogeochemistry{<:Any, <:Any, <:Any, <:Tuple}
}

@inline step_lagrangian_particles!(::Nothing, model::NonhydrostaticWithCoupledBiogeochemicalParticles, Δt) =
    update_lagrangian_particle_properties!(model.biogeochemistry.particles[2], model, model.biogeochemistry, Δt)

@inline canopy_volume(kelp::GrowingGiantKelp) = @inbounds canopy_volume(kelp[2])
@inline subcanopy_volume(kelp::GrowingGiantKelp) = @inbounds subcanopy_volume(kelp[2])

@inline canopy_indices(kelp::GrowingGiantKelp) = @inbounds canopy_indices(kelp[2])
@inline subcanopy_indices(kelp::GrowingGiantKelp) = @inbounds subcanopy_indices(kelp[2])