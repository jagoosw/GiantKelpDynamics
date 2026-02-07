using GiantKelpGrowth, OceanBioME, Oceananigans

import Oceananigans.TimeSteppers: step_lagrangian_particles!, update_state!
import OceanBioME: update_tendencies!
import OceanBioME.Particles: compute_particle_tendencies!, update_particle_state!, step_particle_biogeochemistry!

const GrowingGiantKelp = Tuple{GiantKelp, BiogeochemicalParticles}

compute_particle_tendencies!(particles::GrowingGiantKelp, model) =
    compute_particle_tendencies!(@inbounds particles[2], model)

update_particle_state!(particles::GrowingGiantKelp, model, Δt) =
    update_particle_state!(@inbounds particles[2], model, Δt)

function update_tendencies!(bgc, particles::GrowingGiantKelp, model)
    dynamics, growth = particles

    update_blade_area!(dynamics, growth, model) # put the growth areas into the dynamics

    update_tendencies!(bgc, growth, model)
    update_tendencies!(bgc, dynamics, model)

    return nothing
end

# assumes only one growth model, also hopes and prayers that I've remembered to colocate them
function update_blade_area!(dynamics, growth::BiogeochemicalParticles{1, <:GiantKelpGrowth.GiantKelp{NC, NS}}, model) where {NC, NS}
    # sub canopy on the first node, canopy onto the second?
    ΣAc = zero(model.grid)
    ΣAs = zero(model.grid)

    for n in 1:NC
        ΣAc += sum(getproperty(growth.fields, Symbol(:A, n)))
    end

    for n in NC+1:NC+NS
        ΣAs += sum(getproperty(growth.fields, Symbol(:A, n)))
    end

    CUDA.@allowscalar begin # TODO: think of a better way todo this
        dynamics.blade_areas[:, 1] .= ΣAc / 10000
        dynamics.blade_areas[:, 2] .= ΣAs / 10000
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