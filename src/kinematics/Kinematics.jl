using LinearAlgebra

using OceanBioME.Particles: get_node, collapse_position

using Oceananigans.Fields: _fractional_indices, interpolator
using Oceananigans.Grids: AbstractGrid

function update_lagrangian_particle_properties!(particles::GiantKelp, model, bgc, Δt)
    # this will need to be modified when we have biological properties to update
    n_particles = size(particles, 1)
    worksize = (n_particles, )
    workgroup = (min(256, worksize[1]), )

    kinematics_kernel! = particles.kinematics(device(model.architecture), workgroup, worksize)
    step_kernel! = step_nodes!(device(model.architecture), workgroup, worksize)

    water_accelerations = @inbounds model.timestepper.Gⁿ[(:u, :v, :w)]

    step_t = zero(eltype(particles.positions.x))

    while step_t < Δt
        kinematics_kernel!(particles.positions, 
                            particles.velocities,
                            particles.stipe_radii,  
                            particles.blade_areas, particles.relaxed_lengths, 
                            particles.accelerations, particles.drag_forces, 
                            model.velocities, water_accelerations,
                            particles.kinematics, model.grid,
                            particles.max_Δt)

        synchronize(device(architecture(model)))

        stage_Δt = min(minimum(particles.max_Δt), Δt - step_t)

        step_kernel!(particles.accelerations, particles.old_accelerations, 
                        particles.velocities, particles.old_velocities,
                        particles.positions, 
                        particles.timestepper, stage_Δt)

        synchronize(device(architecture(model)))

        step_t += stage_Δt
    end

    particles.custom_dynamics(particles, model, bgc, Δt)
end

include("field_functions.jl")


@inline tension(Δx, l₀, Aᶜ, k, α) = ifelse(Δx > l₀ && !(Δx == 0.0), k * (max(0, (Δx - l₀)) / l₀) ^ α * Aᶜ, 0.0)

#include("utter_denny.jl")
include("utter_denny_speed.jl")