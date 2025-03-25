using LinearAlgebra

using OceanBioME.Particles: get_node, collapse_position

using Oceananigans.Fields: _fractional_indices, interpolator
using Oceananigans.Grids: AbstractGrid

function update_lagrangian_particle_properties!(particles::GiantKelp, model, bgc, Δt)
    # this will need to be modified when we have biological properties to update
    n_particles = size(particles, 1)
    n_nodes = @inbounds size(particles, 2)
    worksize = (n_particles, n_nodes)
    workgroup = (1, min(256, worksize[1]))

    kinematics_kernel! = particles.kinematics(device(model.architecture), workgroup, worksize)
    step_kernel! = step_nodes!(device(model.architecture), workgroup, worksize)

    n_substeps = max(1, 1 + floor(Int, Δt / (particles.max_Δt)))

    water_accelerations = @inbounds model.timestepper.Gⁿ[(:u, :v, :w)]

    for _ in 1:n_substeps, stage in stages(particles.timestepper)
        kinematics_kernel!(particles.holdfast_x, particles.holdfast_y, particles.holdfast_z, 
                           particles.positions, particles.positions_ijk,
                           particles.velocities,
                           particles.pneumatocyst_volumes, particles.stipe_radii,  
                           particles.blade_areas, particles.relaxed_lengths, 
                           particles.accelerations, particles.drag_forces, 
                           model.velocities, water_accelerations,
                           particles.kinematics, model.grid) # you cant do `(f::F)(args...)` and access the paramaters of f for kernels

        synchronize(device(architecture(model)))

        step_kernel!(particles.accelerations, particles.old_accelerations, 
                     particles.velocities, particles.old_velocities,
                     particles.positions, particles.holdfast_z,
                     particles.timestepper, Δt / n_substeps, stage)

        synchronize(device(architecture(model)))
    end

    particles.custom_dynamics(particles, model, bgc, Δt)
end

include("utter_denny.jl")
