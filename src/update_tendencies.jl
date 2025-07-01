# Speed
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

@kernel function _update_tendencies!(particles::GiantKelp{<:UtterDennySpeed}, Gᵘ, Gᵛ, Gʷ, tracer_tendencies, grid, tracers, tracer_forcings)
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

    vol1 = total_volume(grid, i₁, j₁, k1₁, k2₁)
    vol2 = total_volume(grid, i₂, j₂, k1₂, k2₂)

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

# generic
function update_tendencies!(bgc, particles::GiantKelp{<:UtterDenny}, model)
    Gᵘ, Gᵛ, Gʷ = @inbounds model.timestepper.Gⁿ[(:u, :v, :w)]

    tracer_tendencies = @inbounds model.timestepper.Gⁿ[keys(particles.tracer_forcing)]

    n_particles = size(particles, 1)
    n_nodes = size(particles, 2) - 1
    worksize = (n_particles, n_nodes)
    workgroup = (1, min(256, worksize[1]))

    #####
    ##### Apply the tracer tendencies from each particle
    ####
    update_tendencies_kernel! = _update_tendencies!(device(model.architecture), workgroup, worksize)

    update_tendencies_kernel!(particles, Gᵘ, Gᵛ, Gʷ, tracer_tendencies, model.grid, model.tracers, values(particles.tracer_forcing)) 

    synchronize(device(architecture(model)))
end

@kernel function _update_tendencies!(particles::GiantKelp{<:UtterDenny}, Gᵘ, Gᵛ, Gʷ, tracer_tendencies, grid, tracers, tracer_forcings)
    p, n = @index(Global, NTuple)

    n += 1

    sf = particles.scalefactor[p]
    positions = particles.positions

    x⃗ᵢ = @inbounds (x = positions.x[p, n], y = positions.y[p, n], z = positions.z[p, n])
    x⃗ᵢ₋₁ = @inbounds (x = positions.x[p, n-1], y = positions.y[p, n-1], z = positions.z[p, n-1])

    iᵢ, jᵢ, kᵢ = get_closest_ijk(grid, x⃗ᵢ)
    iᵢ₋₁, jᵢ₋₁, kᵢ₋₁ = get_closest_ijk(grid, x⃗ᵢ₋₁)

    k1 = min(kᵢ, kᵢ₋₁)
    k2 = max(kᵢ, kᵢ₋₁)

    vol = total_volume(grid, iᵢ, jᵢ, k1, k2)

    for k in k1:k2
        scaling = sf / vol /  particles.kinematics.water_density

        @inbounds atomic_add!(Gᵘ, iᵢ, jᵢ, k, - particles.drag_forces.x[p, n] * scaling)
        @inbounds atomic_add!(Gᵛ, iᵢ, jᵢ, k, - particles.drag_forces.y[p, n] * scaling)
        @inbounds atomic_add!(Gʷ, iᵢ, jᵢ, k, - particles.drag_forces.z[p, n] * scaling)

        for (tracer_idx, forcing) in enumerate(tracer_forcings)
            tracer_tendency = @inbounds tracer_tendencies[tracer_idx]

            total_scaling = sf / vol * volume(iᵢ, jᵢ, k, grid, Center(), Center(), Center())
            atomic_add!(tracer_tendency, iᵢ, jᵢ, k, total_scaling * forcing.func(iᵢ, jᵢ, k, p, 1, grid, clock, particles, tracers, forcing.parameters))
        end
    end
end
