"""
    UtterDenny(; spring_constant = 1.91 * 10 ^ 7,
                 spring_exponent = 1.41,
                 water_density = 1026.0,
                 pneumatocyst_specific_buoyancy = 5.,
                 gravitational_acceleration = 9.81,
                 stipe_drag_coefficient = 1.,
                 blade_drag_coefficient = 0.4 * 12 ^ -0.485,
                 added_mass_coefficient = 3.,
                 damping_timescale = 5.)

Sets up the kinematic model for giant kelp motion from [Utter1996](@citet) and [Rosman2013](@citet).
"""

@kwdef struct UtterDenny{FT}
                 spring_constant :: FT = 1.91 * 10 ^ 7
                 spring_exponent :: FT = 1.41
                   water_density :: FT = 1026.0
  pneumatocyst_specific_buoyancy :: FT = 5.
      gravitational_acceleration :: FT = 9.81
          stipe_drag_coefficient :: FT = 1.
          blade_drag_coefficient :: FT = 0.87
          added_mass_coefficient :: FT = 3.
               damping_timescale :: FT = 5.
end


function update_lagrangian_particle_properties!(particles::GiantKelp{<:UtterDenny}, model, bgc, Δt)
    # this will need to be modified when we have biological properties to update
    n_particles = size(particles, 1)
    n_nodes = size(particles, 2) - 1
    worksize = (n_particles, n_nodes)
    workgroup = (1, min(256, worksize[1]))

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

        stage_Δt = min(minimum(particles.max_Δt), Δt - step_t)# max(1e-3 * exp(-model.clock.time/6000), min(minimum(particles.max_Δt), Δt - step_t))

        if stage_Δt > Δt / 1e10
            step_kernel!(particles.accelerations, particles.old_accelerations, 
                        particles.velocities, particles.old_velocities,
                        particles.positions, 
                        particles.timestepper, stage_Δt)

            synchronize(device(architecture(model)))
        end

        step_t += stage_Δt
    end

    particles.custom_dynamics(particles, model, bgc, Δt)
end

@kernel function (kinematics::UtterDenny)(
            positions, 
            velocities, 
            stipe_radii, 
            blade_areas, relaxed_lengths, 
            accelerations, drag_forces, 
            water_velocities, water_accelerations,
            kinematics, grid::AbstractGrid{FT, TX, TY, TZ},
            max_Δt) where {FT, TX, TY, TZ}

    p, n = @index(Global, NTuple)

    n += 1

    n_nodes = size(positions.x, 2)

    zero_vector!(p, n, accelerations)

    spring_constant = kinematics.spring_constant
    spring_exponent = kinematics.spring_exponent
    ρₒ  = kinematics.water_density
    Cᵈˢ = kinematics.stipe_drag_coefficient
    Cᵈᵇ = kinematics.blade_drag_coefficient
    Cᵃ  = kinematics.added_mass_coefficient
    τ   = kinematics.damping_timescale
    Fᵇ  = kinematics.pneumatocyst_specific_buoyancy

    rˢ = stipe_radii

    x⃗ᵢ = @inbounds (x = positions.x[p, n], y = positions.y[p, n], z = positions.z[p, n])
    x⃗ᵢ₋₁ = @inbounds (x = positions.x[p, n-1], y = positions.y[p, n-1], z = positions.z[p, n-1])

    u⃗ᵢ = @inbounds (x = velocities.x[p, n], y = velocities.y[p, n], z = velocities.z[p, n])

    Δxᵢ = (x = x⃗ᵢ.x - x⃗ᵢ₋₁.x, y = x⃗ᵢ.y - x⃗ᵢ₋₁.y, z = x⃗ᵢ.z - x⃗ᵢ₋₁.z)
    lᵢ = mag(Δxᵢ)
    
    # drag
    Aᵇ = @inbounds blade_areas[p, n-1]
    Vᵐ = π * rˢ ^ 2 * lᵢ + Aᵇ * 0.01
    l⁰ = @inbounds relaxed_lengths[p, n-1]

    mᵉ = 0.774*0.297*l⁰^0.995#Vᵐ₁ * (1 + Cᵃ) * 20#50

    iᵢ, jᵢ, kᵢ = get_closest_ijk(grid, x⃗ᵢ)
    iᵢ₋₁, jᵢ₋₁, kᵢ₋₁ = get_closest_ijk(grid, x⃗ᵢ₋₁)

    k1 = min(kᵢ, kᵢ₋₁)
    k2 = max(kᵢ, kᵢ₋₁)

    Uʷ = mean_water_velocity(iᵢ, jᵢ, k1, k2, water_velocities)

    Uʳ = @inbounds (x = Uʷ.x - u⃗ᵢ.x, y = Uʷ.y - u⃗ᵢ.y, z = Uʷ.z - u⃗ᵢ.z)

    sʳ = sqrt(Uʳ.x^2 + Uʳ.y^2 + Uʳ.z^2)

    Aʷ = mean_water_velocity(iᵢ, jᵢ, k1, k2, water_accelerations)
 
    θ = @inbounds acos(min(1, abs(Uʳ.x * Δxᵢ.x + Uʳ.y * Δxᵢ.y + Uʳ.z * Δxᵢ.z) / (sʳ * lᵢ + eps(0.0))))

    Aˢ = 2 * rˢ * lᵢ * abs(sin(θ)) + π * rˢ * abs(cos(θ))

    Fᴰ = ρₒ/2 * (Cᵈˢ * Aˢ + 0.0148 * Aᵇ) * sʳ^1.596 #(Cᵈˢ * Aˢ₁ + Cᵈᵇ * Aᵇ₁) * sʳ₁^2

    add_components!(p, n, accelerations, Fᴰ, (x = Uʳ.x / (sʳ+eps(0.0)), y = Uʳ.y / (sʳ+eps(0.0)), z = Uʳ.z / (sʳ+eps(0.0))))

    # tension
    x⃗ᵢ₊₁ = @inbounds (x = positions.x[p, min(n+1, n_nodes)], y = positions.y[p, min(n+1, n_nodes)], z = positions.z[p, min(n+1, n_nodes)])
    Δxᵢ₊₁ = (x = x⃗ᵢ₊₁.x - x⃗ᵢ.x, y = x⃗ᵢ₊₁.y - x⃗ᵢ.y, z = x⃗ᵢ₊₁.z - x⃗ᵢ.z)
    lᵢ₊₁ = mag(Δxᵢ₊₁)

    l⁰₊ = @inbounds relaxed_lengths[p, min(n, n_nodes-1)]

    Aᶜ = π * rˢ ^ 2

    T₋ = tension(lᵢ, l⁰, Aᶜ, spring_constant, spring_exponent)
    T₊ = tension(lᵢ₊₁, l⁰₊, Aᶜ, spring_constant, spring_exponent)

    #=if n == 3
        @info T₋, T₊, n_nodes, n+1, x⃗ᵢ₋₁, x⃗ᵢ, x⃗ᵢ₊₁, lᵢ, lᵢ₊₁
    end=#

    add_components!(p, n, accelerations, T₋, (x = - Δxᵢ.x / (lᵢ+eps(0.0)), y = - Δxᵢ.y / (lᵢ+eps(0.0)), z = - Δxᵢ.z / (lᵢ+eps(0.0))))
    add_components!(p, n, accelerations, T₊, (x = Δxᵢ₊₁.x / (lᵢ₊₁+eps(0.0)), y = Δxᵢ₊₁.y / (lᵢ₊₁+eps(0.0)), z = Δxᵢ₊₁.z / (lᵢ₊₁+eps(0.0))))

    # inertial force
    Fⁱ = ρₒ * Vᵐ * mag(Aʷ)

    add_components!(p, n, accelerations, Fⁱ, (x = Aʷ.x / (mag(Aʷ)+eps(0.0)), y = Aʷ.y / (mag(Aʷ)+eps(0.0)), z = Aʷ.z / (mag(Aʷ)+eps(0.0))))
    
    # buoyancy

    add_components!(p, n, accelerations, ifelse(x⃗ᵢ.z < 0, Fᵇ * l⁰ / sum(relaxed_lengths[p, :]), 0), (x = 0, y = 0, z = 1))

    # finishing up

    multiply_components!(p, n, accelerations, 1/mᵉ)
    add_components!(p, n, accelerations, - 1 / τ, u⃗ᵢ)

    set_components!(p, n, drag_forces, Fᴰ, (x = Uʳ.x / (sʳ+eps(0.0)), y = Uʳ.y / (sʳ+eps(0.0)), z = Uʳ.z / (sʳ+eps(0.0))))

    # max timestep
    α = spring_exponent

    τₜ₁ = ifelse(T₊ == 0, Inf, sqrt(mᵉ / T₊ / (max(lᵢ, l⁰) * α - l⁰) * lᵢ * l⁰))
    τₜ₂ = ifelse((T₋ == 0) || (lᵢ₊₁ == 0), Inf, sqrt(mᵉ / T₋ / (max(lᵢ₊₁, l⁰₊) * α - l⁰₊) * lᵢ₊₁ * l⁰₊))
    τₐ  = mᵉ / abs(Fᴰ) / (sʳ + eps(0.0))
    τᵇ  = sqrt(0.1 / (abs(Fᵇ) / mᵉ))
    
   #=if n == 2
    @info τₜ₁, τₜ₂, τₐ, τᵇ
    end=#
    @inbounds max_Δt[p, n] = 0.5 * min(τₜ₁, τₜ₂, τₐ, τᵇ)
end
