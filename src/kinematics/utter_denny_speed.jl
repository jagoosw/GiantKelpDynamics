using Oceananigans.Units: minutes, hour

"""
    UtterDennySpeed(; spring_constant = 1.91 * 10 ^ 7,
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
@kwdef struct UtterDennySpeed{FT}
                 spring_constant :: FT = 1.91 * 10 ^ 7
                 spring_exponent :: FT = 1.41
                   water_density :: FT = 1026.0
  pneumatocyst_specific_buoyancy :: FT = 2.5 # https://doi.org/10.1242/jeb.199.12.2645
      gravitational_acceleration :: FT = 9.81
          stipe_drag_coefficient :: FT = 1.
          blade_drag_coefficient :: FT = 0.0148#0.87# change from origional publication due to error in how tendancy scaling was calculated, but this results in the same drag (i.e. ηCd is the samd where η is the scaling factor)
          added_mass_coefficient :: FT = 3.
               damping_timescale :: FT = 5.
               turn_on_timescale :: FT = 1hour
end

function update_lagrangian_particle_properties!(particles::GiantKelp{<:UtterDennySpeed}, model, bgc, Δt)
    # this will need to be modified when we have biological properties to update
#=
    # TODO: move this into the default logic
    step_t = zero(eltype(particles.positions.x))

    while step_t < Δt
        stage_Δt = time_step_kelp!(particles.timestepper, particles, model, bgc, Δt, step_t)

        step_t += stage_Δt
    end

    particles.custom_dynamics(particles, model, bgc, Δt)
=#
    return nothing
end

# default for Euler
function time_step_kelp!(timestepper, particles, model, bgc, Δt, step_t)
    n_particles = size(particles, 1)
    worksize = (n_particles, )
    workgroup = (min(256, worksize[1]), )

    kinematics_kernel! = particles.kinematics(device(model.architecture), workgroup, worksize)
    step_kernel! = step_nodes!(device(model.architecture), workgroup, worksize)

    water_accelerations = @inbounds model.timestepper.Gⁿ[(:u, :v, :w)]

    kinematics_kernel!(particles.positions, 
                       particles.velocities,
                       particles.stipe_radii,  
                       particles.blade_areas, particles.relaxed_lengths, 
                       particles.accelerations, particles.drag_forces, 
                       model.velocities, water_accelerations,
                       particles.kinematics, model.grid,
                       particles.max_Δt,
                       model.clock.time)

    stage_Δt = min(minimum(particles.max_Δt)/2,#0.1,#one(Δt)/10, #
                  Δt - step_t)

    if stage_Δt > Δt / 1e10
        step_kernel!(particles.accelerations, particles.old_accelerations, 
                     particles.velocities, particles.old_velocities,
                     particles.positions, 
                     particles.timestepper, stage_Δt, Val(3))

        synchronize(device(architecture(model)))
    end

    return stage_Δt
end

# Newmark-β
function time_step_kelp!(timestepper::Newmarkβ, particles, model, bgc, Δt, step_t)
    # estimaiton of reasonable step size
    k = particles.kinematics.spring_constant
    α = particles.kinematics.spring_exponent
    m = 5 # could be bigger could be smaller, todo: should make a function for this
    Ac = π * 0.004^2
    l₀ = minimum(particles.relaxed_lengths)
    Δt_stable = 1 / sqrt(k * Ac * α / (l₀^α * m))
    step_Δt = min(Δt, Δt_stable)

    # setup
    n_particles = size(particles, 1)
    worksize = (n_particles, )
    workgroup = (min(256, worksize[1]), )

    kinematics_kernel! = particles.kinematics(device(model.architecture), workgroup, worksize)
    predictor_kernel! = predictor_step!(device(model.architecture), workgroup, worksize)
    corrector_kernel! = corrector_step!(device(model.architecture), workgroup, worksize)

    water_accelerations = @inbounds model.timestepper.Gⁿ[(:u, :v, :w)]

    # Aₙ
    kinematics_kernel!(particles.positions, 
                       particles.velocities,
                       particles.stipe_radii,  
                       particles.blade_areas, particles.relaxed_lengths, 
                       particles.accelerations, particles.drag_forces, 
                       model.velocities, water_accelerations,
                       particles.kinematics, model.grid,
                       particles.max_Δt,
                       model.clock.time)

    # positions = Xn, velocities = Vn, accelerations = An

    predictor_kernel!(timestepper, 
                      step_Δt, 
                      particles.positions, 
                      particles.velocities, 
                      particles.accelerations,
                      particles.old_velocities, 
                      particles.old_accelerations,
                      Val(3))

    # we have positions = X* = X0, velocities = V* = V0, old_accelerations = An = A0, old_velocities = An

    for _ in 1:timestepper.iterations
        kinematics_kernel!(particles.positions, 
                           particles.velocities,
                           particles.stipe_radii,  
                           particles.blade_areas, particles.relaxed_lengths, 
                           particles.accelerations, particles.drag_forces, 
                           model.velocities, water_accelerations,
                           particles.kinematics, model.grid,
                           particles.max_Δt,
                           model.clock.time)

        # now acceleration is Anew

        corrector_kernel!(timestepper, 
                          step_Δt,
                          particles.positions, 
                          particles.velocities, 
                          particles.accelerations,
                          particles.old_velocities, 
                          particles.old_accelerations,
                          Val(3))
    end


kinematics_kernel!(particles.positions,
                           particles.velocities,
                           particles.stipe_radii,
                           particles.blade_areas, particles.relaxed_lengths,
                           particles.accelerations, particles.drag_forces,
                           model.velocities, water_accelerations,
                           particles.kinematics, model.grid,
                           particles.max_Δt,
                           model.clock.time)
    return step_Δt
end

@kernel function (kinematics::UtterDennySpeed)(
                  positions, 
                  velocities, 
                  stipe_radii, 
                  blade_areas, relaxed_lengths, 
                  accelerations, drag_forces, 
                  water_velocities, water_accelerations,
                  kinematics, grid::AbstractGrid{FT, TX, TY, TZ},
                  max_Δt,
                  t) where {FT, TX, TY, TZ}

    p = @index(Global)

    zero_vector!(p, 2, accelerations)
    zero_vector!(p, 3, accelerations)

    spring_constant = kinematics.spring_constant
    spring_exponent = kinematics.spring_exponent
    ρₒ  = kinematics.water_density
    Cᵈˢ = kinematics.stipe_drag_coefficient
    Cᵈᵇ = kinematics.blade_drag_coefficient
    Cᵃ  = kinematics.added_mass_coefficient
    τ   = kinematics.damping_timescale
    Fᵇ  = kinematics.pneumatocyst_specific_buoyancy

    τ_spinup = kinematics.turn_on_timescale

    rˢ = stipe_radii

    x⃗₀ = @inbounds (x = positions.x[p, 1], y = positions.y[p, 1], z = positions.z[p, 1])
    x⃗₁ = @inbounds (x = positions.x[p, 2], y = positions.y[p, 2], z = positions.z[p, 2])
    x⃗₂ = @inbounds (x = positions.x[p, 3], y = positions.y[p, 3], z = positions.z[p, 3])

    u⃗₁ = @inbounds (x = velocities.x[p, 2], y = velocities.y[p, 2], z = velocities.z[p, 2])
    u⃗₂ = @inbounds (x = velocities.x[p, 3], y = velocities.y[p, 3], z = velocities.z[p, 3])

    Δx₁ = (x = x⃗₁.x - x⃗₀.x, y = x⃗₁.y - x⃗₀.y, z = x⃗₁.z - x⃗₀.z)
    l₁ = mag(Δx₁)

    Δx₂ = (x = x⃗₂.x - x⃗₁.x, y = x⃗₂.y - x⃗₁.y, z = x⃗₂.z - x⃗₁.z)
    l₂ = mag(Δx₂)

    # drag
    Aᵇ₁ = @inbounds blade_areas[p, 1]
    Aᵇ₂ = @inbounds blade_areas[p, 2]
    Vᵐ₁ = π * rˢ ^ 2 * l₁ + Aᵇ₁ * 0.01
    Vᵐ₂ = π * rˢ ^ 2 * l₂ + Aᵇ₂ * 0.01

    l⁰₁ = @inbounds relaxed_lengths[p, 1]
    l⁰₂ = @inbounds relaxed_lengths[p, 2]

    #https://doi.org/10.1242/jeb.199.12.2645
    mᵉ₁ = 0.774*0.297*l⁰₁^0.995 * (1 + Cᵃ)#Vᵐ₁ * (1 + Cᵃ) * 20#50#0.774*0.297*l⁰₁^0.995#
    mᵉ₂ = 0.774*0.297*l⁰₂^0.995 * (1 + Cᵃ)#Vᵐ₂ * (1 + Cᵃ) * 20#50#0.774*0.297*l⁰₂^0.995#

    # we need ijk and this also reduces repetition of finding ijk
    i₀, j₀, k₀ = get_closest_ijk(grid, x⃗₀)
    i₁, j₁, k₁ = get_closest_ijk(grid, x⃗₁)
    i₂, j₂, k₂ = get_closest_ijk(grid, x⃗₂)
#@info i₀, j₀, k₀, i₁, j₁, k₁, i₂, j₂, k₂, t
    k1₁ = min(k₀, k₁)
    k2₁ = max(k₀, k₁)

    k1₂ = min(k₁, k₂)
    k2₂ = max(k₁, k₂)

    Uʷ₁ = mean_water_velocity(i₁, j₁, k1₁, k2₁, water_velocities)
    Uʷ₂ = mean_water_velocity(i₂, j₂, k1₂, k2₂, water_velocities)

    Uʳ₁ = @inbounds (x = Uʷ₁.x - u⃗₁.x, y = Uʷ₁.y - u⃗₁.y, z = Uʷ₁.z - u⃗₁.z)
    Uʳ₂ = @inbounds (x = Uʷ₂.x - u⃗₂.x, y = Uʷ₂.y - u⃗₂.y, z = Uʷ₂.z - u⃗₂.z)
    Uʳ₁ = @inbounds (x = ifelse(Uʷ₁.x == u⃗₁.x, zero(Uʷ₁.x), Uʳ₁.x),
                     y = ifelse(Uʷ₁.y == u⃗₁.y, zero(Uʷ₁.y), Uʳ₁.y),
                     z = ifelse(Uʷ₁.z == u⃗₁.z, zero(Uʷ₁.z), Uʳ₁.z))

    Uʳ₂ = @inbounds (x = ifelse(Uʷ₂.x == u⃗₂.x, zero(Uʷ₂.x), Uʳ₂.x),
                     y = ifelse(Uʷ₂.y == u⃗₂.y, zero(Uʷ₂.y), Uʳ₂.y),
                     z = ifelse(Uʷ₂.z == u⃗₂.z, zero(Uʷ₂.z), Uʳ₂.z))
  #Uʳ₁ = Uʷ₁
  #Uʳ₂ = Uʷ₂
    sʳ₁ = sqrt(Uʳ₁.x^2 + Uʳ₁.y^2 + Uʳ₁.z^2)
    sʳ₂ = sqrt(Uʳ₂.x^2 + Uʳ₂.y^2 + Uʳ₂.z^2)

    Aʷ₁ = mean_water_velocity(i₁, j₁, k1₁, k2₁, water_accelerations)
    Aʷ₂ = mean_water_velocity(i₂, j₂, k1₂, k2₂, water_accelerations)

    θ₁ = @inbounds acos(min(1, abs(Uʳ₁.x * Δx₁.x + Uʳ₁.y * Δx₁.y + Uʳ₁.z * Δx₁.z) / (sʳ₁ * l₁ + eps(0.0))))
    θ₂ = @inbounds acos(min(1, abs(Uʳ₂.x * Δx₂.x + Uʳ₂.y * Δx₂.y + Uʳ₂.z * Δx₂.z) / (sʳ₂ * l₂ + eps(0.0))))

    Aˢ₁ = 2 * rˢ * l₁ * abs(sin(θ₁)) + π * rˢ * abs(cos(θ₁))
    Aˢ₂ = 2 * rˢ * l₂ * abs(sin(θ₂)) + π * rˢ * abs(cos(θ₂))

    Fᴰ₁ = ρₒ/2 * (Cᵈˢ * Aˢ₁ + Cᵈᵇ * Aᵇ₁) * sʳ₁^1.596 * (1 - exp(-t/τ_spinup)) #(Cᵈˢ * Aˢ₁ + 0.0148 * Aᵇ₁) * sʳ₁^1.596 #
    Fᴰ₂ = ρₒ/2 * (Cᵈˢ * Aˢ₂ + Cᵈᵇ * Aᵇ₂) * sʳ₂^1.596 * (1 - exp(-t/τ_spinup)) #(Cᵈˢ * Aˢ₂ + 0.0148 * Aᵇ₂) * sʳ₂^1.596 #


    Fᴰ₁′ = (x = ifelse(Uʷ₁.x == u⃗₁.x, zero(Uʷ₁.x), Uʳ₁.x / sʳ₁), 
            y = ifelse(Uʷ₁.y == u⃗₁.y, zero(Uʷ₁.y), Uʳ₁.y / sʳ₁),
            z = ifelse(Uʷ₁.z == u⃗₁.z, zero(Uʷ₁.z), Uʳ₁.z / sʳ₁))

    Fᴰ₂′ = (x = ifelse(Uʷ₂.x == u⃗₂.x, zero(Uʷ₂.x), Uʳ₂.x / sʳ₂), 
            y = ifelse(Uʷ₂.y == u⃗₂.y, zero(Uʷ₂.y), Uʳ₂.y / sʳ₂),
            z = ifelse(Uʷ₂.z == u⃗₂.z, zero(Uʷ₂.z), Uʳ₂.z / sʳ₂))

    add_components!(p, 2, accelerations, Fᴰ₁, Fᴰ₁′)
    add_components!(p, 3, accelerations, Fᴰ₂, Fᴰ₂′)

    # Tension
    Aᶜ = π * rˢ ^ 2

    T₀₁ = tension(l₁, l⁰₁, Aᶜ, spring_constant, spring_exponent)
    T₁₂ = tension(l₂, l⁰₂, Aᶜ, spring_constant, spring_exponent)

    add_components!(p, 2, accelerations, T₀₁, (x = - Δx₁.x / (l₁+eps(0.0)), y = - Δx₁.y / (l₁+eps(0.0)), z = - Δx₁.z / (l₁+eps(0.0))))
    add_components!(p, 2, accelerations, T₁₂, (x = Δx₂.x / (l₂+eps(0.0)), y = Δx₂.y / (l₂+eps(0.0)), z = Δx₂.z / (l₂+eps(0.0))))
    add_components!(p, 3, accelerations, T₁₂, (x = - Δx₂.x / (l₂+eps(0.0)), y = - Δx₂.y / (l₂+eps(0.0)), z = - Δx₂.z / (l₂+eps(0.0))))

    # inertial force
    Fⁱ₁ = ρₒ * Vᵐ₁ * mag(Aʷ₁)
    Fⁱ₂ = ρₒ * Vᵐ₂ * mag(Aʷ₂)

    add_components!(p, 2, accelerations, Fⁱ₁, (x = Aʷ₁.x / (mag(Aʷ₁)+eps(0.0)), y = Aʷ₁.y / (mag(Aʷ₁)+eps(0.0)), z = Aʷ₁.z / (mag(Aʷ₁)+eps(0.0))))
    add_components!(p, 3, accelerations, Fⁱ₂, (x = Aʷ₂.x / (mag(Aʷ₂)+eps(0.0)), y = Aʷ₂.y / (mag(Aʷ₂)+eps(0.0)), z = Aʷ₂.z / (mag(Aʷ₂)+eps(0.0))))
    
    add_components!(p, 2, accelerations, ifelse(x⃗₁.z <= 0, Fᵇ * l⁰₁ / (l⁰₁ + l⁰₂ + eps(0.0)), 0), (x = 0, y = 0, z = 1))
    add_components!(p, 3, accelerations, ifelse(x⃗₂.z <= 0, Fᵇ * l⁰₂ / (l⁰₁ + l⁰₂ + eps(0.0)), 0), (x = 0, y = 0, z = 1))

    multiply_components!(p, 2, accelerations, 1/mᵉ₁)
    multiply_components!(p, 3, accelerations, 1/mᵉ₂)
    
    add_components!(p, 2, accelerations, - 1 / τ, u⃗₁)
    add_components!(p, 3, accelerations, - 1 / τ, u⃗₂)

    set_components!(p, 2, drag_forces, Fᴰ₁, Fᴰ₁′)
    set_components!(p, 3, drag_forces, Fᴰ₂, Fᴰ₂′)

    α = spring_exponent

    τₜ₁ = ifelse(T₀₁ == 0, Inf, sqrt(mᵉ₁ / T₀₁ / (max(l₁, l⁰₁) * α - l⁰₁) * l₁ * l⁰₁))
    τₜ₂ = ifelse(T₁₂ == 0, Inf, sqrt(mᵉ₂ / T₁₂ / (max(l₂, l⁰₂) * α - l⁰₂) * l₂ * l⁰₂))
    τₐ₁ = mᵉ₁ / abs(Fᴰ₁ / (sʳ₁ + eps(0.0)))
    τₐ₂ = mᵉ₂ / abs(Fᴰ₂ / (sʳ₂ + eps(0.0)))
    τᵇ  = sqrt(0.1 / (abs(Fᵇ) / (mᵉ₁ + mᵉ₂)))

    @inbounds max_Δt[p] = min(0.5 * min(τₜ₁, τₜ₂, τₐ₁,  τₐ₂, τᵇ), 1.1 * max_Δt[p]) # limit to 10% growth
end
