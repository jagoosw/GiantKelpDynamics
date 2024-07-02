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

@kernel function (kinematics::UtterDenny)(x_holdfast, y_holdfast, z_holdfast, 
                                          positions, positions_ijk,
                                          velocities, 
                                          pneumatocyst_volumes, stipe_radii, 
                                          blade_areas, relaxed_lengths, 
                                          accelerations, drag_forces, 
                                          water_velocities, water_accelerations,
                                          kinematics, grid::AbstractGrid{FT, TX, TY, TZ}) where {FT, TX, TY, TZ}
    p, n = @index(Global, NTuple)

    spring_constant = kinematics.spring_constant
    spring_exponent = kinematics.spring_exponent
    ρₒ = kinematics.water_density
    ρₚ = kinematics.pneumatocyst_specific_buoyancy
    g = kinematics.gravitational_acceleration
    Cᵈˢ = kinematics.stipe_drag_coefficient
    Cᵈᵇ = kinematics.blade_drag_coefficient
    Cᵃ = kinematics.added_mass_coefficient
    τ = kinematics.damping_timescale

    xⁱ = positions[p, n, 1]
    yⁱ = positions[p, n, 2]
    zⁱ = positions[p, n, 3]

    uⁱ = velocities[p, n, 1]
    vⁱ = velocities[p, n, 2]
    wⁱ = velocities[p, n, 3]

    # can we eliminate this branching logic
    if n == 1
        x⁻, y⁻, z⁻ = 0, 0, 0
        u⁻, v⁻, w⁻ = 0, 0, 0
    else
        x⁻ = positions[p, n-1, 1]
        y⁻ = positions[p, n-1, 2]
        z⁻ = positions[p, n-1, 3]

        u⁻ = velocities[p, n-1, 1]
        v⁻ = velocities[p, n-1, 2]
        w⁻ = velocities[p, n-1, 3]
    end

    Δx = xⁱ - x⁻
    Δy = yⁱ - y⁻
    Δz = zⁱ - z⁻

    X = x_holdfast[p] + xⁱ
    Y = y_holdfast[p] + yⁱ
    Z = z_holdfast[p] + zⁱ

    l = sqrt(Δx^2 + Δy^2 + Δz^2)

    # buoyancy
    Vᵖ = pneumatocyst_volumes[p, n]

    Fᴮ = ρₚ * Vᵖ * g

    if Z >= 0
        Fᴮ = 0
    end

    # drag
    Aᵇ = blade_areas[p, n]
    rˢ = stipe_radii[p, n]
    Vᵐ = π * rˢ ^ 2 * l + Aᵇ * 0.01
    mᵉ = (Vᵐ + Cᵃ * (Vᵐ + Vᵖ)) * ρₒ + Vᵖ * (ρₒ - 500) 

    # we need ijk and this also reduces repetition of finding ijk
    ii, jj, kk = fractional_indices((X, Y, Z), grid, Center(), Center(), Center())

    ix = interpolator(ii)
    iy = interpolator(jj)
    iz = interpolator(kk)

    i, j, k = (get_node(TX(), Int(ifelse(ix[3] < 0.5, ix[1], ix[2])), grid.Nx),
               get_node(TY(), Int(ifelse(iy[3] < 0.5, iy[1], iy[2])), grid.Ny),
               get_node(TZ(), Int(ifelse(iz[3] < 0.5, iz[1], iz[2])), grid.Nz))

    positions_ijk[p, n, 1] = i
    positions_ijk[p, n, 2] = j
    positions_ijk[p, n, 3] = k

    _, _, kk⁻ = fractional_indices((x⁻ + x_holdfast[p], y⁻ + y_holdfast[p], z⁻ + y_holdfast[p]), grid, Center(), Center(), Center())

    iz⁻ = interpolator(kk⁻)

    k⁻ = get_node(TZ(), Int(ifelse(iz[3] < 0.5, iz⁻[1], iz⁻[2])), grid.Nz)

    k1 = min(k⁻, k)
    k2 = max(k⁻, k)

    uʷ = mean_squared_field(water_velocities[1], i, j, k1, k2)
    vʷ = mean_squared_field(water_velocities[2], i, j, k1, k2)
    wʷ = mean_squared_field(water_velocities[3], i, j, k1, k2)

    uʳ = uʷ - uⁱ
    vʳ = vʷ - vⁱ
    wʳ = wʷ - wⁱ

    sʳ = sqrt(uʳ^2 + vʳ^2 + wʳ^2)

    ∂ₜuʷ = mean_squared_field(water_accelerations[1], i, j, k1, k2)
    ∂ₜvʷ = mean_squared_field(water_accelerations[2], i, j, k1, k2)
    ∂ₜwʷ = mean_squared_field(water_accelerations[3], i, j, k1, k2)

    θ = acos(min(1, abs(uʳ * Δx + vʳ * Δy + wʳ * Δz) / (sʳ * l + eps(0.0))))
    Aˢ = 2 * rˢ * l * abs(sin(θ)) + π * rˢ * abs(cos(θ))

    Fᴰ₁ = 0.5 * ρₒ * (Cᵈˢ * Aˢ + Cᵈᵇ * Aᵇ) * sʳ * uʳ
    Fᴰ₂ = 0.5 * ρₒ * (Cᵈˢ * Aˢ + Cᵈᵇ * Aᵇ) * sʳ * vʳ
    Fᴰ₃ = 0.5 * ρₒ * (Cᵈˢ * Aˢ + Cᵈᵇ * Aᵇ) * sʳ * wʳ

    # Tension
    if n == size(relaxed_lengths, 2)
        x⁺, y⁺, z⁺ = 0, 0, 0
        u⁺, v⁺, w⁺ = 0, 0, 0

        Aᶜ⁺ = 0.0 
        l₀⁺ = relaxed_lengths[p, n] 
    else
        x⁺ = positions[p, n+1, 1]
        y⁺ = positions[p, n+1, 2]
        z⁺ = positions[p, n+1, 3]

        u⁺ = velocities[p, n+1, 1]
        v⁺ = velocities[p, n+1, 2]
        w⁺ = velocities[p, n+1, 3]

        Aᶜ⁺ = π * stipe_radii[p, n + 1] ^ 2
        l₀⁺ = relaxed_lengths[p, n + 1]
    end

    Aᶜ⁻ = π * rˢ ^ 2
    l₀⁻ = relaxed_lengths[p, n]

    Δx⁻ = x⁻ - xⁱ
    Δy⁻ = y⁻ - yⁱ
    Δz⁻ = z⁻ - zⁱ

    Δx⁺ = x⁺ - xⁱ
    Δy⁺ = y⁺ - yⁱ
    Δz⁺ = z⁺ - zⁱ

    l⁻ = sqrt(Δx⁻^2 + Δy⁻^2 + Δz⁻^2)
    l⁺ = sqrt(Δx⁺^2 + Δy⁺^2 + Δz⁺^2)

    T⁻  = tension(l⁻, l₀⁻, Aᶜ⁻, spring_constant, spring_exponent)
    T⁻₁ = T⁻ * Δx⁻ / (l⁻ + eps(0.0))
    T⁻₂ = T⁻ * Δy⁻ / (l⁻ + eps(0.0))
    T⁻₃ = T⁻ * Δz⁻ / (l⁻ + eps(0.0))

    T⁺ = tension(l⁺, l₀⁺, Aᶜ⁺, spring_constant, spring_exponent)
    T⁺₁ = T⁺ * Δx⁺ / (l⁺ + eps(0.0))
    T⁺₂ = T⁺ * Δy⁺ / (l⁺ + eps(0.0))
    T⁺₃ = T⁺ * Δz⁺ / (l⁺ + eps(0.0))

    # inertial force
    Fⁱ₁ = ρₒ * (Vᵐ + Vᵖ) * ∂ₜuʷ
    Fⁱ₂ = ρₒ * (Vᵐ + Vᵖ) * ∂ₜvʷ
    Fⁱ₃ = ρₒ * (Vᵐ + Vᵖ) * ∂ₜwʷ
    
    # add it all together

    accelerations[p, n, 1] = (Fᴰ₁ + T⁻₁ + T⁺₁ + Fⁱ₁) / mᵉ - velocities[p, n, 1] / τ
    accelerations[p, n, 2] = (Fᴰ₂ + T⁻₂ + T⁺₂ + Fⁱ₂) / mᵉ - velocities[p, n, 2] / τ
    accelerations[p, n, 3] = (Fᴰ₃ + T⁻₃ + T⁺₃ + Fⁱ₃ + Fᴮ) / mᵉ - velocities[p, n, 3] / τ

    drag_forces[p, n, 1] = Fᴰ₁
    drag_forces[p, n, 2] = Fᴰ₂
    drag_forces[p, n, 3] = Fᴰ₃
end

# This is only valid on a regularly spaced grid
# Benchmarks a lot lot faster than mean or sum()/dk etc. and about same speed as _interpolate which is weird
@inline function mean_squared_field(velocity, i::Int, j::Int, k1::Int, k2::Int)
    res = 0.0
    for k in k1:k2
        v = @inbounds velocity[i, j, k]
        res += v * abs(v)
    end
    return sign(res) * sqrt(abs(res)) / (k2 - k1 + 1)
end

@inline function mean_field(velocity, i::Int, j::Int, k1::Int, k2::Int)
    res = 0.0
    for k in k1:k2
        res += @inbounds velocity[i, j, k]
    end
    return res / (k2 - k1 + 1)
end

@inline tension(Δx, l₀, Aᶜ, k, α) = ifelse(Δx > l₀ && !(Δx == 0.0), k * (max(0, (Δx - l₀)) / l₀) ^ α * Aᶜ, 0.0)
