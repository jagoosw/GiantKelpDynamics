"""
    Euler()

Sets up an Euler timestepper.
"""
struct Euler end

@inline function (::Euler)(u⃗, u⃗⁻, Δt, args...)
    return Δt * u⃗
end

@inline function (ts::Euler)(p, n, u⃗, u⃗⁻, Δt)
    x = ts(u⃗.x[p, n], u⃗⁻.x[p, n], Δt)
    y = ts(u⃗.y[p, n], u⃗⁻.y[p, n], Δt)
    z = ts(u⃗.z[p, n], u⃗⁻.z[p, n], Δt)
    return (; x, y, z)
end

@kernel function step_nodes!(accelerations, old_accelerations, velocities, old_velocities, positions, timestepper, Δt, ::Val{N}) where N
    p = @index(Global)

    @inbounds for n=2:N
        copy_components!(p, n, old_velocities, velocities)

        dU = timestepper(p, n, accelerations, old_accelerations, Δt)

        add_components!(p, n, velocities, 1, dU)

        copy_components!(p, n, old_accelerations, accelerations)

        dX = timestepper(p, n, velocities, old_velocities, Δt)

        add_components!(p, n, positions, 1, dX)

        positions.z[p, n] = ifelse(positions.z[p, n] > 0.0, zero(eltype(accelerations.x)), positions.z[p, n])
    end
end


@kernel function step_nodes!(accelerations, old_accelerations, velocities, old_velocities, positions, timestepper, Δt)
    p, n = @index(Global, NTuple)

    n += 1

    @inbounds begin
        copy_components!(p, n, old_velocities, velocities)

        dU = timestepper(p, n, accelerations, old_accelerations, Δt)

        add_components!(p, n, velocities, 1, dU)

        copy_components!(p, n, old_accelerations, accelerations)

        dX = timestepper(p, n, velocities, old_velocities, Δt)

        add_components!(p, n, positions, 1, dX)

        positions.z[p, n] = ifelse(positions.z[p, n] > 0.0, zero(eltype(accelerations.x)), positions.z[p, n])
    end
end

@inline function copy_components!(p, n, A, B)
    @inbounds begin
        A.x[p, n] = B.x[p, n]
        A.y[p, n] = B.y[p, n]
        A.z[p, n] = B.z[p, n]
    end

    return nothing
end

@kwdef struct Newmarkβ{FT, IT}
             γ :: FT = 0.5
             β :: FT = 0.25
             ω :: FT = 0.3
    iterations :: IT = 3
end

@kernel function predictor_step!(ts::Newmarkβ, Δt, position, velocity, acceleration, old_position, old_velocity, old_acceleration)
    p, n = @index(Global, NTuple)
    # follwing inital acceperation compoitation

    add_vector_components!(p, n, position, Δt, velocity)
    add_vector_components!(p, n, position, Δt^2/2, acceleration)

    add_vector_components!(p, n, velocity, Δt, acceleration)

    # old_accelerations is Ak
    copy_components!(p, n, old_acceleration, acceleration)

    # old velocitys in -γΔt An and positions ...
    copy_components!(p, n, old_velocity, acceleration)
    multiply_components!(p, n, old_velocity, -ts.γ * Δt)

    copy_components!(p, n, old_position, acceleration)
    multiply_components!(p, n, old_position, -ts.β * Δt^2)
    # so that positions are X*, velocitys are V*, and old_accelerations are An
end

@kernel function corrector_step!(ts::Newmarkβ, Δt, position, velocity, acceleration, old_position, old_velocity, old_acceleration)
    p, n = @index(Global, NTuple)
    # acceleration is always Anew = A(x*, v*) so we set old acceleration to the new Ak
    multiply_components!(p, n, old_acceleration, 1-ts.ω)
    add_vector_components!(p, n, old_acceleration, ts.ω, acceleration)

    # corrector
    add_vector_components!(p, n, position, ts.β * Δt^2, acceleration)
    add_vector_components!(p, n, position, 1, old_position) # - βΔt^2Aₙ

    add_vector_components!(p, n, velocity, ts.γ * Δt, acceleration)
    add_vector_components!(p, n, velocity, 1, old_velocity) # -γΔtAₙ
end
