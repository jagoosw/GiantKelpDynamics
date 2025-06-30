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