using Oceananigans.Grids: XYZRegularRG, xspacings, yspacings, zspacings, topology

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

@inline function get_closest_ijk(grid::XYZRegularRG, X)
    Δx = grid.Δxᶜᵃᵃ
    Δy = grid.Δyᵃᶜᵃ
    Δz = grid.z.Δᵃᵃᶜ

    i = round(Int, X.x/Δx)
    j = round(Int, X.y/Δy)
    k = grid.Nz - round(Int, X.z/Δz)

    TX, TY, TZ = topology(grid)

    i = get_node(TX(), i, grid.Nx)
    j = get_node(TY(), j, grid.Ny)
    k = get_node(TZ(), k, grid.Nz)

    return (i, j, k)
end

@inline function mean_water_velocity(i, j, k1, k2, water_velocities)
    u = @inbounds mean_squared_field(water_velocities[1], i, j, k1, k2)
    v = @inbounds mean_squared_field(water_velocities[2], i, j, k1, k2)
    w = @inbounds mean_squared_field(water_velocities[3], i, j, k1, k2)

    return (x = u, y = v, z = w)
end

@inline function zero_vector!(p, n, X)
    @inbounds X.x[p, n] = 0
    @inbounds X.y[p, n] = 0
    @inbounds X.z[p, n] = 0

    return nothing
end

@inline function add_components!(p, n, X, F, unit_vector)
    @inbounds begin
        X.x[p, n] += @inbounds F * unit_vector.x
        X.y[p, n] += @inbounds F * unit_vector.y
        X.z[p, n] += @inbounds F * unit_vector.z
    end

    return nothing
end

@inline function multiply_components!(p, n, X, F, unit_vector = (x = 1, y = 1, z = 1))
    @inbounds begin
        X.x[p, n] *= @inbounds F * unit_vector.x
        X.y[p, n] *= @inbounds F * unit_vector.y
        X.z[p, n] *= @inbounds F * unit_vector.z
    end

    return nothing
end

@inline function set_components!(p, n, X, F, unit_vector)
    @inbounds begin
        X.x[p, n] = @inbounds F * unit_vector.x
        X.y[p, n] = @inbounds F * unit_vector.y
        X.z[p, n] = @inbounds F * unit_vector.z
    end

    return nothing
end

@inline mag(x, y, z) = sqrt(x^2 + y^2 + z^2)
@inline mag(X) = sqrt(X.x^2 + X.y^2 + X.z^2)