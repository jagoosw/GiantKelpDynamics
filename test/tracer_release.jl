using Oceananigans.Operators: volume

holdfast_x = [5.]
holdfast_y = [5.]

max_Δt = 1.0

number_nodes = 2
segment_unstretched_length = [10., 10.]

@inline constant_tracer_release(i, j, k, p, n, grid, clock, particles, tracers, parameters) = 
    ifelse(n == 1, 0.1 / volume(i, j, k, grid, Center(), Center(), Center()), 0)

@inline scaled_tracer_release(i, j, k, p, n, grid, clock, particles, tracers, parameters) = 
    ifelse(n == 1, parameters * 0.1 / volume(i, j, k, grid, Center(), Center(), Center()), 0)

@inline total_released_tracer(t) = 0.1 * t

@testset "Tracer release" begin
    for horizontal_resolution in (10, 20), vertical_resolution in (10, 20)
        grid = RectilinearGrid(arch; size = (horizontal_resolution, horizontal_resolution, vertical_resolution), extent = (10, 10, 10))

        kelp = GiantKelp(; grid,
                        holdfast_x, holdfast_y,
                        number_nodes,
                        segment_unstretched_length,
                        tracer_forcing = (; C = Forcing(constant_tracer_release)))

        model = NonhydrostaticModel(; grid, 
                                    tracers = (:C, ),
                                    biogeochemistry = Biogeochemistry(NothingBGC(),
                                                                        particles = kelp),
                                    advection = WENO())

        kelp.positions.z[:, 2:3] .= 0

        concentration_record = Float64[]

        Δt = 1.

        CUDA.@allowscalar for n in 1:500
            time_step!(model, Δt)
            push!(concentration_record, sum([model.tracers.C[floor(Int, grid.Nx/2)+1, floor(Int, grid.Ny/2)+1, k] * volume(1, 1, k, grid, Center(), Center(), Center()) for k=1:grid.Nz]))
        end

        @test all([isapprox(conc, total_released_tracer(n * Δt), atol = 0.01) for (n, conc) in enumerate(concentration_record)])
    end

    # check scale factor and parametrers work
    grid = RectilinearGrid(arch; size = (10, 10, 10), extent = (10, 10, 10))

    kelp = GiantKelp(; grid,
                      holdfast_x, holdfast_y,
                      number_nodes,
                      segment_unstretched_length,
                      scalefactor = [2., 2.],
                      tracer_forcing = (; C = Forcing(scaled_tracer_release; parameters = 0.5)))

    model = NonhydrostaticModel(; grid, 
                                tracers = (:C, ),
                                biogeochemistry = Biogeochemistry(NothingBGC(),
                                                                    particles = kelp),
                                advection = WENO())

    kelp.positions.z[:, 2:3] .= 0

    concentration_record = Float64[]

    Δt = 1.

    CUDA.@allowscalar for n in 1:500
        time_step!(model, Δt)
        push!(concentration_record, sum([model.tracers.C[floor(Int, grid.Nx/2)+1, floor(Int, grid.Ny/2)+1, k] * volume(1, 1, k, grid, Center(), Center(), Center()) for k=1:grid.Nz]))
    end

    @test all([isapprox(conc, total_released_tracer(n * Δt), atol = 0.01) for (n, conc) in enumerate(concentration_record)])
end