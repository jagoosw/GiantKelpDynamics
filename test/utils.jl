grid = RectilinearGrid(arch; size = (128, 128, 8), extent = (500, 500, 8))

spacing = 100.

x_pattern = 200.:spacing:300.
y_pattern = 200.:spacing:300.

holdfast_x = vec([x for x in x_pattern, y in y_pattern])
holdfast_y = vec([y for x in x_pattern, y in y_pattern])

number_nodes = 2
segment_unstretched_length = [16, 8]

kelp = GiantKelp(; grid,
                   holdfast_x, holdfast_y,
                   number_nodes,
                   segment_unstretched_length)

model = NonhydrostaticModel(; grid, 
                            biogeochemistry = Biogeochemistry(NothingBGC(),
                                                              particles = kelp),
                            advection = WENO())

@testset "Set" begin
    initial_positions = (x = [0, 0, 8], y = [0, 0, 0], z = [-8, 0, 0])

    set!(kelp, positions = initial_positions)

    @test all([all(Array(kelp.positions.x[p, :]) .== initial_positions.x) for p in 1:length(kelp)])
    @test all([all(Array(kelp.positions.y[p, :]) .== initial_positions.y) for p in 1:length(kelp)])
    @test all([all(Array(kelp.positions.z[p, :]) .== initial_positions.z) for p in 1:length(kelp)])

    initial_positions = (x = repeat([0, 0, 8], 1, 4)', y = 0, z = repeat([-8, 0, 0], 1, 4)')

    for p in 1:length(kelp)
        initial_positions.x[p, :] .= p
    end

    set!(kelp, positions = initial_positions)

    @test all(Array(kelp.positions.x) .== initial_positions.x)
    @test all(Array(kelp.positions.y) .== initial_positions.y)
    @test all(Array(kelp.positions.z) .== initial_positions.z)
end

@testset "Output" begin
    simulation = Simulation(model, Δt = 1, stop_iteration = model.clock.iteration + 10)

    simulation.output_writers[:kelp] = JLD2Writer(model, (; x = kelp.positions.x, y = kelp.positions.y, z = kelp.positions.z, blade_areas = kelp.blade_areas), overwrite_existing = true, schedule = IterationInterval(1), filename = "kelp.jld2")
    
    run!(simulation)

    # TODO: make some utility to load this stuff

    file = jldopen("kelp.jld2")

    @test keys(file["timeseries"]) == ["x", "y", "z", "blade_areas", "t"]

    indices = keys(file["timeseries/t"])

    x = [file["timeseries/x/$idx"] for idx in indices]
    blade_areas = [file["timeseries/blade_areas/$idx"] for idx in indices]

    close(file)

    @test all(x[end] .== Array(kelp.positions.x))

    @test all(blade_areas[end] .≈ Array(kelp.blade_areas))
end