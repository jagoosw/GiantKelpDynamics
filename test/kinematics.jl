grid = RectilinearGrid(arch; size = (128, 128, 8), extent = (500, 500, 8))

spacing = 100.

x_pattern = 200.:spacing:300.
y_pattern = 200.:spacing:300.

holdfast_x = vec([x for x in x_pattern, y in y_pattern])
holdfast_y = vec([y for x in x_pattern, y in y_pattern])

number_nodes = 2
segment_unstretched_length = [16, 8]

@testset "Kelp move" begin
    kelp = GiantKelp(; grid,
                       holdfast_x, holdfast_y,
                       number_nodes,
                       segment_unstretched_length)

    model = NonhydrostaticModel(; grid, 
                                  biogeochemistry = Biogeochemistry(NothingBGC(),
                                                                    particles = kelp),                          
                                  advection = WENO(),
                                  timestepper = :QuasiAdamsBashforth2)

    kelp.positions.z .= 0

    time_step!(model, 10.)

    # not moving when no flow and unstretched

    CUDA.@allowscalar @test (all(Array(kelp.positions.x) .== holdfast_x) & all(Array(kelp.positions.y) .== holdfast_y) & all(Array(kelp.positions.z) .== 0))

    kelp.positions.x[:, 2] .= 15
    kelp.positions.x[:, 3] .= 25
    kelp.positions.z[:, 1] .= -8

    time_step!(model, 300)

    position_record = (x = Array(copy(kelp.positions.x)), y = Array(copy(kelp.positions.y)), z = Array(copy(kelp.positions.z)))

    time_step!(model, 300)

    CUDA.@allowscalar begin
        @test ((all(isapprox.(Array(kelp.positions.x), position_record.x, atol = 0.1)) & 
                all(isapprox.(Array(kelp.positions.y), position_record.y, atol = 0.1)) & 
                all(isapprox.(Array(kelp.positions.z), position_record.z, atol = 0.1))))
    end

    #@info @btime time_step!($model, 1) #32ms
end

@testset "Drag" begin
    scalefactor = ones(length(holdfast_x))

    kelp = GiantKelp(; grid,
                       holdfast_x, holdfast_y,
                       number_nodes,
                       segment_unstretched_length)

    model = NonhydrostaticModel(; grid, 
                                  biogeochemistry = Biogeochemistry(NothingBGC(),
                                                                    particles = kelp),
                                  advection = WENO())

    u₀ = 0.2

    kelp.positions.x[:, 3] .+= 8
    kelp.positions.z[:, 2:3] .= 0

    initial_x = Array(copy(kelp.positions.x))

    set!(model, u = u₀)

    Δt = 0.5 * minimum_xspacing(grid) / u₀

    for n in 1:10
        time_step!(model, Δt)
    end

    # the kelp are being moved by the flow
    
    CUDA.@allowscalar @test !any(isapprox.(initial_x[:, 2:3], Array(kelp.positions.x[:, 2:3]); atol = 0.001))

    # the kelp are dragging the water
    @test !(mean(model.velocities.u) ≈ u₀)
    @test !isapprox(maximum(abs, model.velocities.v), 0)
end