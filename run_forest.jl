include("kelp_forest.jl")

using JLD2

output_dir = joinpath(@__DIR__, ARGS[1])
member = parse(Int64, ARGS[2])

@load "experiment_params.jl" forest_density 

simulaiton = setup_forest(arch; Lx = 5kilometers, Ly = 2kilometers, forest_density = forest_density[member])

wizard = TimeStepWizard(cfl = 0.8, max_change = 1.1, min_change = 0.8)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

progress_message(sim) = @printf("Iteration: %07d, time: %s, Δt: %s, max(|u|) = %.1e ms⁻¹, min(|u|) = %.1e ms⁻¹, wall time: %s, min(|U|) = %.1e , max(|O|) = %.1e \n",
                                iteration(sim), prettytime(sim), prettytime(sim.Δt),
                                maximum(abs, sim.model.velocities.u), minimum(abs, sim.model.velocities.u), prettytime(sim.run_wall_time),
                                minimum(sim.model.tracers.U), maximum(sim.model.tracers.O))
    
simulation.callbacks[:progress] = Callback(progress_message, TimeInterval(5minute))

simulation.output_writers[:profiles] =
    JLD2OutputWriter(model, merge(model.velocities, model.tracers),
                     filename = "$(output_dir)/$member.jld2",
                     schedule = TimeInterval(5minute),
                     overwrite_existing = true)

function store_particles!(sim)
    jldopen("$(output_dir)/$(member)_particles.jld2", "a+") do file
        file["x⃗/$(sim.model.clock.time)"] = sim.model.particles.properties.positions
    end
end

simulation.callbacks[:save_particles] = Callback(store_particles!, TimeInterval(5minute))

simulation.stop_time = 2.5 * 2π / 1.41e-4

run!(simulation)