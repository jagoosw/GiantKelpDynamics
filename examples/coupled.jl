
using Oceananigans, GiantKelpDynamics, OceanBioME, Oceananigans.Units, GiantKelpGrowth, JLD2, Interpolations, CaliforniaBightNitrate
using OceanBioME: Biogeochemistry
using Oceananigans.Fields: FunctionField

@load "par_ts_2010_2022.jld2" PAR_itp # 2010/01/01
@load "temp_ts_2006_2024.jld2" temp_itp # 2010/01/01

no3_itp(t) = nitrate(temp_itp(t))

grid = RectilinearGrid(size = (256, 32, 32), extent = (100, 8, 8))

holdfast_x = [20.]
holdfast_y = [4.]

dynamics = GiantKelpDynamics.GiantKelp(; grid,
                       holdfast_x, holdfast_y,
                       number_nodes = 2,
                       segment_unstretched_length = 8,
                       kinematics = UtterDennySpeed(; turn_on_timescale = 0.0),
                       timestepper = GiantKelpDynamics.Newmarkβ())

n_blades = 32

growth = GiantKelpParticles(1, grid; n_blades, scalefactors = [0.5] .* n_blades/128)

@load "start_file.jld2" # As0 PARs0 lifespan0 age0 N0 C0

set!(growth, x = holdfast_x, y = holdfast_y, N = N0, C = C0)

for n in 1:n_blades
    growth.fields[Symbol(:A, n)] .= As0[n]
    growth.fields[Symbol(:PAR, n)] .= PARs0[n]
    growth.fields[Symbol(:τ, n)] .= age0[n]
    growth.fields[Symbol(:base_lifespan, n)] .= lifespan0[n]

    if n <= 5
        growth.fields[Symbol(:frond_depth, n)] .= depths0[n]
    end
end

@inline sponge(x, y, z) = ifelse(x < 10, 1, 0)

u = Relaxation(; rate = 1/20, target = 0.1, mask = sponge)
v = Relaxation(; rate = 1/20, mask = sponge)
w = Relaxation(; rate = 1/20, mask = sponge)

underlying_light_model = TwoBandPhotosyntheticallyActiveRadiation(; grid, surface_PAR=(x, y, t)->PAR_itp(t))

light_attenuation = GiantKelpGrowth.KelpShadedLight(grid, underlying_light_model)

biogeochemistry = LOBSTER(; grid, 
                            detritus = VariableRedfieldDetritus(grid), 
                            carbonate_system = CarbonateSystem(),
                            particles = (dynamics, growth), 
                            light_attenuation, 
                            scale_negatives = true)

clock = Clock(; time = one(eltype(grid)) * 730days)

T = FunctionField{Center, Center, Center}((x, y, z, t)->temp_itp(t), grid; clock)

@inline function restore_tracer(x, y, z, t, X, parameters)
    τ = parameters.timescale
    background = parameters.background

    X₀ = background(t)

    return (X₀ - X) / τ
end

τ = 1Units.hours#200 / 0.2

NO₃_forcing = Forcing(restore_tracer, field_dependencies = :NO₃, parameters = (; timescale = τ, background = no3_itp))

model = NonhydrostaticModel(grid; 
                            biogeochemistry,
                            advection = WENO(),
                            forcing = (; u, v, w, NO₃ = NO₃_forcing),
                            clock,
                            auxiliary_fields = (; T))

model.clock.time = 7.6248e7

set!(model, NO₃ = no3_itp(model.clock.time), DIC = 10000)#, P = 1)

simulation = Simulation(model, Δt = 0.5, stop_time = 10minutes)

prog(sim) = @info "Completed $(prettytime(time(sim))) in $(sim.model.clock.iteration) steps with Δt = $(prettytime(sim.Δt))"

simulation.callbacks[:progress] = Callback(prog, IterationInterval(100))

wizard = TimeStepWizard(cfl = 0.5, max_Δt=1)
simulation.callbacks[:timestep] = Callback(wizard, IterationInterval(10))

simulation.output_writers[:flow] = JLD2Writer(model, merge(model.velocities, model.tracers), overwrite_existing = true, filename = "single_coupled_flow.jld2", schedule = TimeInterval(10))
simulation.output_writers[:kelp] = JLD2Writer(model, dynamics.positions, overwrite_existing = true, filename = "single_coupled_kelp.jld2", schedule = TimeInterval(10))
simulation.output_writers[:kelp_growth] = JLD2Writer(model, growth.fields, overwrite_existing = true, filename = "single_coupled_kelp_bgc.jld2", schedule = TimeInterval(10))
