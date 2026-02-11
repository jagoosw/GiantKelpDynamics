
using Oceananigans, GiantKelpDynamics, OceanBioME, Oceananigans.Units, GiantKelpGrowth, JLD2, Interpolations, CaliforniaBightNitrate
using OceanBioME: Biogeochemistry
using Oceananigans.Fields: FunctionField

@load "par_ts_2010_2022.jld2" PAR_itp # 2010/01/01
@load "temp_ts_2006_2024.jld2" temp_itp # 2010/01/01

no3_itp(t) = nitrate(temp_itp(t))

nitrate_itp = SimpleInterpolation(CaliforniaBightNitrate.VALUES[:inshore].itp.knots[1], 
                                  CaliforniaBightNitrate.VALUES[:inshore].itp.coefs; 
                                  arch, 
                                  closure = Walrus.Interpolations.Fixed(length(CaliforniaBightNitrate.VALUES[:inshore].itp.knots[1])))

temp_itp = SimpleInterpolation(temp_itp.itp.knots[1],
                               temp_itp.itp.coefs;
                               arch )

PAR_itp = SimpleInterpolation(PAR_itp.itp.knots[1],
                               PAR_itp.itp.coefs;
                               arch )


arch = CPU()#GPU()

grid = RectilinearGrid(arch; size = (100, 8, 8), extent = (100, 8, 8))

holdfast_x = [20.]
holdfast_y = [4.]

dynamics = GiantKelpDynamics.GiantKelp(; grid,
                       holdfast_x, holdfast_y,
                       number_nodes = 2,
                       segment_unstretched_length = 8,
                       kinematics = UtterDennySpeed(; turn_on_timescale = 0.0),
                       timestepper = GiantKelpDynamics.Newmarkβ())

set!(dynamics, positions = (x = [20, 20, 28], y = [4, 4, 4], z = [-8, 0, 0]))

n_blades = 32

initial_kelp_positions(x, y, z) = (20 < x < 20+16) & (3.5 < y < 4.5)

growth = GiantKelpParticles(grid; n_blades, scalefactors = [0.5] .* 128/n_blades,
                                  tracer_values = GiantKelpGrowth.TracerValues(grid, initial_kelp_positions))

@load "start_file.jld2" # As0 PARs0 lifespan0 age0 N0 C0

set!(growth, x = holdfast_x, y = holdfast_y, N = N0, C = C0, A = As0, PAR = PARs0, τ = age0, base_lifespan = lifespan0)

@inline sponge(x, y, z) = ifelse(x < 10, 1, 0)

u = Relaxation(; rate = 1/20, target = 0.1, mask = sponge)
v = Relaxation(; rate = 1/20, mask = sponge)
w = Relaxation(; rate = 1/20, mask = sponge)

underlying_light_model = TwoBandPhotosyntheticallyActiveRadiation(; grid, surface_PAR=PAR_itp)

light_attenuation = GiantKelpGrowth.KelpShadedLight(grid, underlying_light_model)

biogeochemistry = LOBSTER(; grid, 
                            detritus = VariableRedfieldDetritus(grid), 
                            particles = (dynamics, growth), 
                            light_attenuation, 
                            scale_negatives = true)

#biogeochemistry = Biogeochemistry(NothingBGC(); particles = dynamics)

@inline function restore_tracer(z, t, X, parameters)
    τ = parameters.timescale
    background = parameters.background
    temp = parameters.temp

    X₀ = background(temp(t))

    return (X₀ - X) / τ
end

@inline function restore_temp(z, t, X, parameters)
    τ = parameters.timescale
    temp = parameters.temp(t)

    return (temp - X) / τ
end

τ = 1Units.hours#200 / 0.2

NO₃_forcing = Forcing(restore_tracer, field_dependencies = :NO₃, parameters = (; timescale = τ, temp = temp_itp, background = nitrate_itp))

T_forcing = Forcing(restore_temp, field_dependencies = :T, parameters = (; timescale = τ, temp = temp_itp))

model = NonhydrostaticModel(grid; 
                            biogeochemistry,
                            advection = WENO(),
                            forcing = (; u, v, w, NO₃ = NO₃_forcing, T = T_forcing),
                            closure = AnisotropicMinimumDissipation(),
                            tracers = :T)

model.clock.time = 7.6248e7

set!(model, NO₃ = no3_itp(model.clock.time), u = 0.1, T = temp_itp(0))#, P = 1)

simulation = Simulation(model, Δt = 0.5, stop_time = 10minutes)

prog(sim) = @info "Completed $(prettytime(time(sim))) in $(sim.model.clock.iteration) steps with Δt = $(prettytime(sim.Δt))"

simulation.callbacks[:progress] = Callback(prog, IterationInterval(100))

wizard = TimeStepWizard(cfl = 0.5, max_Δt=1)
simulation.callbacks[:timestep] = Callback(wizard, IterationInterval(10))

simulation.output_writers[:flow] = JLD2Writer(model, merge(model.velocities, model.tracers), overwrite_existing = true, filename = "single_coupled_flow.jld2", schedule = TimeInterval(10))
simulation.output_writers[:kelp] = JLD2Writer(model, dynamics.positions, overwrite_existing = true, filename = "single_coupled_kelp.jld2", schedule = TimeInterval(10))
simulation.output_writers[:kelp_growth] = JLD2Writer(model, growth.fields, overwrite_existing = true, filename = "single_coupled_kelp_bgc.jld2", schedule = TimeInterval(10))
