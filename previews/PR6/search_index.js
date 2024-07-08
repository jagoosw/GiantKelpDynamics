var documenterSearchIndex = {"docs":
[{"location":"appendix/library/#library_api","page":"Library","title":"Library","text":"","category":"section"},{"location":"appendix/library/","page":"Library","title":"Library","text":"Documenting the user interface.","category":"page"},{"location":"appendix/library/#GiantKelpDynamics.jl","page":"Library","title":"GiantKelpDynamics.jl","text":"","category":"section"},{"location":"appendix/library/","page":"Library","title":"Library","text":"Modules = [GiantKelpDynamics]\nprivate = false","category":"page"},{"location":"appendix/library/#GiantKelpDynamics.GiantKelpDynamics","page":"Library","title":"GiantKelpDynamics.GiantKelpDynamics","text":"A coupled model for the motion (and in the future growth and biogeochemical interactions) of giant kelp (Macrocystis pyrifera).\n\nBased on the models proposed by Utter and Denny (1996) and Rosman et al. (2013), and used in Strong-Wright et al. (2023).\n\nImplemented in the framework of OceanBioME.jl(Strong-Wright et al., 2023) and the coupled with the fluid dynamics of Oceananigans.jl(Ramadhan et al., 2020).\n\n\n\n\n\n","category":"module"},{"location":"appendix/library/#GiantKelpDynamics.Euler","page":"Library","title":"GiantKelpDynamics.Euler","text":"Euler()\n\nSets up an Euler timestepper.\n\n\n\n\n\n","category":"type"},{"location":"appendix/library/#GiantKelpDynamics.GiantKelp-Tuple{}","page":"Library","title":"GiantKelpDynamics.GiantKelp","text":"GiantKelp(; grid, \n            holdfast_x, holdfast_y, holdfast_z,\n            scalefactor = ones(length(holdfast_x)),\n            number_nodes = 8,\n            segment_unstretched_length = 3.,\n            initial_stipe_radii = 0.004,\n            initial_blade_areas = 3.0 * (isa(segment_unstretched_length, Number) ? \n                                           ones(number_nodes) ./ number_nodes :\n                                           segment_area_fraction(segment_unstretched_length)),\n            initial_pneumatocyst_volume = (2.5 / (5 * 9.81)) .* (isa(segment_unstretched_length, Number) ?\n                                                                   1 / number_nodes .* ones(number_nodes) :\n                                                                   segment_unstretched_length ./ sum(segment_unstretched_length)),\n            kinematics = UtterDenny(),\n            timestepper = Euler(),\n            max_Δt = Inf,\n            tracer_forcing = NamedTuple(),\n            custom_dynamics = nothingfunc)\n\nConstructs a model of giant kelps with bases at holdfast_x, _y, _z.\n\nKeyword Arguments\n\ngrid: (required) the geometry to build the model on\nholdfast_x, holdfast_y, holdfast_z: An array of the base/holdfast positions of the individuals\nscalefactor: array of the scalefactor for each plant (used to allow each plant model to represnt the effect of multiple individuals)\nnumber_nodes: the number of nodes to split each individual interior\nsegment_unstretched_length: either a scalar specifying the unstretched length of all segments,   or an array of the length of each segment (at the moment each plant must have the same)\ninitial_stipe_radii: either a scalar specifying the stipe radii of all segments,   or an array of the stipe radii of each segment (at the moment each plant must have the same)\ninitial_blade_areas: an array of the blade area attatched to each segment\ninitial_pneumatocyst_volume: an array of the volume of pneumatocyst attatched to each segment\nkinematics: the kinematics model specifying the individuals motion\ntimestepper: the timestepper to integrate the motion with (at each substep)\nmax_Δt: the maximum timestep for integrating the motion\ntracer_forcing: a NamedTuple of Oceananigans.Forcings(func; field_dependencies, parameters) with for discrete form forcing only. Functions  must be of the form func(i, j, k, p, n, grid, clock, tracers, particles, parameters) where field_dependencies can be particle properties or  fields from the underlying model (tracers or velocities)\ncustom_dynamics: function of the form func(particles, model, bgc, Δt) to be executed at every timestep after the kelp model properties are updated.\n\nExample\n\njulia> using GiantKelpDynamics, Oceananigans\n\njulia> grid = RectilinearGrid(size=(16, 16, 16), extent=(100, 100, 8));\n\njulia> kelp = GiantKelp(; grid, holdfast_x = [10., 20.], holdfast_y = [10., 20], holdfast_z = [-8., -8.])\nGiant kelp (Macrocystis pyrifera) model with 2 individuals of 8 nodes. \n Base positions:\n - x ∈ [10.0, 20.0]\n - y ∈ [10.0, 20.0]\n - z ∈ [-8.0, -8.0]\n\n\n\n\n\n\n","category":"method"},{"location":"appendix/library/#GiantKelpDynamics.NothingBGC","page":"Library","title":"GiantKelpDynamics.NothingBGC","text":"NothingBGC()\n\nAn Oceananigans AbstractContinuousFormBiogeochemistry which specifies no biogeochemical interactions to allow the giant kelp model to be run alone.\n\nExample\n\njulia> using GiantKelpDynamics, Oceananigans, OceanBioME\n\njulia> grid = RectilinearGrid(size=(16, 16, 16), extent=(100, 100, 8));\n\njulia> kelp = GiantKelp(; grid, number_nodes = 2, holdfast_x = [10., 20.], holdfast_y = [10., 20], holdfast_z = [-8., -8.])\nGiant kelp (Macrocystis pyrifera) model with 2 individuals of 2 nodes. \n Base positions:\n - x ∈ [10.0, 20.0]\n - y ∈ [10.0, 20.0]\n - z ∈ [-8.0, -8.0]\n\njulia> biogeochemistry = Biogeochemistry(NothingBGC(); particles = kelp)\nNo biogeochemistry \n Light attenuation: Nothing\n Sediment: Nothing\n Particles: Giant kelp (Macrocystis pyrifera) model with 2 individuals of 2 nodes.\n Modifiers: Nothing\n\n\n\n\n\n\n","category":"type"},{"location":"appendix/library/#GiantKelpDynamics.RK3","page":"Library","title":"GiantKelpDynamics.RK3","text":"RK3(; γ :: G = (8//15, 5//12, 3//4),\n      ζ :: Z = (0.0, -17//60, -5//12)\n\nHolds parameters for a third-order Runge-Kutta-Wray time-stepping scheme described by Le and Moin (1991).\n\n\n\n\n\n","category":"type"},{"location":"appendix/library/#GiantKelpDynamics.nothingfunc-Tuple","page":"Library","title":"GiantKelpDynamics.nothingfunc","text":"nothingfunc(args...)\n\nReturns nothing for nothing(args...)\n\n\n\n\n\n","category":"method"},{"location":"appendix/library/#Oceananigans.Fields.set!-Tuple{GiantKelp}","page":"Library","title":"Oceananigans.Fields.set!","text":"set!(kelp::GiantKelp; kwargs...)\n\nSets the properties of the kelp model. The keyword arguments kwargs... take the form name=data, where name refers to one of the properties of kelp, and the data may be an array mathcing the size of the property for one individual (i.e. size(kelp.name[1])), or for all (i.e. size(kelp.name)).\n\nExample\n\njulia> using GiantKelpDynamics, Oceananigans\n\njulia> grid = RectilinearGrid(size=(16, 16, 16), extent=(100, 100, 8));\n\njulia> kelp = GiantKelp(; grid, number_nodes = 2, holdfast_x = [10., 20.], holdfast_y = [10., 20], holdfast_z = [-8., -8.])\nGiant kelp (Macrocystis pyrifera) model with 2 individuals of 2 nodes. \n Base positions:\n - x ∈ [10.0, 20.0]\n - y ∈ [10.0, 20.0]\n - z ∈ [-8.0, -8.0]\n\njulia> set!(kelp, positions = [0 0 8; 8 0 8])\n\njulia> initial_positions = zeros(2, 2, 3);\n\njulia> initial_positions[1, :, :] = [0 0 8; 8 0 8];\n\njulia> initial_positions[1, :, :] = [0 0 -8; 8 0 -8];\n\njulia> set!(kelp, positions = initial_positions)\n\n\n\n\n\n\n","category":"method"},{"location":"references/#References","page":"References","title":"References","text":"","category":"section"},{"location":"references/","page":"References","title":"References","text":"Ramadhan, A.; Wagner, G. L.; Hill, C.; Campin, J.-M.; Churavy, V.; Besard, T.; Souza, A.; Edelman, A.; Ferrari, R. and Marshall, J. (2020). Oceananigans.jl: Fast and friendly geophysical fluid dynamics on GPUs. Journal of Open Source Software 5, 2018.\n\n\n\nRosman, J. H.; Denny, M. W.; Zeller, R. B.; Monismith, S. G. and Koseff, J. R. (2013). Interaction of waves and currents with kelp forests (Macrocystis pyrifera): Insights from a dynamically scaled laboratory model. Limnology and Oceanography 58, 790–802.\n\n\n\nStrong-Wright, J.; Chen, S.; Constantinou, N. C.; Silvestri, S.; Wagner, G. L. and Taylor, J. R. (2023). OceanBioME.jl: A flexible environment for modelling the coupled interactions between ocean biogeochemistry and physics. Journal of Open Source Software 8, 5669.\n\n\n\nStrong-Wright, J. and Taylor, J. R. A model of tidal flow and tracer release in a giant kelp forest. In review.\n\n\n\nUtter, B. and Denny, M. (1996). Wave-induced forces ont he giant kelp macrosystis pyrifera(agardh): field test of a computational model. The Journal of Experimental Biology, 2645–2654.\n\n\n\n","category":"page"},{"location":"coming-soon/#Coming-soon","page":"Quick start","title":"Coming soon","text":"","category":"section"},{"location":"coming-soon/","page":"Quick start","title":"Quick start","text":"...","category":"page"},{"location":"generated/forest/","page":"Forest","title":"Forest","text":"EditURL = \"../../../examples/forest.jl\"","category":"page"},{"location":"generated/forest/#forest_example","page":"Forest","title":"Single plant","text":"","category":"section"},{"location":"generated/forest/","page":"Forest","title":"Forest","text":"In this example we setup a single plant in a narrow periodic channel to help understand the drag of the kelp on the water","category":"page"},{"location":"generated/forest/#Install-dependencies","page":"Forest","title":"Install dependencies","text":"","category":"section"},{"location":"generated/forest/","page":"Forest","title":"Forest","text":"First we check we have the dependencies installed","category":"page"},{"location":"generated/forest/","page":"Forest","title":"Forest","text":"using Pkg\npkg\"add Oceananigans OceanBioME GiantKelpDynamics CairoMakie JLD2\"","category":"page"},{"location":"generated/forest/","page":"Forest","title":"Forest","text":"Load the packages and setup the models","category":"page"},{"location":"generated/forest/","page":"Forest","title":"Forest","text":"using Oceananigans, GiantKelpDynamics, OceanBioME, Oceananigans.Units\nusing OceanBioME: Biogeochemistry\n\ngrid = RectilinearGrid(size = (128, 64, 8), extent = (1kilometer, 500, 8))\n\nxc, yc, zc = nodes(grid, Center(), Center(), Center())\n\nx_spacing = xc[27]:xspacings(grid, Center()):xc[38]\ny_spacing = yc[27]:yspacings(grid, Center()):yc[38]\n\nholdfast_x = vec([x for x in x_spacing, y in y_spacing])\nholdfast_y = vec([y for x in x_spacing, y in y_spacing])\nholdfast_z = vec([-8. for x in x_spacing, y in y_spacing])\n\nscalefactor = 1.5 * (xspacings(grid, Center()) * yspacings(grid, Center())) .* ones(length(holdfast_x))\n\nscalefactor = vec([x for x in x_spacing, y in y_spacing])\n\nnumber_nodes = 2\n\nsegment_unstretched_length = [16., 8.]\n\nmax_Δt = 1.\n\nkelp = GiantKelp(; grid,\n                   holdfast_x, holdfast_y, holdfast_z,\n                   scalefactor, number_nodes, segment_unstretched_length,\n                   max_Δt,\n                   initial_blade_areas = 3 .* [0.2, 0.8])\n\n@inline sponge(x, y, z) = ifelse(x < 100, 1, 0)\n\nu = Relaxation(; rate = 1/200, target = 0.05, mask = sponge)\nv = Relaxation(; rate = 1/200, mask = sponge)\nw = Relaxation(; rate = 1/200, mask = sponge)\n\nmodel = NonhydrostaticModel(; grid,\n                              biogeochemistry = Biogeochemistry(NothingBGC(),\n                                                                particles = kelp),\n                              advection = WENO(),\n                              forcing = (; u, v, w),\n                              closure = AnisotropicMinimumDissipation())","category":"page"},{"location":"generated/forest/","page":"Forest","title":"Forest","text":"NonhydrostaticModel{CPU, RectilinearGrid}(time = 0 seconds, iteration = 0)\n├── grid: 128×64×8 RectilinearGrid{Float64, Oceananigans.Grids.Periodic, Oceananigans.Grids.Periodic, Oceananigans.Grids.Bounded} on Oceananigans.Architectures.CPU with 3×3×3 halo\n├── timestepper: QuasiAdamsBashforth2TimeStepper\n├── tracers: ()\n├── closure: Oceananigans.TurbulenceClosures.AnisotropicMinimumDissipation{Oceananigans.TurbulenceClosures.ExplicitTimeDiscretization, @NamedTuple{}, Float64, Nothing}\n├── buoyancy: Nothing\n└── coriolis: Nothing","category":"page"},{"location":"generated/forest/","page":"Forest","title":"Forest","text":"Set the initial positions of the plant nodes (relaxed floating to the surface)","category":"page"},{"location":"generated/forest/","page":"Forest","title":"Forest","text":"set!(kelp, positions = [0. 0. 8.; 8. 0. 8.])#[13.86 0. 8.; 21.86 0. 8.;])","category":"page"},{"location":"generated/forest/","page":"Forest","title":"Forest","text":"Sset an initial water velocity with random noise to initial conditions to induce turbulance","category":"page"},{"location":"generated/forest/","page":"Forest","title":"Forest","text":"u₀(x, y, z) = 0.05 * (1 + 0.001 * randn())\nv₀(x, y, z) = 0.001 * randn()\n\nset!(model, u = u₀, v = v₀, w = v₀)","category":"page"},{"location":"generated/forest/","page":"Forest","title":"Forest","text":"Setup the simulaiton to save the flow and kelp positions","category":"page"},{"location":"generated/forest/","page":"Forest","title":"Forest","text":"simulation = Simulation(model, Δt = 20, stop_time = 4hours)\n\nprog(sim) = @info \"Completed $(prettytime(time(simulation))) in $(simulation.model.clock.iteration) steps with Δt = $(prettytime(simulation.Δt))\"\nsimulation.callbacks[:progress] = Callback(prog, IterationInterval(100))\n\nwizard = TimeStepWizard(cfl = 0.5)\nsimulation.callbacks[:timestep] = Callback(wizard, IterationInterval(10))\n\nsimulation.output_writers[:flow] = JLD2OutputWriter(model, model.velocities, overwrite_existing = true, filename = \"forest_flow.jld2\", schedule = TimeInterval(2minutes))\nsimulation.output_writers[:kelp] = JLD2OutputWriter(model, (; positions = kelp.positions), overwrite_existing = true, filename = \"forest_kelp.jld2\", schedule = TimeInterval(2minutes))","category":"page"},{"location":"generated/forest/","page":"Forest","title":"Forest","text":"JLD2OutputWriter scheduled on TimeInterval(2 minutes):\n├── filepath: ./forest_kelp.jld2\n├── 1 outputs: positions\n├── array type: Array{Float64}\n├── including: [:grid, :coriolis, :buoyancy, :closure]\n└── max filesize: Inf YiB","category":"page"},{"location":"generated/forest/","page":"Forest","title":"Forest","text":"Run!","category":"page"},{"location":"generated/forest/","page":"Forest","title":"Forest","text":"run!(simulation)","category":"page"},{"location":"generated/forest/","page":"Forest","title":"Forest","text":"[ Info: Initializing simulation...\n[ Info: Completed 0 seconds in 0 steps with Δt = 20 seconds\n[ Info:     ... simulation initialization complete (1.277 seconds)\n[ Info: Executing initial time step...\n[ Info:     ... initial time step complete (4.552 seconds).\n[ Info: Completed 50.685 minutes in 100 steps with Δt = 41.100 seconds\n[ Info: Completed 1.700 hours in 200 steps with Δt = 40.082 seconds\n[ Info: Completed 2.578 hours in 300 steps with Δt = 40.295 seconds\n[ Info: Completed 3.611 hours in 400 steps with Δt = 39.962 seconds\n[ Info: Simulation is stopping after running for 2.267 minutes.\n[ Info: Simulation time 4 hours equals or exceeds stop time 4 hours.\n","category":"page"},{"location":"generated/forest/","page":"Forest","title":"Forest","text":"Next we load the data","category":"page"},{"location":"generated/forest/","page":"Forest","title":"Forest","text":"using CairoMakie, JLD2\n\nu = FieldTimeSeries(\"forest_flow.jld2\", \"u\")\n\nfile = jldopen(\"forest_kelp.jld2\")\n\niterations = keys(file[\"timeseries/t\"])\n\npositions = [file[\"timeseries/positions/$it\"] for it in iterations]\n\nclose(file)\n\ntimes = u.times\n\nnothing","category":"page"},{"location":"generated/forest/","page":"Forest","title":"Forest","text":"Now we can animate the motion of the plant and attenuation of the flow","category":"page"},{"location":"generated/forest/","page":"Forest","title":"Forest","text":"n = Observable(1)\n\nx_position_first = @lift vec([positions[$n][p, 1, 1] for (p, x₀) in enumerate(holdfast_x)])\nz_position_first = @lift vec([positions[$n][p, 1, 3] for (p, z₀) in enumerate(holdfast_z)])\n\nabs_x_position_first = @lift vec([positions[$n][p, 1, 1] + x₀ for (p, x₀) in enumerate(holdfast_x)])\nabs_z_position_first = @lift vec([positions[$n][p, 1, 3] + z₀ for (p, z₀) in enumerate(holdfast_z)])\n\nx_position_ends = @lift vec([positions[$n][p, 2, 1] for (p, x₀) in enumerate(holdfast_x)])\ny_position_ends = @lift vec([positions[$n][p, 2, 2] for (p, y₀) in enumerate(holdfast_y)])\n\nrel_x_position_ends = @lift vec([positions[$n][p, 2, 1] - positions[$n][p, 1, 1] for (p, x₀) in enumerate(holdfast_x)])\nrel_z_position_ends = @lift vec([positions[$n][p, 2, 3] - positions[$n][p, 1, 3] for (p, z₀) in enumerate(holdfast_z)])\n\nu_vert = @lift interior(u[$n], :, Int(grid.Ny/2), :) .- 0.05\n\nu_surface = @lift interior(u[$n], :, :, grid.Nz) .- 0.05\n\nu_lims = (-0.06, 0.06)\n\nxf, yc, zc = nodes(u.grid, Face(), Center(), Center())\n\nfig = Figure(resolution = (1200, 800));\n\ntitle = @lift \"t = $(prettytime(u.times[$n]))\"\n\nax = Axis(fig[1:3, 1], aspect = DataAspect(); title, ylabel = \"y (m)\")\n\nhm = heatmap!(ax, xf, yc, u_surface, colorrange = u_lims, colormap = Reverse(:roma))\n\narrows!(ax, holdfast_x, holdfast_y, x_position_ends, y_position_ends, color = :black)\n\nax = Axis(fig[4, 1], limits = (190, 350, -8, 0), aspect = AxisAspect(15), xlabel = \"x (m)\", ylabel = \"z (m)\")\n\nhm = heatmap!(ax, xf, zc, u_vert, colorrange = u_lims, colormap = Reverse(:roma))\n\nColorbar(fig[1:4, 2], hm, label = \"Velocity anomaly (m / s)\")\n\narrows!(ax, holdfast_x, holdfast_z, x_position_first, z_position_first, color = :black)\narrows!(ax, abs_x_position_first, abs_z_position_first, rel_x_position_ends, rel_z_position_ends, color = :black)\n\nrecord(fig, \"forest.mp4\", 1:length(times); framerate = 10) do i;\n    n[] = i\nend","category":"page"},{"location":"generated/forest/","page":"Forest","title":"Forest","text":"\"forest.mp4\"","category":"page"},{"location":"generated/forest/","page":"Forest","title":"Forest","text":"(Image: ) In this video the limitations of the simplified drag stencil can be seen (see previous versions for a more complex stencil). It is better suited to the forest application like in the forest example","category":"page"},{"location":"generated/forest/","page":"Forest","title":"Forest","text":"","category":"page"},{"location":"generated/forest/","page":"Forest","title":"Forest","text":"This page was generated using Literate.jl.","category":"page"},{"location":"appendix/function_index/#Index","page":"Function index","title":"Index","text":"","category":"section"},{"location":"appendix/function_index/","page":"Function index","title":"Function index","text":"","category":"page"},{"location":"generated/single_plant/","page":"Single plant","title":"Single plant","text":"EditURL = \"../../../examples/single_plant.jl\"","category":"page"},{"location":"generated/single_plant/#single_example","page":"Single plant","title":"Single plant","text":"","category":"section"},{"location":"generated/single_plant/","page":"Single plant","title":"Single plant","text":"In this example we setup a single plant in a narrow periodic channel to help understand the drag of the kelp on the water","category":"page"},{"location":"generated/single_plant/#Install-dependencies","page":"Single plant","title":"Install dependencies","text":"","category":"section"},{"location":"generated/single_plant/","page":"Single plant","title":"Single plant","text":"First we check we have the dependencies installed","category":"page"},{"location":"generated/single_plant/","page":"Single plant","title":"Single plant","text":"using Pkg\npkg\"add Oceananigans OceanBioME GiantKelpDynamics CairoMakie JLD2\"","category":"page"},{"location":"generated/single_plant/","page":"Single plant","title":"Single plant","text":"Load the packages and setup the models","category":"page"},{"location":"generated/single_plant/","page":"Single plant","title":"Single plant","text":"using Oceananigans, GiantKelpDynamics, OceanBioME, Oceananigans.Units\nusing OceanBioME: Biogeochemistry\n\ngrid = RectilinearGrid(size = (256, 32, 32), extent = (100, 8, 8))\n\nholdfast_x = [20.]\nholdfast_y = [4.]\nholdfast_z = [-8.]\n\nmax_Δt = 0.5\n\nkelp = GiantKelp(; grid,\n                   holdfast_x, holdfast_y, holdfast_z,\n                   max_Δt)\n\n@inline sponge(x, y, z) = ifelse(x < 10, 1, 0)\n\nu = Relaxation(; rate = 1/20, target = 0.1, mask = sponge)\nv = Relaxation(; rate = 1/20, mask = sponge)\nw = Relaxation(; rate = 1/20, mask = sponge)\n\nmodel = NonhydrostaticModel(; grid,\n                              biogeochemistry = Biogeochemistry(NothingBGC(),\n                                                                particles = kelp),\n                              advection = WENO(),\n                              forcing = (; u, v, w),\n                              closure = AnisotropicMinimumDissipation())","category":"page"},{"location":"generated/single_plant/","page":"Single plant","title":"Single plant","text":"NonhydrostaticModel{CPU, RectilinearGrid}(time = 0 seconds, iteration = 0)\n├── grid: 256×32×32 RectilinearGrid{Float64, Oceananigans.Grids.Periodic, Oceananigans.Grids.Periodic, Oceananigans.Grids.Bounded} on Oceananigans.Architectures.CPU with 3×3×3 halo\n├── timestepper: QuasiAdamsBashforth2TimeStepper\n├── tracers: ()\n├── closure: Oceananigans.TurbulenceClosures.AnisotropicMinimumDissipation{Oceananigans.TurbulenceClosures.ExplicitTimeDiscretization, @NamedTuple{}, Float64, Nothing}\n├── buoyancy: Nothing\n└── coriolis: Nothing","category":"page"},{"location":"generated/single_plant/","page":"Single plant","title":"Single plant","text":"Set the initial positions of the plant nodes (relaxed floating to the surface), and the set an initial water velocity","category":"page"},{"location":"generated/single_plant/","page":"Single plant","title":"Single plant","text":"set!(kelp, positions = [0. 0. 3.; 0. 0. 6.; 0. 0. 8.; -3. 0. 8.; -6. 0. 8.; -9. 0. 8.; -12. 0. 8.; -9. 0. 8.;])\n\nset!(model, u = 0.1)","category":"page"},{"location":"generated/single_plant/","page":"Single plant","title":"Single plant","text":"Setup the simulaiton to save the flow and kelp positions","category":"page"},{"location":"generated/single_plant/","page":"Single plant","title":"Single plant","text":"simulation = Simulation(model, Δt = 0.5, stop_time = 10minutes)\n\nprog(sim) = @info \"Completed $(prettytime(time(simulation))) in $(simulation.model.clock.iteration) steps with Δt = $(prettytime(simulation.Δt))\"\n\nsimulation.callbacks[:progress] = Callback(prog, IterationInterval(100))\n\nwizard = TimeStepWizard(cfl = 0.5)\nsimulation.callbacks[:timestep] = Callback(wizard, IterationInterval(10))\n\nsimulation.output_writers[:flow] = JLD2OutputWriter(model, model.velocities, overwrite_existing = true, filename = \"single_flow.jld2\", schedule = TimeInterval(10))\nsimulation.output_writers[:kelp] = JLD2OutputWriter(model, (; positions = kelp.positions), overwrite_existing = true, filename = \"single_kelp.jld2\", schedule = TimeInterval(10))","category":"page"},{"location":"generated/single_plant/","page":"Single plant","title":"Single plant","text":"JLD2OutputWriter scheduled on TimeInterval(10 seconds):\n├── filepath: ./single_kelp.jld2\n├── 1 outputs: positions\n├── array type: Array{Float64}\n├── including: [:grid, :coriolis, :buoyancy, :closure]\n└── max filesize: Inf YiB","category":"page"},{"location":"generated/single_plant/","page":"Single plant","title":"Single plant","text":"Run!","category":"page"},{"location":"generated/single_plant/","page":"Single plant","title":"Single plant","text":"run!(simulation)","category":"page"},{"location":"generated/single_plant/","page":"Single plant","title":"Single plant","text":"[ Info: Initializing simulation...\n[ Info: Completed 0 seconds in 0 steps with Δt = 500 ms\n[ Info:     ... simulation initialization complete (6.848 seconds)\n[ Info: Executing initial time step...\n[ Info:     ... initial time step complete (6.008 seconds).\n[ Info: Completed 1.398 minutes in 100 steps with Δt = 1.297 seconds\n[ Info: Completed 4.029 minutes in 200 steps with Δt = 1.726 seconds\n[ Info: Completed 6.781 minutes in 300 steps with Δt = 1.712 seconds\n[ Info: Completed 9.325 minutes in 400 steps with Δt = 1.583 seconds\n[ Info: Simulation is stopping after running for 7.163 minutes.\n[ Info: Simulation time 10 minutes equals or exceeds stop time 10 minutes.\n","category":"page"},{"location":"generated/single_plant/","page":"Single plant","title":"Single plant","text":"Next we load the data","category":"page"},{"location":"generated/single_plant/","page":"Single plant","title":"Single plant","text":"using CairoMakie, JLD2\n\nu = FieldTimeSeries(\"single_flow.jld2\", \"u\")\n\nfile = jldopen(\"single_kelp.jld2\")\n\niterations = keys(file[\"timeseries/t\"])\n\npositions = [file[\"timeseries/positions/$it\"] for it in iterations]\n\nclose(file)\n\ntimes = u.times\n\nnothing","category":"page"},{"location":"generated/single_plant/","page":"Single plant","title":"Single plant","text":"Now we can animate the motion of the plant and attenuation of the flow","category":"page"},{"location":"generated/single_plant/","page":"Single plant","title":"Single plant","text":"n = Observable(1)\n\nx_position = @lift positions[$n][1, :, 1] .+ 20\ny_position = @lift positions[$n][1, :, 2] .+ 4\nz_position = @lift positions[$n][1, :, 3] .- 8\n\nu_vert = @lift interior(u[$n], :, Int(grid.Ny / 2), :)\n\nu_surface = @lift interior(u[$n], :, :, grid.Nz)\n\nxf, yc, zc = nodes(u.grid, Face(), Center(), Center())\n\nfig = Figure(resolution = (1200, 400));\n\ntitle = @lift \"t = $(prettytime(u.times[$n]))\"\n\nax = Axis(fig[1, 1], aspect = DataAspect(); title, ylabel = \"z (m)\")\n\nhm = heatmap!(ax, xf, zc, u_vert, colormap = :lajolla)\n\nscatter!(ax, x_position, z_position, color = :black)\n\nax = Axis(fig[2, 1], aspect = DataAspect(), xlabel = \"x (m)\", ylabel = \"y (m)\")\n\nhm = heatmap!(ax, xf, yc, u_surface, colormap = :lajolla)\n\nscatter!(ax, x_position, y_position, color = :black)\n\nrecord(fig, \"single.mp4\", 1:length(times); framerate = 10) do i;\n    n[] = i\nend","category":"page"},{"location":"generated/single_plant/","page":"Single plant","title":"Single plant","text":"\"single.mp4\"","category":"page"},{"location":"generated/single_plant/","page":"Single plant","title":"Single plant","text":"(Image: ) In this video the limitations of the simplified drag stencil can be seen (see previous versions for a more complex stencil). It is better suited to the forest application like in the forest example","category":"page"},{"location":"generated/single_plant/","page":"Single plant","title":"Single plant","text":"","category":"page"},{"location":"generated/single_plant/","page":"Single plant","title":"Single plant","text":"This page was generated using Literate.jl.","category":"page"},{"location":"#GiantKelpDynamics","page":"Home","title":"GiantKelpDynamics","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"GiantKelpDynamicsjl is a Julia package providing a dynamical model for the motion (and in the future the growth and biogeochemical interactions) of giant kelp (Macrocystis pyrifera).","category":"page"},{"location":"","page":"Home","title":"Home","text":"The kinematic model is based on the work of Utter and Denny (1996) and Rosman et al. (2013), and used in Strong-Wright and Taylor (undated).","category":"page"},{"location":"","page":"Home","title":"Home","text":"We have implemented it in the particle in the framework of Strong-Wright et al. (2023) and the coupled with the fluid dynamics of Ramadhan et al. (2020).","category":"page"},{"location":"#Model-details","page":"Home","title":"Model details","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"The kinematic model discretises each kelp individual into segments to which several forces: drag, buoyancy, tension, and inertial, are applied. This is shown in this diagram:","category":"page"},{"location":"","page":"Home","title":"Home","text":"(Image: Diagram of the discretisation and forces on each segment)","category":"page"},{"location":"","page":"Home","title":"Home","text":"Further details can be found in Strong-Wright and Taylor (undated).","category":"page"},{"location":"#Kelp-forests","page":"Home","title":"Kelp forests","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"The model is designed to be used to simulate kelp forests and reproduce their motion well (Strong-Wright and Taylor, undated). This figure shows an example of the flow around a forest:","category":"page"},{"location":"","page":"Home","title":"Home","text":"(Image: Tracer released from a kelp forest within a simple tidal flow)","category":"page"}]
}
