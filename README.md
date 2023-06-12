# Giant kelp (_Macrocystis pyrifera_) dynamics model
This repository contains the code for numerically integrating the motion of giant kelp coupled with solving the flow around the plants using [`Oceananigans.jl`](https://github.com/CliMA/Oceananigans.jl/). 

This repository supports our paper ["Flow interaction and nutrient uptake dynamics in an idealised giant kelp forest"](comming.soon) which contains details of the model and implimentation.

The numerical model of giant kelp motion was based on the work of [Utter, B. and Denny, M. (1996)](https://doi.org/10.1242/jeb.199.12.2645) and [Rosman et al. (2013)](https://doi.org/10.4319/lo.2013.58.3.0790). 

This repository also contains code to initialize a forest of plants, and for the forest to release a tracer into the water. Please see the `density_experiment.jl` example, simpler examples coming soon.