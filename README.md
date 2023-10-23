# GiantKelpDynamics

``GiantKelpDynamics.jl`` is a Julia package providing a dynamical model for the motion (and in the future the growth and biogeochemical interactions) of giant kelp (Macrocystis pyrifera).

The kinematic model is based on the work of [Utter and Denny, 1996](https://doi.org/10.4319/lo.2013.58.3.0790) and [Rosman et al., 2013](https://doi.org/10.4319/lo.2013.58.3.0790), and used in [Strong-Wright and Taylor, 2023 (Submitted)](https://coming-soon).

We have implemented it in the particle in the framework of [OceanBioME.jl](https://github.com/OceanBioME/OceanBioME.jl/) and coupled with the fluid dynamics of [Oceananigans.jl](https://github.com/CliMA/Oceananigans.jl/).

## Model details


The kinematic model discretises each kelp individual into segments to which several forces: drag, buoyancy, tension, and inertial, are applied. This is shown in this diagram:


![Diagram of the discretisation and forces on each segment](https://github.com/jagoosw/GiantKelpDynamics/assets/26657828/887a0860-83ff-44f6-bb64-cc7a859a05d9)

Further details can be found in [Strong-Wright and Taylor, 2023 (Submitted)](https://coming-soon).

## Kelp forests
The model is designed to be used to simulate kelp forests and reproduce their motion well ([Strong-Wright and Taylor, 2023](https://coming-soon)). This figure shows an example of the flow around a forest:


![Tracer released from a kelp forest within a simple tidal flow](https://github.com/jagoosw/GiantKelpDynamics/assets/26657828/4df8b614-240f-4e44-bec1-4bbae4ebd7bb)
![Uploading forest.png…]()


## Documentation
Simple examples and documentation can be found [here](https://coming-soon)
