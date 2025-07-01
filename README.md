# GiantKelpDynamics
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10727202.svg)](https://doi.org/10.5281/zenodo.10727202)

``GiantKelpDynamics.jl`` is a Julia package providing a dynamical model for the motion (and in the future the growth and biogeochemical interactions) of giant kelp (Macrocystis pyrifera).

The kinematic model is based on the work of [Utter and Denny, 1996](https://doi.org/10.4319/lo.2013.58.3.0790) and [Rosman et al., 2013](https://doi.org/10.4319/lo.2013.58.3.0790), and used in [Strong-Wright and Taylor, 2024](https://doi.org/10.1017/flo.2024.13).

https://github.com/jagoosw/GiantKelpDynamics/assets/26657828/2383251b-656a-4f1d-aaee-d16605592045

We have implemented it in the particle in the framework of [OceanBioME.jl](https://github.com/OceanBioME/OceanBioME.jl/) and coupled with the fluid dynamics of [Oceananigans.jl](https://github.com/CliMA/Oceananigans.jl/).

> Please note that the release [v0.0.1](https://doi.org/10.5281/zenodo.10727203) accompanies the publication [*A model of tidal flow and tracer release in a giant kelp forest* (Strong-Wright and Taylor, 2024)](https://doi.org/10.1017/flo.2024.13) and contains the version of the code used for the experiments in that work. We **do not** recommend using that version and suggest that the current version is used as it has been substantially rewritten to improve the maintainability of the code (while leaving the mathematical model of the motion unchanged) and therefore a significant amount of breaking changes have been introduced since this version.

## Model details
The kinematic model discretises each kelp individual into segments to which several forces: drag, buoyancy, tension, and inertial, are applied. This is shown in this diagram:

![Diagram of the discretization and forces on each segment](https://github.com/jagoosw/GiantKelpDynamics/assets/26657828/4a8aef46-e5c8-45b0-939e-0bbc281b3253)

Further details can be found in [Strong-Wright and Taylor, 2024](https://doi.org/10.1017/flo.2024.13 ).

## Kelp forests
The model is designed to be used to simulate kelp forests and reproduce their motion ([Strong-Wright and Taylor, 2024](https://doi.org/10.1017/flo.2024.13)). This figure shows an example of the flow around a forest:

![Tracer released from a kelp forest within a simple tidal flow](https://github.com/jagoosw/GiantKelpDynamics/assets/26657828/4df8b614-240f-4e44-bec1-4bbae4ebd7bb)

## Documentation
Simple examples and documentation can be found [here](mailto:js2430@cam.ac.uk)

## Citation
If you use this model in your work please cite [Strong-Wright and Taylor, 2023](mailto:js2430@cam.ac.uk) and link to this repository. You may also cite the specific version used by citing the [Zenode archive](https://zenodo.org/records/10727202).
