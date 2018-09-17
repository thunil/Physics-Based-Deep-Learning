# Pysics-Based Deep Learning

This repository collects links to works on deep learning algorithms for physics
problems with a paricular emphasis on fluid flows, i.e., Navier-Stokes
problems. It contains links to the works of the I15 lab at TUM, as well as
miscellaneous works from other groups.  This is by no means a complete list, so
please help us if you come across additional papers in this area.

## I15

Latent-space Physics: Towards Learning the Temporal Evolution of Fluid Flow
https://ge.in.tum.de/publications/latent-space-physics/
This work focuses on pressure fields over time. In contrast to the others, it predicts the temporal evolution using the latent-space of a trained encoder network.

tempoGAN: A Temporally Coherent, Volumetric GAN for Super-resolution Fluid Flow
https://ge.in.tum.de/publications/tempogan/
This GAN approach directly synthesizes a temporally coherent state of an advected quantity, such as smoke.

Deep Fluids: A Generative Network for Parameterized Fluid Simulations
http://www.byungsoo.me/project/deep-fluids/
An encoder framework for learning to represent the space-time functions of liquids and smoke simulations.

Data-Driven Synthesis of Smoke Flows with CNN-based Feature Descriptors
http://ge.in.tum.de/publications/2017-sig-chu/
Flow descriptors are learned exploiting flow invariants, which are used to look up pre-computed patches of 4D data.

Liquid Splash Modeling with Neural Networks
https://ge.in.tum.de/publications/2018-mlflip-um/
This data-driven model captures sub-grid scale formation of droplets for liquid simulations.

Generating Liquid Simulations with Deformation-aware Neural Networks
https://ge.in.tum.de/publications/2017-prantl-defonn/
This method captures full solutions for classes of liquid problems in terms of space-time deformations, allowing for real-time interactions.

## Additional Links

Accelerating Eulerian Fluid Simulation With Convolutional Networks
https://cims.nyu.edu/~schlacht/CNNFluids.htm


## Software

mantaflow, general fluid simulation and deep learning framework
http://mantaflow.com


