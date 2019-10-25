# Physics-Based Deep Learning

The following collection of materials targets _"Physics-Based Deep Learning"_
(PBDL), i.e. the field of methods with combinations of physical modeling and
deep learning (DL) techniques. Here, DL will typically refer to methods based
on artificial neural networks. The general direction of PBDL represents a very
active and quickly growing field of research. 

![An overview of categories of physics-based deep learning methods](resources/physics-based-deep-learning-overview.jpg)

Within this area we can roughly distinguish the following five categories:

- _Physics-based design_: incorporate domain knowledge about the
  system in the architecture of the learning process, e.g.,
  adapt the connectivity of a neural network such that it facilitates 
  certain solutions.

- _Physics-based constraints_: guide and constrain the learning
  process with physical models, e.g., in the simplest case
  by including additional terms in the loss function.

- _Combined methods_: hybrid solvers that employ traditional
  numerical methods alongside deep-learning techniques to arrive
  at simulations methods that are improved in terms of, e.g., 
  efficiency, accuracy, generalization etc.

- _Outer-loop optimizations_:
  approaches that aim for higher level control or inverse
  problems, typically with an outer loop around a simulation.
  Here, fast and differentiable models exhibit particular promise.

- _Applications_: there are also many methods that apply established DL
  techniques to physical problems without really changing them. These
  works are nonetheless often highly interesting and much needed
  to establish methodologies of how DL can be best used for physical
  problems.

In practice, these categories only serve as a rough guide to the space of
methods in the area of physics-based deep learning, and there are many works
below that combine aspects from more than one of these classes.

This repository collects links to works on _deep learning algorithms for physics
problems_, with a particular emphasis on _fluid flow_, i.e., Navier-Stokes related
problems. It especially collects links to the works of the I15 lab at TUM, as
well as miscellaneous works from other groups. This is by no means a complete
list, so let us know if you come across additional papers in this area. We
intentionally also focus on works from the _deep learning_ field, not machine
learning in general.

![An example flow result from tempoGAN](resources/physics-based-deep-learning-teaser1.jpg)


## I15 Links

tempoGAN: A Temporally Coherent, Volumetric GAN for Super-resolution Fluid Flow , 
Project+Code: <https://ge.in.tum.de/publications/tempogan/>

Deep Fluids: A Generative Network for Parameterized Fluid Simulations , 
Project+Code: <http://www.byungsoo.me/project/deep-fluids/>

Latent-space Physics: Towards Learning the Temporal Evolution of Fluid Flow , 
Project+Code: <https://ge.in.tum.de/publications/latent-space-physics/>

A Multi-Pass GAN for Fluid Flow Super-Resolution , 
PDF: <https://ge.in.tum.de/publications/2019-multi-pass-gan/>

A Study of Deep Learning Methods for Reynolds-Averaged Navier-Stokes Simulations , 
Project+Code: <https://github.com/thunil/Deep-Flow-Prediction>

Data-Driven Synthesis of Smoke Flows with CNN-based Feature Descriptors , 
Project+Code: <http://ge.in.tum.de/publications/2017-sig-chu/>

Liquid Splash Modeling with Neural Networks , 
Project+Code: <https://ge.in.tum.de/publications/2018-mlflip-um/>

Generating Liquid Simulations with Deformation-aware Neural Networks , 
Project+Code: <https://ge.in.tum.de/publications/2017-prantl-defonn/>


## Additional Links for Fluids

Dynamic Upsampling of Smoke through Dictionary-based Learning , 
PDF: <https://arxiv.org/abs/1910.09166>

Computing interface curvature from volume fractions: A machine learning approach , 
PDF: <https://www.sciencedirect.com/science/article/abs/pii/S0045793019302282>

Deep Neural Networks for Data-Driven Turbulence Models , 
PDF: <https://export.arxiv.org/pdf/1806.04482>

Deep Dynamical Modeling and Control of Unsteady Fluid Flows , 
PDF: <http://papers.nips.cc/paper/8138-deep-dynamical-modeling-and-control-of-unsteady-fluid-flows>

Learning Particle Dynamics for Manipulating Rigid Bodies, Deformable Objects, and Fluids , 
Project+Code: <http://dpi.csail.mit.edu>

Reasoning About Liquids via Closed-Loop Simulation , 
PDF: <https://arxiv.org/pdf/1703.01656>

Application of Convolutional Neural Network to Predict Airfoil Lift Coefficient , 
PDF: <https://arxiv.org/pdf/1712.10082>

Lat-Net: Compressing Lattice Boltzmann Flow Simulations using Deep Neural Networks , 
PDF: <https://arxiv.org/pdf/1705.09036>

Prediction of laminar vortex shedding over a cylinder using deep learning , 
PDF: <https://arxiv.org/pdf/1712.07854>

Prediction model of velocity field around circular cylinder over various Reynolds numbers by fusion convolutional neural networks based on pressure on the cylinder , 
PDF: <https://doi.org/10.1063/1.5024595>

Accelerating Eulerian Fluid Simulation With Convolutional Networks , 
Project+Code: <https://cims.nyu.edu/~schlacht/CNNFluids.htm>


## Additional Links for General PDEs

IDENT: Identifying Differential Equations with Numerical Time evolution , 
PDF: <https://arxiv.org/pdf/1904.03538.pdf>

Data-driven discretization: a method for systematic coarse graining of partial differential equations , 
PDF: <https://arxiv.org/pdf/1808.04930>

PDE-Net 2.0: Learning PDEs from Data with A Numeric-Symbolic Hybrid Deep Network , 
PDF: <https://arxiv.org/pdf/1812.04426>

Deep Learning the Physics of Transport Phenomena , 
PDF: <https://arxiv.org/pdf/1709.02432>

Hidden Physics Models: Machine Learning of Nonlinear Partial Differential Equations , 
PDF: <https://arxiv.org/pdf/1708.00588>

Data-assisted reduced-order modeling of extreme events in complex dynamical systems , 
Project+Code: <https://github.com/zhong1wan/data-assisted>

PDE-Net: Learning PDEs from Data , 
Project+Code: <https://github.com/ZichaoLong/PDE-Net>

Learning Deep Neural Network Representations for Koopman Operators of Nonlinear Dynamical Systems , 
PDF: <https://arxiv.org/pdf/1708.06850>


## Additional Links for Other Physics Problems and Physics-related Problems

Latent-space Dynamics for Reduced Deformable Simulation ,
Project+Code: http://www.dgp.toronto.edu/projects/latent-space-dynamics/

Learning-Based Animation of Clothing for Virtual Try-On ,
PDF: http://www.gmrv.es/Publications/2019/SOC19/

Deep Lagrangian Networks: Using Physics as Model Prior for Deep Learning , 
PDF: <https://openreview.net/pdf?id=BklHpjCqKm>

Flexible Neural Representation for Physics Prediction , 
Project+Code: <https://neuroailab.github.io/physics/>

Robust Reference Frame Extraction from Unsteady 2D Vector Fields with Convolutional Neural Networks , 
PDF: <https://arxiv.org/pdf/1903.10255.pdf>

Physics-as-Inverse-Graphics: Joint Unsupervised Learning of Objects and Physics from Video , 
PDF: <https://arxiv.org/pdf/1905.11169.pdf>

Unsupervised Intuitive Physics from Past Experiences , 
PDF: <https://arxiv.org/pdf/1905.10793.pdf>

Reasoning About Physical Interactions with Object-Oriented Prediction and Planning , 
PDF: <https://arxiv.org/pdf/1812.10972.pdf>

Neural Material: Learning Elastic Constitutive Material and Damping Models from Sparse Data , 
PDF: <https://arxiv.org/pdf/1808.04931>

Discovering physical concepts with neural networks , 
PDF: https://arxiv.org/pdf/1807.10300.pdf

Fluid directed rigid body control using deep reinforcement learning , 
Project: <http://gamma.cs.unc.edu/DRL_FluidRigid/>

DeepMimic, Example-Guided Deep Reinforcement Learning of Physics-Based Character Skills , 
PDF: <https://arxiv.org/pdf/1804.02717.pdf>

Unsupervised Intuitive Physics from Visual Observations , 
PDF: <https://arxiv.org/pdf/1805.05086>

Graph networks as learnable physics engines for inference and control , 
PDF: <https://arxiv.org/pdf/1806.01242.pdf>

DeepWarp: DNN-based Nonlinear Deformation , 
PDF: <https://arxiv.org/pdf/1803.09109>

Interaction Networks for Learning about Objects, Relations and Physics , 
PDF: <https://arxiv.org/pdf/1612.00222.pdf>



## Surveys and Overview Articles

A review on Deep Reinforcement Learning for Fluid Mechanics ,
PDF: https://arxiv.org/pdf/1908.04127.pdf

Machine Learning for Fluid Mechanics , 
PDF: https://arxiv.org/pdf/1905.11075.pdf



# Concluding Remarks

Physics-based deep learning is a very dynamic field. Please let us know if we've overlooked
papers that you think should be included by sending a mail to _i15ge at cs.tum.de_.

Also feel free to check out our homepage at <https://ge.in.tum.de/>, 
and our open source fluid simulation and deep learning framework _mantaflow_: <http://mantaflow.com>
 
