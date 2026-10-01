# Nonlinear Pipe-Network Solver Using Broyden's Method

A numerical solver for large pressurized liquid-transmission networks,
developed as a project for Advanced Numerical Methods in Chemical Engineering
at Sharif University of Technology.

## Overview

The project solves the steady-state pressures and volumetric flow rates of
nonlinear pipe networks containing source, demand, and internal nodes.

The network equations combine:

- Node continuity equations
- Pipe energy balances
- Frictional pressure losses
- Laminar and turbulent friction-factor correlations
- Elevation effects
- Pump head contributions

The resulting large nonlinear system is solved using a custom
Broyden-based numerical solver.

## Numerical Methods

### Limited-Memory Broyden Solver

A limited-memory inverse-Jacobian update is used to avoid explicitly
rebuilding and storing a dense inverse Jacobian at every iteration.

The solver also includes:

- Line search for step-size control
- Jacobian rebuilding when updates become unreliable
- Variable and residual scaling
- Custom LU factorization and linear-system solution

### Sparse Numerical Jacobian

The sparsity structure of the pipe-network equations is exploited to group
independent Jacobian columns.

Variables whose perturbations affect non-overlapping residual equations are
perturbed simultaneously, reducing the number of function evaluations
required for finite-difference Jacobian construction.

### Network Generation

Random connected pipe networks are generated with:

- Source, demand, and internal nodes
- Multiple network loops
- Random pipe diameters, lengths, roughnesses, and inclinations
- Pump placement
- Connectivity constraints

## Process Model

Unknown variables are organized as

x = [internal node pressures, pipe flow rates]

and are obtained by simultaneously satisfying mass-conservation and
energy-balance equations throughout the network.

## Example Scale

The implementation was designed to solve networks containing approximately
1000 nodes and more than 1000 pipe connections.

## Visualization

Solved networks are visualized as graphs in which node color represents
pressure, while source and demand nodes are highlighted separately.

![Solved pipe network](figures/)

## Technologies

- Python
- NumPy
- NetworkX
- Matplotlib

## Running

Install the dependencies:

```bash
pip install -r requirements.txt
