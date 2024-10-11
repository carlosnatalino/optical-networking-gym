# Optical Networking Gym (ONG)

## Overview

The **Optical Networking Gym (ONG)** is a comprehensive toolkit for simulating and optimizing resource allocation in Elastic Optical Networks (EONs). It provides flexible environments for simulating real-world optical networks and supports both traditional and reinforcement learning (RL) algorithms.

ONG is built to simplify the development and benchmarking of algorithms that solve complex network optimization problems. It includes:

- **Dynamic Service Provisioning**: Simulates real-time service requests and releases in optical networks.
- **Quality of Transmission (QoT) Awareness**: Evaluates the Optical Signal-to-Noise Ratio (OSNR) for each service.
- **Routing, Modulation, and Spectrum Assignment (RMSA)**: Implements RMSA algorithms for efficiently allocating network resources.
- **Reinforcement Learning Integration**: Compatible with the OpenAI Gym interface, making it easy to train and benchmark RL algorithms.
- **Cython Optimization**: Performance-enhanced with Cython, making it ideal for large-scale simulations.

## Key Features

- **Service Provisioning**: Dynamically allocate spectrum resources based on service requests, routing paths, and modulation formats.
- **Network QoT Calculation**: Compute OSNR, ASE noise, and nonlinear impairments for each service.
- **Support for RL Algorithms**: Allows for the development and benchmarking of RL algorithms using real-world network scenarios.
- **Customizable Network Topologies**: Define your own network topology using the NetworkX library.
- **Modular Design**: Add custom environments, algorithms, and metrics to the toolkit.

## Getting Started

To get started with **Optical Networking Gym**, follow the [installation instructions](get_started.md) or visit the [GitHub repository](https://github.com/carlosnatalino/optical-networking-gym).

Explore the documentation to learn more about the environments, algorithms, and features of ONG.
