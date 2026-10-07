
# Installation Guide

**Optical Networking Gym** is an open-source toolkit for benchmarking resource assignment problems in optical networks. This guide provides step-by-step instructions to install and set up the necessary environment for developing and running simulations with Optical Networking Gym.

## Table of Contents

1. [Introduction](#introduction)
2. [Prerequisites](#prerequisites)
3. [Setting Up a Virtual Environment](#setting-up-a-virtual-environment)
4. [Installing the Project](#installing-the-project)
5. [Installing Development Dependencies](#installing-development-dependencies)
6. [Installing Research Dependencies](#installing-research-dependencies)
7. [Building the Package and Running Tests](#building-the-package-and-running-tests)
8. [Configuring the Development Environment](#configuring-the-development-environment)
9. [Memory Use of Long Simulations](#memory-use-of-long-simulations)
10. [Additional Resources](#additional-resources)
11. [License and Credits](#license-and-credits)

---

## 1. Introduction

Welcome to **Optical Networking Gym**! This toolkit is designed to facilitate the benchmarking of resource assignment problems in optical networks, providing researchers and developers with the tools needed to develop and evaluate their solutions efficiently.

## 2. Prerequisites

Before proceeding with the installation, ensure that your system meets the following prerequisites:

- **Python 3.8 or higher**: [Download Python](https://www.python.org/downloads/)
- **Git**: [Download Git](https://git-scm.com/downloads)
- **pip**: Python's package installer (usually included with Python installation)

To verify that these are installed, run the following commands in your terminal or command prompt:

```bash
python --version
pip --version
git --version
```

## 3. Setting Up a Virtual Environment

It is recommended to create a virtual environment to manage the project's dependencies without affecting your global Python installation. Use Python's built-in `venv` module:

```bash
python -m venv .venv
```

### Activating the Virtual Environment

- **Linux/macOS**:

  ```bash
  source .venv/bin/activate
  ```

- **Windows**:

  ```bash
  .venv\Scripts\activate
  ```

After activation, your terminal prompt should indicate that you are working within the `.venv` environment.

## 4. Installing the Project

With the virtual environment activated, install the project using `pip`:

```bash
pip install -e .
```

This command installs the project in editable mode, allowing you to make changes to the code without reinstalling the package.

### Installing with Development and Research Dependencies

To include additional dependencies required for development and research activities, use:

```bash
pip install -e ".[dev,research]"
```

## 5. Installing Development Dependencies

If you have already installed the project, you can add development-specific dependencies with:

```bash
pip install -e ".[dev]"
```

## 6. Installing Research Dependencies

To add dependencies necessary for research purposes, execute:

```bash
pip install -e ".[research]"
```

## 7. Building the Package and Running Tests

To build the package and run tests with coverage reporting, use the following command:

```bash
DEBUG=1 python setup.py clean --all build_ext --force --inplace && coverage run -m pytest && coverage report
```

### Breakdown of Commands:

- `DEBUG=1`: Enables debug mode.
- `python setup.py clean --all`: Cleans previous builds.
- `build_ext --force --inplace`: Compiles Cython extensions.
- `coverage run -m pytest`: Executes tests with coverage measurement.
- `coverage report`: Generates a coverage report.

**Note for Windows Users**: The `DEBUG=1` environment variable syntax may differ. You can set it using `set DEBUG=1` before running the command or adjust accordingly.

## 8. Configuring the Development Environment

We recommend using **Visual Studio Code (VSCode)** for development. To enhance support for Cython (`.pyx`) files, install the [Cython Extension](https://marketplace.visualstudio.com/items?itemName=ktnrg45.vscode-cython) (`ktnrg45.vscode-cython`).

### Steps to Configure VSCode:

1. **Install VSCode**: [Download VSCode](https://code.visualstudio.com/download)
2. **Open the Project in VSCode**:

   ```bash
   code .
   ```

3. **Install the Cython Extension**:
   - Open the command palette (`Ctrl+Shift+P` or `Cmd+Shift+P`).
   - Type `Extensions: Install Extensions` and press Enter.
   - Search for `Cython` and install the `ktnrg45.vscode-cython` extension.

## 9. Memory Use of Long Simulations

The request-analysis engine memoizes the analysis of each request (candidate
paths, formats and start slots, with their QoT). An analysis holds several
`(k_paths, modulations, slots)` arrays, about 0.2–0.3 MB on nobel-eu with 320
slots, and its cache key includes the allocation state, which changes with
almost every arrival. In a normal step loop the cache is therefore never hit,
and an unbounded cache grows by one entry per request (about 9 GB after
120,000 arrivals).

`ScenarioConfig.request_buffer_limit` bounds the cache (least recently used
entries are evicted first):

| Value | Effect |
|---|---|
| `8` (default) | At most 8 analyses in memory, a few MB. |
| `-1` | Unlimited, the behaviour of 0.3.0 and earlier. |
| `0` | No caching: every analysis is built anew. |

The cache only memoizes, so results do not depend on this value. Set it with
`build_scenario(name, request_buffer_limit=...)`,
`make_env(..., request_buffer_limit=...)` or directly in `ScenarioConfig`. The
engine reports `cache_hits`, `cache_misses`, `cache_evictions` and
`cache_size`.

The opt-in capture buffers are **not** bounded, because they are exports and
truncating them would silently lose data: `capture_traffic_table` keeps every
request, and `capture_step_trace` keeps every step with three full network
snapshots. Their memory grows with every arrival, so enable them only for runs
whose trace you will export.

## 10. Additional Resources

After completing the installation steps, you're ready to develop and run simulations using **Optical Networking Gym**. For more information and resources, refer to:

- **GitHub Repository**: [GitHub](https://github.com/carlosnatalino/optical-networking-gym)
- **Issues and Support**: Use the [Issues](https://github.com/carlosnatalino/optical-networking-gym/issues) section on GitHub to report problems or request new features.


## 11. License and Credits

**Optical Networking Gym** is licensed under the [MIT License](https://github.com/carlosnatalino/optical-networking-gym/blob/main/LICENSE). We extend our gratitude to all contributors and the open-source community for their support and contributions.

---
