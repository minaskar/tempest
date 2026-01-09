# Installation

This guide covers different methods to install Tempest.

## Requirements

Tempest requires Python 3.8 or later and depends on the following packages:

- `numpy >= 1.20.0`
- `tqdm >= 4.60.0`
- `scipy >= 1.4.0`
- `dill >= 0.3.8`
- `multiprocess >= 0.70.15`

---

## Using pip (Recommended)

The simplest way to install Tempest is via pip:

```bash
pip install tempest
```

This will install Tempest and all its dependencies.

---

## Using conda

You can also install Tempest using conda by first creating a conda environment and then using pip:

```bash
conda create -n tempest-env python=3.11
conda activate tempest-env
pip install tempest
```

---

## From Source

To install the latest development version from source:

```bash
git clone https://github.com/minaskar/tempest.git
cd tempest
pip install -e .
```

The `-e` flag installs the package in "editable" mode, which is useful for development.

---

## Optional Dependencies

### MPI Support

For running Tempest with MPI parallelization (useful for HPC clusters), install `mpi4py`:

```bash
pip install mpi4py
```

!!! note "MPI Installation"
    Make sure you have an MPI implementation installed on your system (e.g., OpenMPI, MPICH) before installing mpi4py.

### Development Dependencies

For development and testing:

```bash
pip install tempest[dev]
```

Or install testing dependencies manually:

```bash
pip install pytest pytest-cov
```

---

## Verifying Installation

To verify that Tempest is installed correctly, open a Python shell and run:

```python
import tempest as pc
print(pc.__version__)
```

This should print the version number of Tempest.

You can also run the test suite:

```bash
pytest tests/
```

---

## Troubleshooting

### PyTorch Issues

If you encounter issues with PyTorch, try installing it separately first:

```bash
pip install torch
```

For GPU support, follow the [PyTorch installation guide](https://pytorch.org/get-started/locally/).

### Permission Errors

If you get permission errors during installation, try:

```bash
pip install --user tempest
```

Or use a virtual environment (recommended).
