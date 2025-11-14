This repository contains code and resources for Multi-pinhole imaging simulation and reconstruction.

## Contents
- `multi_pinhole/`: Python package to simulate the multi-pinhole imaging.
- `utils/`: Utility functions for data processing and visualization.
- `docs/`: Documentation files and user guides.
- `examples/`: Example scripts demonstrating how to use the package.

## Installation
To install the package, clone the repository and run:
```bash
    git clone
    cd multi_pinhole_imaging
    pip install -e .
```

## Classes
- `World`: Represents the 3D world to be imaged.
- `Camera`: Simulates a multi-pinhole camera system.
- `Voxel`: Represents a 3D voxel grid for the imaging volume.

`Camera` class contains the following sub-classes:
- `Eye`: Models of the pinholes and their arrangement in the camera.
- `Screen`: Represents the imaging sensor where the projections are captured.
- `Aperture`: Models of the apertures to control an image size.
