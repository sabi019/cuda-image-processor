# High-Performance CUDA Image Processing Pipeline

## Overview
This project implements a high-performance image processing pipeline using CUDA C++. It focuses on spatial domain filtering (Sobel Edge Detection) to demonstrate parallel execution speedups over traditional single-threaded CPU processing.

## Features
- **Custom CUDA Kernels**: Optimized 2D grid/block thread layout for image processing.
- **CLI Support**: Command-line interface accepting parameters for input/output files and thread configuration.
- **Benchmarking Artifacts**: Automated CPU vs GPU timing comparisons and log generation.
- **Google C++ Style**: Clean, documented code following industry standards.

## Requirements
- NVIDIA GPU with CUDA Toolkit 11.0+
- CMake 3.18+
- GCC / G++ compiler

## How to Build and Run
Execute the provided script to build and run the processing pipeline automatically:
```bash
chmod +x run.sh
./run.sh
