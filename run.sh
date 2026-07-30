#!/bin/bash
mkdir -p build artifacts data/output
cd build
cmake ..
make -j4
cd ..

echo "Resolution,CPU_Time_ms,GPU_Time_ms,Speedup" > artifacts/benchmark_results.csv

echo "Running CUDA Image Processor..."
./build/cuda_processor -i data/input/sample.pgm -o data/output/result.pgm -t 16 | tee artifacts/execution_log.txt

echo "Execution finished. Check artifacts/ folder."
