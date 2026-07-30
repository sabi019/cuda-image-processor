#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <cmath>
#include <string>
#include <getopt.h>

// CUDA Error Checking Utility
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// CUDA Kernel: Sobel Filter for Edge Detection
__global__ void SobelFilterKernel(const unsigned char* input, unsigned char* output, 
                                  int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        int Gx = (-1 * input[(y - 1) * width + (x - 1)]) + (1 * input[(y - 1) * width + (x + 1)]) +
                 (-2 * input[y * width + (x - 1)])       + (2 * input[y * width + (x + 1)]) +
                 (-1 * input[(y + 1) * width + (x - 1)]) + (1 * input[(y + 1) * width + (x + 1)]);

        int Gy = (-1 * input[(y - 1) * width + (x - 1)]) + (-2 * input[(y - 1) * width + x]) + (-1 * input[(y - 1) * width + (x + 1)]) +
                 ( 1 * input[(y + 1) * width + (x - 1)]) + ( 2 * input[(y + 1) * width + x]) + ( 1 * input[(y + 1) * width + (x + 1)]);

        int val = sqrtf((float)(Gx * Gx + Gy * Gy));
        output[y * width + x] = (val > 255) ? 255 : val;
    }
}

// CPU Reference Implementation for Benchmark
void CpuSobelFilter(const std::vector<unsigned char>& input, std::vector<unsigned char>& output, 
                    int width, int height) {
    for (int y = 1; y < height - 1; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            int Gx = (-1 * input[(y - 1) * width + (x - 1)]) + (1 * input[(y - 1) * width + (x + 1)]) +
                     (-2 * input[y * width + (x - 1)])       + (2 * input[y * width + (x + 1)]) +
                     (-1 * input[(y + 1) * width + (x - 1)]) + (1 * input[(y + 1) * width + (x + 1)]);

            int Gy = (-1 * input[(y - 1) * width + (x - 1)]) + (-2 * input[(y - 1) * width + x]) + (-1 * input[(y - 1) * width + (x + 1)]) +
                     ( 1 * input[(y + 1) * width + (x - 1)]) + ( 2 * input[(y + 1) * width + x]) + ( 1 * input[(y + 1) * width + (x + 1)]);

            int val = std::sqrt(Gx * Gx + Gy * Gy);
            output[y * width + x] = (val > 255) ? 255 : val;
        }
    }
}

// Helper: Read PGM Image
bool ReadPGM(const std::string& filename, std::vector<unsigned char>& data, int& width, int& height) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) return false;

    std::string format;
    int max_val;
    file >> format >> width >> height >> max_val;
    file.ignore(256, '\n');

    if (format != "P5") return false;

    data.resize(width * height);
    file.read(reinterpret_cast<char*>(data.data()), data.size());
    return true;
}

// Helper: Write PGM Image
bool WritePGM(const std::string& filename, const std::vector<unsigned char>& data, int width, int height) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) return false;

    file << "P5\n" << width << " " << height << "\n255\n";
    file.write(reinterpret_cast<const char*>(data.data()), data.size());
    return true;
}

int main(int argc, char* argv[]) {
    std::string input_path = "";
    std::string output_path = "";
    int threads_per_block = 16;

    int opt;
    while ((opt = getopt(argc, argv, "i:o:t:")) != -1) {
        switch (opt) {
            case 'i': input_path = optarg; break;
            case 'o': output_path = optarg; break;
            case 't': threads_per_block = std::atoi(optarg); break;
            default:
                std::cerr << "Usage: " << argv[0] << " -i <input_pgm> -o <output_pgm> [-t <threads_per_block>]\n";
                return EXIT_FAILURE;
        }
    }

    if (input_path.empty() || output_path.empty()) {
        std::cerr << "Error: Input and Output file paths are required.\n";
        return EXIT_FAILURE;
    }

    int width = 0, height = 0;
    std::vector<unsigned char> h_input;
    if (!ReadPGM(input_path, h_input, width, height)) {
        std::cerr << "Failed to read input image: " << input_path << "\n";
        return EXIT_FAILURE;
    }

    std::vector<unsigned char> h_output_gpu(width * height, 0);
    std::vector<unsigned char> h_output_cpu(width * height, 0);

    // CPU Execution
    auto start_cpu = std::chrono::high_resolution_clock::now();
    CpuSobelFilter(h_input, h_output_cpu, width, height);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double, std::milli>(end_cpu - start_cpu).count();

    // GPU Memory Allocation
    unsigned char *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, width * height * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_output, width * height * sizeof(unsigned char)));

    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), width * height * sizeof(unsigned char), cudaMemcpyHostToDevice));

    dim3 threads(threads_per_block, threads_per_block);
    dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);

    // GPU Execution
    cudaEvent_t start_gpu, stop_gpu;
    CUDA_CHECK(cudaEventCreate(&start_gpu));
    CUDA_CHECK(cudaEventCreate(&stop_gpu));

    CUDA_CHECK(cudaEventRecord(start_gpu));
    SobelFilterKernel<<<blocks, threads>>>(d_input, d_output, width, height);
    CUDA_CHECK(cudaEventRecord(stop_gpu));
    CUDA_CHECK(cudaEventSynchronize(stop_gpu));

    float gpu_time = 0;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_time, start_gpu, stop_gpu));

    CUDA_CHECK(cudaMemcpy(h_output_gpu.data(), d_output, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost));

    WritePGM(output_path, h_output_gpu, width, height);

    std::cout << "--- Performance Results ---\n";
    std::cout << "Image Size: " << width << "x" << height << "\n";
    std::cout << "CPU Time: " << cpu_time << " ms\n";
    std::cout << "GPU Kernel Time: " << gpu_time << " ms\n";
    std::cout << "Speedup: " << cpu_time / gpu_time << "x\n";

    // Write to benchmark log
    std::ofstream csv_file("artifacts/benchmark_results.csv", std::ios::app);
    csv_file << width << "x" << height << "," << cpu_time << "," << gpu_time << "," << (cpu_time / gpu_time) << "\n";

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return 0;
}
