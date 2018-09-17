#include "cuda.h"
#include "stdio.h"
#include "stdlib.h"

// for cuda profiler
#include "cuda_profiler_api.h"

#define M_s 1. // Solar mass
#define G 39.5 // Gravitational constant Solar mass, AU

// single precision CUDA function to be called on GPU
__device__ float potential_thingy(float x, float y) {
    return G * M_s * x / powf((powf(x, 2) + powf(y, 2)), 3/2);
}

__global__ void set_initial_cond(float *x, float *y, float *vx, float *vy, float *x_out, float *y_out, float *vx_out,
                                    float *vy_out, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < n){
        x_out[tid] = x[tid];
        y_out[tid] = y[tid];
        vx_out[tid] = vx[tid];
        vy_out[tid] = vy[tid];
        tid += gridDim.x * blockDim.x;
    }
}

__global__ void euler_integration_vx(float *x_out, float *y_out, float *vx_out, int n, int steps, int current_step, float dt) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < n){
        vx_out[n*current_step+tid] = vx_out[(n*current_step-n)+tid] - potential_thingy(x_out[(n*current_step-n)+tid], y_out[(n*current_step-n)+tid]) * dt;
        tid += gridDim.x * blockDim.x;
    }
}

__global__ void euler_integration_x(float *x_out, float *vx_out, int n, int steps, int current_step, float dt) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < n){
        x_out[n*current_step+tid] = x_out[(n*current_step-n)+tid] + vx_out[n*current_step+tid] * dt;
        tid += gridDim.x * blockDim.x;
    }
}

extern "C" int integrate_euler_cuda(float *x, float *y, float *vx, float *vy, float *x_out, float *y_out, float *vx_out,
                                    float *vy_out, int n, int steps, float dt) {
    // dev_** variables for variables on CUDA device
    float *dev_x, *dev_y, *dev_vx, *dev_vy, *dev_x_out, *dev_y_out, *dev_vx_out, *dev_vy_out;

    // streams related constants and things
    const int nStreams = 2;
    const int streamSize = n / nStreams;

    // stream for kernel
    cudaStream_t stream[nStreams];
    for (int i = 0; i < nStreams; ++i)
        cudaStreamCreate(&stream[i]);

    // allocate the memory on the GPU (VRAM)
    // cudaMalloc docs: http://horacio9573.no-ip.org/cuda/group__CUDART__MEMORY_gc63ffd93e344b939d6399199d8b12fef.html
    cudaMalloc((void**)&dev_x, n * sizeof(float));
    cudaMalloc((void**)&dev_y, n * sizeof(float));
    cudaMalloc((void**)&dev_vx, n * sizeof(float));
    cudaMalloc((void**)&dev_vy, n*  sizeof(float));
    cudaMalloc((void**)&dev_x_out, steps * n * sizeof(float));
    cudaMalloc((void**)&dev_y_out, steps * n * sizeof(float));
    cudaMalloc((void**)&dev_vx_out, steps * n * sizeof(float));
    cudaMalloc((void**)&dev_vy_out, steps * n * sizeof(float));

    // copy the arrays x, y, vx, vy to the GPU
    // cudaMemcpy docs: http://horacio9573.no-ip.org/cuda/group__CUDART__MEMORY_g48efa06b81cc031b2aa6fdc2e9930741.html
    cudaMemcpy(dev_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_y, y, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_vx, vx, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_vy, vy, n * sizeof(float), cudaMemcpyHostToDevice);

    // CUDA kernel configuration: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-model
    // set initial condition for Euler's method
    set_initial_cond<<<128, 128>>>(dev_x, dev_y, dev_vx, dev_vy, dev_x_out, dev_y_out, dev_vx_out, dev_vy_out,
                                     n);
    // free memory
    cudaFree(dev_x);
    cudaFree(dev_y);
    cudaFree(dev_vx);
    cudaFree(dev_vy);

    // loop time, because time steps cannot be paralleled
    int cstep = 1;  // keep track of the time in integration
    while (cstep < steps){
        euler_integration_vx<<<128, 128, 0, stream[0]>>>(dev_x_out, dev_y_out, dev_vx_out, n, steps, cstep, dt);
        euler_integration_vx<<<128, 128, 0, stream[1]>>>(dev_y_out, dev_x_out, dev_vy_out, n, steps, cstep, dt);\
        euler_integration_x<<<128, 128, 0, stream[0]>>>(dev_x_out, dev_vx_out, n, steps, cstep, dt);
        euler_integration_x<<<128, 128, 0, stream[1]>>>(dev_y_out, dev_vy_out, n, steps, cstep, dt);
        cudaDeviceSynchronize();  // need to ensure this loop finished before the next one
        cstep += 1;
    }

    // copy arrays 'data back from the GPU to the CPU to be ehandled by the Python interface
    cudaMemcpy(&x_out, &dev_x_out, steps * n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(y_out, dev_y_out, steps * n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(vx_out, dev_vx_out, steps * n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(vy_out, dev_vy_out, steps * n * sizeof(float), cudaMemcpyDeviceToHost);

    // free the memory allocated on the GPU after integration, if really galpy, need to take care memory for real
    cudaFree(dev_x_out);
    cudaFree(dev_y_out);
    cudaFree(dev_vx_out);
    cudaFree(dev_vy_out);

    return 0;  // return None basically like galpy??
}