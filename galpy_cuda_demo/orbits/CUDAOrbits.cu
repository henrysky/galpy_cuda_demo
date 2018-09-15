#include "cuda.h"
#include "stdio.h"
#include "stdlib.h"

// single precision CUDA function to be called on GPU
__device__ float potential_thingy(float x, float y) {
    float M_s = 1.; // Solar mass
    float G = 39.5; // Gravitational constant Solar mass, AU
    return G * M_s * x * powf((powf(x, 2) + powf(y, 2)), 3/2);
}

__global__ void euler_integration(float *x, float *y, float *vx, float *vy, float *x_out, float *y_out, float *vx_out,
                                    float *vy_out, int n, int steps, float dt) {
    int current_step = 1; // to store current time

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < n){
        vx_out[tid] = vx[tid];
        vy_out[tid] = vy[tid];
        x_out[tid] = x[tid];
        y_out[tid] = y[tid];
        tid += gridDim.x * blockDim.x;
    }

    while (current_step++ < steps){
        int tid = threadIdx.x + blockIdx.x * blockDim.x;
        while (tid < n){
            // still need some work on array slicing, I miss numpy :(
            vx_out[current_step*tid] = vx_out[(current_step-1)*tid] - potential_thingy(x_out[(current_step-1)*tid], y_out[(current_step-1)*tid]) * dt;
            vy_out[current_step*tid] = vy_out[(current_step-1)*tid] - potential_thingy(y_out[(current_step-1)*tid], x_out[(current_step-1)*tid]) * dt;
            x_out[current_step*tid] = x_out[(current_step-1)*tid] + vx_out[current_step*tid] * dt;
            y_out[current_step*tid] = y_out[(current_step-1)*tid] + vy_out[current_step*tid] * dt;
            tid += gridDim.x * blockDim.x;
        }
    }
}

extern "C" int integrate_euler_cuda(float *x, float *y, float *vx, float *vy, float *x_out, float *y_out, float *vx_out,
                                    float *vy_out, int n, int steps, float dt) {
    // dev_** variables mean variables on CUDA device
    float *dev_x, *dev_y, *dev_vx, *dev_vy, *dev_x_out, *dev_y_out, *dev_vx_out, *dev_vy_out;

    // allocate the memory on the GPU (VRAM)
    // cudaMalloc docs: http://horacio9573.no-ip.org/cuda/group__CUDART__MEMORY_gc63ffd93e344b939d6399199d8b12fef.html
    cudaMalloc((void**)&dev_x, n*sizeof(float));
    cudaMalloc((void**)&dev_y, n*sizeof(float));
    cudaMalloc((void**)&dev_vx, n*sizeof(float));
    cudaMalloc((void**)&dev_vy, n*sizeof(float));
    cudaMalloc((void**)&dev_x_out, steps*n*sizeof(float));
    cudaMalloc((void**)&dev_y_out, steps*n*sizeof(float));
    cudaMalloc((void**)&dev_vx_out, steps*n*sizeof(float));
    cudaMalloc((void**)&dev_vy_out, steps*n*sizeof(float));

    // copy the arrays x, y, vx, vy to the GPU
    // cudaMemcpy docs: http://horacio9573.no-ip.org/cuda/group__CUDART__MEMORY_g48efa06b81cc031b2aa6fdc2e9930741.html
    cudaMemcpy(dev_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_y, y, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_vx, vx, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_vy, vy, n * sizeof(float), cudaMemcpyHostToDevice);

    // CUDA Kkernel configuration: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-model
    euler_integration<<<4096, 4096>>>(dev_x, dev_y, dev_vx, dev_vy, dev_x_out, dev_y_out, dev_vx_out, dev_vy_out,
                                      n, steps, dt);

    // copy arrays 'data back from the GPU to the CPU to be handled by the Python interface
    cudaMemcpy(x_out, dev_x_out, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(y_out, dev_y_out, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(vx_out, dev_vx_out, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(vy_out, dev_vy_out, n * sizeof(float), cudaMemcpyDeviceToHost);

    // free the memory allocated on the GPU after integration, if really galpy, need to take care memory for real
    cudaFree(dev_x);
    cudaFree(dev_y);
    cudaFree(dev_vx);
    cudaFree(dev_vy);
    cudaFree(dev_x_out);
    cudaFree(dev_y_out);
    cudaFree(dev_vx_out);
    cudaFree(dev_vy_out);

    return 0;  // return None basically like galpy??
}