#include "cuda.h"
#include "stdio.h"
#include "stdlib.h"

// single precision CUDA function to be called on GPU
__device__ float potential_thingy(float x, float y) {
    float M_s = 1.; // Solar mass
    float G = 39.5; // Gravitational constant Solar mass, AU
    return G * M_s * x / powf((powf(x, 2) + powf(y, 2)), 3/2);
}

__global__ void set_initial_cond(float *x, float *y, float *vx, float *vy, float *x_out, float *y_out, float *vx_out,
                                    float *vy_out, int n, int steps, float dt) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < n){
        x_out[steps*tid] = x[tid];
        y_out[steps*tid] = y[tid];
        vx_out[steps*tid] = vx[tid];
        vy_out[steps*tid] = vy[tid];
        tid += gridDim.x * blockDim.x;
    }
}

__global__ void euler_integration(float *x, float *y, float *vx, float *vy, float *x_out, float *y_out, float *vx_out,
                                    float *vy_out, int n, int steps, int current_step, float dt) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < n){
        vx_out[current_step+steps*tid] = vx_out[(current_step-1)+steps*tid] - potential_thingy(x_out[(current_step-1)+steps*tid], y_out[(current_step-1)+steps*tid]) * dt;
        vy_out[current_step+steps*tid] = vy_out[(current_step-1)+steps*tid] - potential_thingy(y_out[(current_step-1)+steps*tid], x_out[(current_step-1)+steps*tid]) * dt;
        x_out[current_step+steps*tid] = x_out[(current_step-1)+steps*tid] + vx_out[current_step+steps*tid] * dt;
        y_out[current_step+steps*tid] = y_out[(current_step-1)+steps*tid] + vy_out[current_step+steps*tid] * dt;
        tid += gridDim.x * blockDim.x;
    }
}

extern "C" int integrate_euler_cuda(float *x, float *y, float *vx, float *vy, float *x_out, float *y_out, float *vx_out,
                                    float *vy_out, int n, int steps, float dt) {
    // dev_** variables mean variables on CUDA device
    float *dev_x, *dev_y, *dev_vx, *dev_vy, *dev_x_out, *dev_y_out, *dev_vx_out, *dev_vy_out;

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

    // CUDA Kkernel configuration: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-model
    set_initial_cond<<<1024, 1024>>>(dev_x, dev_y, dev_vx, dev_vy, dev_x_out, dev_y_out, dev_vx_out, dev_vy_out,
                                     n, steps, dt);

    // loop time, because time steps cannot be parallelized
    int cstep = 1;  // keep track of the time in integration
    while (cstep < steps){
        euler_integration<<<1024, 1024>>>(dev_x, dev_y, dev_vx, dev_vy, dev_x_out, dev_y_out, dev_vx_out, dev_vy_out,
                                          n, steps, cstep, dt);
        cstep += 1;
    }

    // copy arrays 'data back from the GPU to the CPU to be handled by the Python interface
    cudaMemcpy(x_out, dev_x_out, steps * n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(y_out, dev_y_out, steps * n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(vx_out, dev_vx_out, steps * n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(vy_out, dev_vy_out, steps * n * sizeof(float), cudaMemcpyDeviceToHost);

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