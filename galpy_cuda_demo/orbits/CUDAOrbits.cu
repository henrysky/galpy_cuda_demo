#include "cuda.h"
#include "stdio.h"
#include "stdlib.h"

// for cuda profiler
#include "cuda_profiler_api.h"

#define M_s 1.f // Solar mass
#define G 39.5f// Gravitational constant Solar mass, AU

// single precision CUDA function to be called on GPU
__device__ float potential_thingy(float x, float y) {
    return G * M_s * x / powf((powf(x, 2) + powf(y, 2)), 1.5f);
}

// euler method for velocity component
__global__ void euler_integration_vx(float *x_out, float *y_out, float *vx_out, int n, int steps, int current_step, float dt) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < n){
        vx_out[n*current_step+tid] = vx_out[(n*current_step-n)+tid] - potential_thingy(x_out[(n*current_step-n)+tid], y_out[(n*current_step-n)+tid]) * dt;
        tid += gridDim.x * blockDim.x;
    }
}

// euler method for position component
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
    float *dev_x_out, *dev_y_out, *dev_vx_out, *dev_vy_out;

    // streams related constants and things
    const int nStreams = 4;

    // stream for kernel
    cudaStream_t stream[nStreams];
    for (int i = 0; i < nStreams; ++i)
        cudaStreamCreate(&stream[i]);

    // allocate the memory on the GPU (VRAM)
    // cudaMalloc docs: http://horacio9573.no-ip.org/cuda/group__CUDART__MEMORY_gc63ffd93e344b939d6399199d8b12fef.html
    cudaMalloc((void**)&dev_x_out, steps * n * sizeof(float));
    cudaMalloc((void**)&dev_y_out, steps * n * sizeof(float));
    cudaMalloc((void**)&dev_vx_out, steps * n * sizeof(float));
    cudaMalloc((void**)&dev_vy_out, steps * n * sizeof(float));

    // map the arrays x, y, vx, vy to the corresponding GPU array
    // cudaMemcpy docs: http://horacio9573.no-ip.org/cuda/group__CUDART__MEMORY_g48efa06b81cc031b2aa6fdc2e9930741.html
    cudaMemcpy(&dev_x_out[0], &x[0], n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(&dev_y_out[0], &y[0], n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(&dev_vx_out[0], &vx[0], n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(&dev_vy_out[0], &vy[0], n * sizeof(float), cudaMemcpyHostToDevice);

    // loop time, because time steps cannot be paralleled
    int cstep = 1;  // keep track of the time in integration
    while (cstep < steps){
        // integrate velocity first in 2 concurrent kernel
        euler_integration_vx<<<128, 128, 0, stream[0]>>>(dev_x_out, dev_y_out, dev_vx_out, n, steps, cstep, dt);
        euler_integration_vx<<<128, 128, 0, stream[1]>>>(dev_y_out, dev_x_out, dev_vy_out, n, steps, cstep, dt);
        // as soon as any kernel finished computation, send the data back to CPU host
        cudaMemcpyAsync(&vx_out[cstep*n], &dev_vx_out[cstep*n], n * sizeof(float), cudaMemcpyDeviceToHost, stream[0]);
        cudaMemcpyAsync(&vy_out[cstep*n], &dev_vy_out[cstep*n], n * sizeof(float), cudaMemcpyDeviceToHost, stream[1]);
        // as soon as above finished, start corresponding position computation
        euler_integration_x<<<128, 128, 0, stream[2]>>>(dev_x_out, dev_vx_out, n, steps, cstep, dt);
        euler_integration_x<<<128, 128, 0, stream[3]>>>(dev_y_out, dev_vy_out, n, steps, cstep, dt);
        // as soon as any kernel finished computation, send the data back to CPU host
        cudaMemcpyAsync(&x_out[cstep*n], &dev_x_out[cstep*n], n * sizeof(float), cudaMemcpyDeviceToHost, stream[2]);
        cudaMemcpyAsync(&y_out[cstep*n], &dev_y_out[cstep*n], n * sizeof(float), cudaMemcpyDeviceToHost, stream[3]);
        // make sure above all finished to start next time step because next time step depends on this step
        cudaDeviceSynchronize();
        cstep += 1;
    }

    // free the memory allocated on the GPU after integration, if really galpy, need to take care memory for real
    cudaFree(dev_x_out);
    cudaFree(dev_y_out);
    cudaFree(dev_vx_out);
    cudaFree(dev_vy_out);

    return 0;
}