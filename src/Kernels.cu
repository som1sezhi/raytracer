#include "Kernels.h"

#include "ErrorCheck.h"
#include <stdio.h>
#include "Utils.h"

__global__
void renderInitKernel(curandState* states, int width, int height)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if ((x >= width) || (y >= height))
        return;
    int i = x + y * width;
    curand_init(42 + i, 0, 0, &states[i]);
}

void renderInit(curandState* states, uint32_t width, uint32_t height)
{
    const int tx = 8, ty = 8;
    dim3 blocks(width / tx + 1, height / ty + 1);
    dim3 threads(tx, ty);
    renderInitKernel<<<blocks, threads >>>(states, width, height);
    CU_CHECK(cudaGetLastError());
    CU_CHECK(cudaDeviceSynchronize());
}

__global__
void renderKernel(RenderKernelParams params)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if ((x >= params.width) || (y >= params.height))
        return;

    int idx = x + params.width * y;
    curandState* state = params.randStates + idx;
    Ray ray = params.renderParams.camera.GetRay(x, y);

    glm::vec3 color = getRayColor(ray, params.renderParams, state);

    // Get previous pixel color
    float4 oldData;
    surf2Dread(&oldData, params.surface, x * 16, y);
    glm::vec3 old{ oldData.x, oldData.y, oldData.z };

    //  Accumulate color
    color = ((float)params.curNumSamples * old + color)
        / ((float)params.curNumSamples + 1);

    // Write new color to surface
    float4 data = make_float4(color.r, color.g, color.b, 1.0f);
    surf2Dwrite(data, params.surface, x * 16, y);
}

void render(RenderKernelParams& kernelParams)
{
    const int tx = 8, ty = 8;
    dim3 blocks(kernelParams.width / tx + 1, kernelParams.height / ty + 1);
    dim3 threads(tx, ty);
    renderKernel<<<blocks, threads>>>(kernelParams);
    CU_CHECK(cudaGetLastError());
    CU_CHECK(cudaDeviceSynchronize());
}
