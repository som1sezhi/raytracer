#include "Kernels.h"

#include "ErrorCheck.h"
#include <stdio.h>

__global__
void traceRaysKernel(RenderParams params)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if ((x >= params.Width) || (y >= params.Height))
        return;

    glm::vec3 rayDir = params.Camera.GetRay(x, y).Dir;
    rayDir = glm::max({ 0, 0, 0 }, rayDir) * 255.99f;

    uint8_t r = (uint8_t)(rayDir.r);
    uint8_t g = (uint8_t)(rayDir.g);
    uint8_t b = (uint8_t)(rayDir.b);
    uchar4 data = make_uchar4(r, g, b, 255);
    surf2Dwrite(data, params.Surface, x * 4, y);
}

void traceRays(RenderParams& renderParams)
{
    const int tx = 8, ty = 8;
    dim3 blocks(renderParams.Width / tx + 1, renderParams.Height / ty + 1);
    dim3 threads(tx, ty);
    traceRaysKernel<<<blocks, threads>>>(renderParams);
    CU_CHECK(cudaGetLastError());
    CU_CHECK(cudaDeviceSynchronize());
}
