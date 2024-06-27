#include "Kernels.h"

#include "ErrorCheck.h"
#include <stdio.h>

__global__
void traceRaysKernel(RenderParams params)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if ((x >= params.width) || (y >= params.height))
        return;

    Ray ray = params.camera.GetRay(x, y);

    HitInfo closestHit;
    for (size_t i = 0; i < params.spheresCount; i++)
    {
        Sphere* sphere = params.spheres + i;
        HitInfo hit = sphere->Intersect(ray, 0.0f, closestHit.dist);

        if (hit.dist < closestHit.dist)
            closestHit = hit;
    }

    glm::vec4 color;
    if (closestHit.DidHit())
        color = glm::vec4(0.5f * closestHit.normal + 0.5f, 1.0f);
    else
        color = glm::vec4(0, 0, 0, 1);

    float4 data = make_float4(color.r, color.g, color.b, color.a);
    surf2Dwrite(data, params.surface, x * 16, y);
}

void traceRays(RenderParams& renderParams)
{
    const int tx = 8, ty = 8;
    dim3 blocks(renderParams.width / tx + 1, renderParams.height / ty + 1);
    dim3 threads(tx, ty);
    traceRaysKernel<<<blocks, threads>>>(renderParams);
    CU_CHECK(cudaGetLastError());
    CU_CHECK(cudaDeviceSynchronize());
}
