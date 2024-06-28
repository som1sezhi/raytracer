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


__device__
HitInfo hitScene(const Ray& ray, RenderParams& params)
{
    HitInfo closestHit;
    for (size_t i = 0; i < params.spheresCount; i++)
    {
        Sphere* sphere = params.spheres + i;
        HitInfo hit = sphere->Intersect(ray, 0.001f, closestHit.dist);

        if (hit.dist < closestHit.dist)
            closestHit = hit;
    }
    return closestHit;
}

__device__
glm::vec3 getRayColor(const Ray& ray, RenderParams& params, curandState* rndState)
{
    Ray curRay = ray;
    glm::vec3 rayColor{ 1.0f };
    for (int i = 0; i < 40; i++)
    {
        HitInfo hit = hitScene(curRay, params);
        if (hit.DidHit())
        {
            glm::vec3 dir = hit.normal + randomUnitVec(rndState);
            dir = glm::normalize(dir);
            curRay = { hit.position + 0.001f * hit.normal, dir };
            rayColor *= 0.5f;
        }
        else
        {
            float a = 0.5f * (curRay.dir.y + 1.0f);
            auto c = (1.0f - a) * glm::vec3(1.0f) + a * glm::vec3(0.5f, 0.7f, 1.0f);
            return rayColor * c;
        }
    }
    
    return glm::vec3(0.0f);
}

__global__
void renderKernel(RenderParams params)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if ((x >= params.width) || (y >= params.height))
        return;
    /*
    Ray ray = params.camera.GetRay(x, y);

    HitInfo closestHit;
    for (size_t i = 0; i < params.spheresCount; i++)
    {
        Sphere* sphere = params.spheres + i;
        HitInfo hit = sphere->Intersect(ray, 0.0f, closestHit.dist);

        if (hit.dist < closestHit.dist)
            closestHit = hit;
    }*/

    //if (closestHit.DidHit())
    //    color = glm::vec4(0.5f * closestHit.normal + 0.5f, 1.0f);
    //else
    //    color = glm::vec4(0, 0, 0, 1);

    int idx = x + params.width * y;
    curandState* state = params.randStates + idx;

    Ray ray = params.camera.GetRay(x, y);
    glm::vec3 color = getRayColor(ray, params, state);

    float4 oldData;
    surf2Dread(&oldData, params.surface, x * 16, y);
    glm::vec3 old{ oldData.x, oldData.y, oldData.z };

    color = ((float)params.curNumSamples * old + color)
        / ((float)params.curNumSamples + 1);

    float4 data = make_float4(color.r, color.g, color.b, 1.0f);
    surf2Dwrite(data, params.surface, x * 16, y);
}

void render(RenderParams& renderParams)
{
    const int tx = 8, ty = 8;
    dim3 blocks(renderParams.width / tx + 1, renderParams.height / ty + 1);
    dim3 threads(tx, ty);
    renderKernel<<<blocks, threads>>>(renderParams);
    CU_CHECK(cudaGetLastError());
    CU_CHECK(cudaDeviceSynchronize());
}
