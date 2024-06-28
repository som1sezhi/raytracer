#include "TracingRoutines.h"

#include <glm/glm.hpp>
#include <cuda_runtime.h>
#include "Utils.h"

__host__ __device__
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

__host__ __device__
glm::vec3 getRayColor(const Ray& ray, RenderParams& params, curandState* rndState)
{
    Ray curRay = ray;
    glm::vec3 rayColor{ 1.0f };
    for (int i = 0; i <= params.settings.bounceLimit; i++)
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