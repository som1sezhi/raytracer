#include "TracingRoutines.h"

#include <glm/glm.hpp>
#include <cuda_runtime.h>
#include "Ray.h"
#include "HitInfo.h"
#include "Material.h"
#include "Scene.h"

__host__ __device__
HitInfo hitScene(const Ray& ray, RenderParams& params)
{
    HitInfo closestHit;
    for (size_t i = 0; i < params.spheresCount; i++)
    {
        Sphere* sphere = params.spheres + i;
        HitInfo hit = sphere->Intersect(ray, 1e-8f, closestHit.dist);

        if (hit.dist < closestHit.dist)
            closestHit = hit;
    }
    return closestHit;
}

__host__ __device__
glm::vec3 getRayColor(const Ray& ray, RenderParams& params, uint32_t& seed)
{
    Ray curRay = ray;
    glm::vec3 lightAbsorption{ 1.0f };
    glm::vec3 lightCollected{ 0.0f };
    for (int i = 0; i <= params.settings.bounceLimit; i++)
    {
        HitInfo hit = hitScene(curRay, params);
        if (hit.DidHit())
        {
            glm::vec3 absorption, emission;
            if (!hit.material->ScatterRay(curRay, hit, absorption, emission, curRay, seed))
                break;
            lightCollected += emission * lightAbsorption;
            lightAbsorption *= absorption;
            // Nudge the ray origin a little off the surface to prevent
            // shadow acne
            curRay.origin += curRay.dir * 1e-4f;
        }
        else
        {
            float a = 0.5f * (curRay.dir.y + 1.0f);
            //const glm::vec3 lightDir = glm::normalize(glm::vec3(1.0f));
            //float a = glm::dot(curRay.dir, lightDir);
            //a = 0.5f * a + 0.5f;
            //a *= a;
            auto sky = a * params.settings.skyColor1 + (1.0f - a) * params.settings.skyColor2;
            return lightCollected + sky * lightAbsorption;
        }
    }

    return lightCollected;
}