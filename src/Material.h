#pragma once

#include <glm/glm.hpp>
#include <cuda_runtime.h>

#include "Utils.h"
#include "HitInfo.h"
#include "Ray.h"

struct Material
{
    glm::vec3 color;
    glm::vec3 emissionColor{ 1.0f };
    float emissionStrength = 0.0f;

    __host__ __device__
    bool ScatterRay(
        const Ray& rayIn,
        const HitInfo& hit,
        glm::vec3& absorption,
        glm::vec3& light,
        Ray& rayOut,
        uint32_t& seed
    ) const
    {
        glm::vec3 dir = hit.normal + randomUnitVec(seed);
        // Catch degenerate scatter direction
        if (nearZero(dir))
            dir = hit.normal;
        else
            dir = glm::normalize(dir);
        rayOut = { hit.position, dir };
        absorption = color;
        light = emissionColor * emissionStrength;
        return true;
    }
};