#pragma once

#include <glm/glm.hpp>
#include <cuda_runtime.h>
#include "Ray.h"

struct Material;

struct HitInfo
{
    glm::vec3 position;
    float dist = FLT_MAX;
    glm::vec3 normal;
    bool frontFace;
    const Material* material = nullptr;

    __host__ __device__
    void SetFaceNormal(const Ray& ray, const glm::vec3& outwardNormal)
    {
        frontFace = glm::dot(ray.dir, outwardNormal) < 0.0f;
        normal = frontFace ? outwardNormal : -outwardNormal;
    }

    __host__ __device__ bool DidHit() const { return dist < FLT_MAX; }
    
};