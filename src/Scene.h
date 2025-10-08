#pragma once

#include <vector>
#include <glm/glm.hpp>
#include <cuda_runtime.h>

#include "Material.h"
#include "Interval.h"

struct Sphere
{
    glm::vec3 center{ 0.0f };
    float radius = 0.5f;
    Material material;

    __host__ __device__
    HitInfo Intersect(const Ray& ray, Interval rayT) const
    {
        glm::vec3 movedOrigin = ray.origin - center;

        float a = glm::dot(ray.dir, ray.dir);
        float b = 2.0f * glm::dot(movedOrigin, ray.dir);
        float c = glm::dot(movedOrigin, movedOrigin) - radius * radius;

        float discriminant = b * b - 4.0f * a * c;
        if (discriminant < 0.0f)
            return HitInfo{};
        
        float root = (-b - glm::sqrt(discriminant)) / (2.0f * a);
        if (!rayT.Surrounds(root))
        {
            root = (-b + glm::sqrt(discriminant)) / (2.0f * a);
            if (!rayT.Surrounds(root))
                return HitInfo{};
        }

        HitInfo hit;
        hit.dist = root;
        hit.position = ray.At(hit.dist);
        glm::vec3 normal = (hit.position - center) / radius;
        hit.SetFaceNormal(ray, normal);
        hit.material = &material;
        
        return hit;
    }
};

struct Scene
{
    std::vector<Sphere> spheres;
};