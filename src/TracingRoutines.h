#pragma once

#include <cuda_runtime.h>
#include "Ray.h"
#include "Scene.h"
#include "Camera.h"
#include "Renderer.h"

struct RenderParams
{
    Sphere* spheres;
    size_t spheresCount;

    Camera camera;

    RenderSettings settings;
};

__host__ __device__
glm::vec3 getRayColor(const Ray& ray, RenderParams& params, uint32_t& seed);