#pragma once

#include <glm/glm.hpp>
#include <cuda_runtime.h>

struct Material;

struct HitInfo
{
	glm::vec3 position;
	float dist = FLT_MAX;
	glm::vec3 normal;
	const Material* material = nullptr;

	__host__ __device__ bool DidHit() const { return dist < FLT_MAX; }
};