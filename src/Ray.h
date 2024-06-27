#pragma once

#include <glm/glm.hpp>
#include <cuda_runtime.h>

struct Ray
{
	glm::vec3 Origin;
	glm::vec3 Dir;

	__device__ Ray(const glm::vec3& orig, const glm::vec3& dir) : Origin(orig), Dir(dir) {};

	__device__ glm::vec3 At(float t) const { return Origin + t * Dir; }
};