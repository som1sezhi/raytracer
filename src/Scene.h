#pragma once

#include <vector>
#include <glm/glm.hpp>
#include <cuda_runtime.h>

#include "Ray.h"

struct HitInfo
{
	glm::vec3 position;
	float dist = FLT_MAX;
	glm::vec3 normal{ 0.0f };

	__host__ __device__ bool DidHit() const { return dist < FLT_MAX; }
};

struct Sphere
{
	glm::vec3 center{ 0.0f };
	float radius = 0.5f;

	__host__ __device__
	HitInfo Intersect(const Ray& ray, float minDist, float maxDist) const
	{
		glm::vec3 movedOrigin = ray.origin - center;

		float a = glm::dot(ray.dir, ray.dir);
		float b = 2.0f * glm::dot(movedOrigin, ray.dir);
		float c = glm::dot(movedOrigin, movedOrigin) - radius * radius;

		float discriminant = b * b - 4.0f * a * c;
		if (discriminant < 0.0f)
			return HitInfo{};
		
		float root = (-b - glm::sqrt(discriminant)) / (2.0f * a);
		if (root <= minDist || maxDist <= root)
		{
			root = (-b + glm::sqrt(discriminant)) / (2.0f * a);
			if (root <= minDist || maxDist <= root)
				return HitInfo{};
		}

		HitInfo hit;
		hit.dist = root;
		hit.position = ray.At(hit.dist);
		hit.normal = (hit.position - center) / radius;
		
		return hit;
	}
};

struct Scene
{
	std::vector<Sphere> spheres;
};