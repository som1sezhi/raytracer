#pragma once

#include <glm/glm.hpp>
#include <cuda_runtime.h>

#include "Utils.h"
#include "HitInfo.h"
#include "Ray.h"

struct Material
{
	glm::vec3 color;

	__host__ __device__
	bool ScatterRay(
		const Ray& rayIn,
		const HitInfo& hit,
		glm::vec3& hitColor,
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
		hitColor *= color;
		return true;
	}
};