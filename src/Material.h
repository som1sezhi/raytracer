#pragma once

#include <glm/glm.hpp>

struct Ray;
struct HitInfo;

struct Material
{
	glm::vec3 color;

	bool ScatterRay const(
		const Ray& rayIn,
		const HitInfo& hit,
		glm::vec3 hitColor,
		Ray& rayOut
	)
	{
		return false;
	}
};