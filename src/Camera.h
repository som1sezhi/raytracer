#pragma once

#include <stdint.h>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <cuda_runtime.h>

#include "Ray.h"

class Camera
{
public:
	Camera(float verticalFOV, float nearClip, float farClip);

	void OnResize(uint32_t width, uint32_t height);

	void Move(const glm::vec3& delta);
	void Rotate(const glm::quat& rotation);

	const glm::vec3& GetPosition() const { return m_Position; }
	const glm::vec3& GetForwardDir() const { return m_ForwardDir; }
	const glm::mat4& GetInvViewMatrix() const { return m_InvViewMatrix; }
	const glm::mat4& GetInvProjectionMatrix() const { return m_InvProjectionMatrix; }
	const glm::vec3* GetRayDirs() { return m_CachedRayDirs; }

	__device__ Ray GetRay(uint32_t x, uint32_t y) const
	{
		return { m_Position, CalcRayDir(x, y) };
	}

	bool RecalcMatrices();
	bool RecalcRayDirs();

private:
	__host__ __device__ glm::vec3 CalcRayDir(uint32_t x, uint32_t y) const
	{
		glm::vec2 clipCoord =
			(glm::vec2(x, y) + 0.5f) / glm::vec2(m_ViewportWidth, m_ViewportHeight);
		clipCoord = clipCoord * 2.0f - 1.0f;

		glm::vec4 target = m_InvProjectionMatrix * glm::vec4(clipCoord, 1, 1);
		glm::vec3 rayDir = glm::vec3(
			m_InvViewMatrix * glm::vec4(
				glm::normalize(glm::vec3(target) / target.w),
				0
			)
		);
		return rayDir;
	}

	bool RecalcViewMatrix();
	bool RecalcProjectionMatrix();

	glm::vec3 m_Position{ 0.0f, 0.0f, 0.0f };
	glm::vec3 m_ForwardDir{ 0.0f, 0.0f, -1.0f };
	float m_VerticalFOV = 45.0f;
	float m_NearClip = 0.1f;
	float m_FarClip = 100.0f;

	glm::mat4 m_ViewMatrix{ 1.0f };
	glm::mat4 m_ProjectionMatrix{ 1.0f };
	glm::mat4 m_InvViewMatrix{ 1.0f };
	glm::mat4 m_InvProjectionMatrix{ 1.0f };

	glm::vec3* m_CachedRayDirs = nullptr;
	size_t m_CachedRayDirsSize = 0;

	uint32_t m_ViewportWidth = 0, m_ViewportHeight = 0;

	bool m_ViewNeedsRecalc = true;
	bool m_ProjectionNeedsRecalc = true;
	bool m_CachedRaysNeedRecalc = true;
};