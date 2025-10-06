#pragma once

#include <stdint.h>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <cuda_runtime.h>

#include "Ray.h"
#include "Utils.h"

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
    glm::vec3 GetRayDir(uint32_t x, uint32_t y, bool antialias, uint32_t& seed) const;

    __device__ Ray CalcRay(
        uint32_t x, uint32_t y, bool antialias, uint32_t& seed
    ) const
    {
        if (antialias)
        {
            float fx = static_cast<float>(x) + randomFloat(seed);
            float fy = static_cast<float>(y) + randomFloat(seed);
            return { m_Position, CalcRayDir(fx, fy) };
        }
        else
            return { m_Position, CalcRayDir(x, y) };
    }

    bool RecalcMatrices();
    bool RecalcRayDirs();

private:
    __host__ __device__ glm::vec3 CalcRayDir(float x, float y) const
    {
        glm::vec2 clipCoord =
            glm::vec2(x, y) / glm::vec2(m_ViewportWidth, m_ViewportHeight);
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

    __host__ __device__ glm::vec3 CalcRayDir(uint32_t x, uint32_t y) const
    {
        return CalcRayDir(
            static_cast<float>(x) + 0.5f, static_cast<float>(y) + 0.5f
        );
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

    glm::vec3 m_RightDir{ 1.0f, 0.0f, 0.0f };
    glm::vec3 m_UpDir{ 0.0f, 1.0f, 0.0f };
    // Size of a pixel in the viewport plane 1 unit in front of the camera
    float m_PixelScale;

    glm::vec3* m_CachedRayDirs = nullptr;
    size_t m_CachedRayDirsSize = 0;

    uint32_t m_ViewportWidth = 0, m_ViewportHeight = 0;

    bool m_ViewNeedsRecalc = true;
    bool m_ProjectionNeedsRecalc = true;
    bool m_CachedRaysNeedRecalc = true;
};