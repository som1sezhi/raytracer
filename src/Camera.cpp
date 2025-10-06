#include "Camera.h"
#include "glm/gtc/matrix_access.hpp"

/* Adapted from TheCherno's Ray Tracing series under the MIT License.

    Copyright (c) 2022 Studio Cherno

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
*/

Camera::Camera(float verticalFOV, float nearClip, float farClip)
    : m_VerticalFOV(verticalFOV), m_NearClip(nearClip), m_FarClip(farClip)
{}

void Camera::OnResize(uint32_t width, uint32_t height)
{
    if (width == m_ViewportWidth && height == m_ViewportHeight)
        return;

    m_ViewportWidth = width;
    m_ViewportHeight = height;

    m_ProjectionNeedsRecalc = true;
}

void Camera::Move(const glm::vec3& delta)
{
    m_Position += delta;
    m_ViewNeedsRecalc = true;
}

void Camera::Rotate(const glm::quat& rotation)
{
    // Re-normalize forward vector to stave off floating-point
    // error accumulation
    m_ForwardDir = glm::normalize(rotation * m_ForwardDir);
    m_ViewNeedsRecalc = true;
}

glm::vec3 Camera::GetRayDir(
    uint32_t x, uint32_t y, bool antialias, uint32_t& seed
) const
{
    const glm::vec3& cachedDir = m_CachedRayDirs[x + y * m_ViewportWidth];
    if (antialias)
    {
        // Nudge precomputed ray dir

        // Size of projection of cached ray dir on forward vector
        float zDist = glm::dot(cachedDir, m_ForwardDir);

        float pxSize = m_PixelScale * zDist;

        return cachedDir + pxSize * (
            (randomFloat(seed) - 0.5f) * m_RightDir
            + (randomFloat(seed) - 0.5f) * m_UpDir
        );
    }
    else
        // Return precomputed ray dir
        return cachedDir;
}

bool Camera::RecalcMatrices()
{
    m_CachedRaysNeedRecalc = m_CachedRaysNeedRecalc || m_ViewNeedsRecalc || m_ProjectionNeedsRecalc;
    bool didRecalc = RecalcViewMatrix();
    didRecalc |= RecalcProjectionMatrix();
    return didRecalc;
}

bool Camera::RecalcViewMatrix()
{
    if (!m_ViewNeedsRecalc)
        return false;

    m_ViewMatrix = glm::lookAt(
        m_Position, m_Position + m_ForwardDir, glm::vec3(0, 1, 0)
    );
    m_InvViewMatrix = glm::inverse(m_ViewMatrix);

    m_RightDir = glm::column(m_InvViewMatrix, 0);
    m_UpDir = glm::column(m_InvViewMatrix, 1);

    m_ViewNeedsRecalc = false;
    return true;
}

bool Camera::RecalcProjectionMatrix()
{
    if (!m_ProjectionNeedsRecalc)
        return false;

    m_ProjectionMatrix = glm::perspectiveFov(
        glm::radians(m_VerticalFOV),
        (float)m_ViewportWidth, (float)m_ViewportHeight,
        m_NearClip, m_FarClip
    );
    m_InvProjectionMatrix = glm::inverse(m_ProjectionMatrix);

    // Height of viewport plane 1 unit in front of the camera
    float viewH = 2.0f * glm::tan(0.5f * glm::radians(m_VerticalFOV));
    m_PixelScale = viewH / (float)m_ViewportHeight;

    m_ProjectionNeedsRecalc = false;
    return true;
}

bool Camera::RecalcRayDirs()
{
    if (!m_CachedRaysNeedRecalc)
        return false;

    // Resize the cache array if the viewport size changed.
    // We use an array instead of an std::vector to avoid copying the vector's
    // contents when copying the Camera to the GPU.
    if (m_ViewportWidth * m_ViewportHeight != m_CachedRayDirsSize)
    {
        delete[] m_CachedRayDirs;
        m_CachedRayDirs = new glm::vec3[m_ViewportWidth * m_ViewportHeight];
        m_CachedRayDirsSize = m_ViewportWidth * m_ViewportHeight;
    }	

    for (uint32_t y = 0; y < m_ViewportHeight; y++)
    {
        for (uint32_t x = 0; x < m_ViewportWidth; x++)
        {
            m_CachedRayDirs[x + y * m_ViewportWidth] = CalcRayDir(x, y);
        }
    }

    m_CachedRaysNeedRecalc = false;
    return true;
}
