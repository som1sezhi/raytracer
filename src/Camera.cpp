#include "Camera.h"

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
