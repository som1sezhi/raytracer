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
{
	RecalcViewMatrix();
	// Since the camera initializes with viewport width/height of 0, this will
	// create a nonsensical projection matrix if we calculate it now.
	// Wait until the camera is given the viewport dimensions via OnResize()
	// before calculating the projection matrix.
}

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

void Camera::RecalcMatrices()
{
	m_CachedRaysNeedRecalc = m_CachedRaysNeedRecalc || m_ViewNeedsRecalc || m_ProjectionNeedsRecalc;
	RecalcViewMatrix();
	RecalcProjectionMatrix();
}

void Camera::RecalcViewMatrix()
{
	if (!m_ViewNeedsRecalc)
		return;

	m_ViewMatrix = glm::lookAt(
		m_Position, m_Position + m_ForwardDir, glm::vec3(0, 1, 0)
	);
	m_InvViewMatrix = glm::inverse(m_ViewMatrix);

	m_ViewNeedsRecalc = false;
}

void Camera::RecalcProjectionMatrix()
{
	if (!m_ProjectionNeedsRecalc)
		return;

	m_ProjectionMatrix = glm::perspectiveFov(
		glm::radians(m_VerticalFOV),
		(float)m_ViewportWidth, (float)m_ViewportHeight,
		m_NearClip, m_FarClip
	);
	m_InvProjectionMatrix = glm::inverse(m_ProjectionMatrix);

	m_ProjectionNeedsRecalc = false;
}

void Camera::RecalcRayDirs()
{
	if (!m_CachedRaysNeedRecalc)
		return;

	m_CachedRayDirs.resize(m_ViewportWidth * m_ViewportHeight);

	for (uint32_t y = 0; y < m_ViewportHeight; y++)
	{
		for (uint32_t x = 0; x < m_ViewportWidth; x++)
		{
			m_CachedRayDirs[x + y * m_ViewportWidth] = CalcRayDir(x, y);
		}
	}

	m_CachedRaysNeedRecalc = false;
}
