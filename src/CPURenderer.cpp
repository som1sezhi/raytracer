#include "CPURenderer.h"

#include <stdio.h>

void CPURenderer::OnResize(uint32_t width, uint32_t height)
{
	// On program startup, this method could be called with 0 width and height,
	// which should be ignored
	if (width == 0 || height == 0)
		return;

	if (m_Image)
	{
		// Don't do anything if dimensions didn't change
		if (width == m_Image->GetWidth() && height == m_Image->GetHeight())
			return;

		// NOTE: Resizing the texture every frame can cause some 
		// memory usage spikes, which isn't super ideal but can work for now
		m_Image->Resize(width, height);
	}
	// Create image if it doesn't exist yet
	else
	{
		m_Image = std::make_unique<Image>(width, height);
	}

	// If the image was just created/resized, we need to allocate space
	// for our image data
	delete[] m_ImageData;
	m_ImageData = new uint32_t[width * height];
}

void CPURenderer::Render(Scene& scene, Camera& camera)
{
	if (!m_Image)
		return;

	camera.RecalcMatrices();
	camera.RecalcRayDirs();
	auto rayDirs = camera.GetRayDirs();

	uint32_t width = m_Image->GetWidth();
	uint32_t height = m_Image->GetHeight();
	Ray ray;
	ray.origin = camera.GetPosition();
	for (uint32_t y = 0; y < height; y++)
	{
		for (uint32_t x = 0; x < width; x++)
		{
			uint32_t i = x + y * width;
			ray.dir = rayDirs[i];

			glm::vec4 color = TraceRay(scene, ray);

			color = glm::clamp(color, glm::vec4(0), glm::vec4(1));
			color *= 255.99f;
			uint8_t r = (uint8_t)(color.r);
			uint8_t g = (uint8_t)(color.g);
			uint8_t b = (uint8_t)(color.b);
			uint8_t a = (uint8_t)(color.a);
			
			m_ImageData[i] = (a << 24) | (b << 16) | (g << 8) | r;
		}
	}
	m_Image->SetData(m_ImageData);
}

glm::vec4 CPURenderer::TraceRay(Scene& scene, Ray& ray)
{
	HitInfo closestHit;
	for (const Sphere& sphere : scene.spheres)
	{
		HitInfo hit = sphere.Intersect(ray, 0.0f, closestHit.dist);

		if (hit.dist < closestHit.dist)
			closestHit = hit;
	}

	if (closestHit.DidHit())
	{
		return glm::vec4(0.5f * closestHit.normal + 0.5f, 1.0f);
	}
	else
		return glm::vec4(0, 0, 0, 1);
}
