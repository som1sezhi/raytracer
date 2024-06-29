#include "CPURenderer.h"

#include <stdio.h>
#include "TracingRoutines.h"
#include "Utils.h"

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
		m_Image = std::make_unique<Image>(width, height, Image::Format::RGBA32F);
	}

	// If the image was just created/resized, we need to allocate space
	// for our image data
	delete[] m_ImageData;
	m_ImageData = new float[width * height * 4];
}

void CPURenderer::Render(Scene& scene, Camera& camera, const RenderSettings& settings)
{
	if (!m_Image)
		return;

	if (camera.RecalcMatrices())
		m_CurNumSamples = 0;
	camera.RecalcRayDirs();
	auto rayDirs = camera.GetRayDirs();

	uint32_t width = m_Image->GetWidth();
	uint32_t height = m_Image->GetHeight();
	Ray ray;
	ray.origin = camera.GetPosition();

	RenderParams params = {
		.spheres = scene.spheres.data(),
		.spheresCount = scene.spheres.size(),
		.camera = camera,
		.settings = settings
	};

	for (uint32_t y = 0; y < height; y++)
	{
		for (uint32_t x = 0; x < width; x++)
		{
			uint32_t i = x + y * width;
			ray.dir = rayDirs[i];

			uint32_t seed = i * (m_CurNumSamples + 1);

			glm::vec3 color = getRayColor(ray, params, seed);

			glm::vec3 old = *reinterpret_cast<glm::vec3*>(m_ImageData + 4 * i);

			//  Accumulate color
			old = gammaToLinear(old);
			color = ((float) m_CurNumSamples * old + color)
				/ ((float) m_CurNumSamples + 1);
			color = linearToGamma(color);

			*reinterpret_cast<glm::vec3*>(m_ImageData + 4 * i) = color;
			m_ImageData[4 * i + 3] = 1.0f;
		}
	}
	m_Image->SetData(m_ImageData);

	m_CurNumSamples++;
}