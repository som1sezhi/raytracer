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

void CPURenderer::Render()
{
	if (!m_Image)
		return;

	uint32_t width = m_Image->GetWidth();
	uint32_t height = m_Image->GetHeight();
	for (uint32_t y = 0; y < height; y++)
	{
		for (uint32_t x = 0; x < width; x++)
		{
			uint32_t i = x + y * width;
			uint8_t r = (uint8_t)((float)x / width * 255.9);
			uint8_t g = (uint8_t)((float)y / height * 255.9);
			//float val = (float)i / (width * height);
			//uint8_t pxVal = (uint8_t)(val * 255.0f);
			//m_ImageData[i] = 0xff000000 | (pxVal << 16) | (pxVal << 8) | pxVal;
			m_ImageData[i] = 0xff000000 | (g << 8) | r;
		}
	}
	m_Image->SetData(m_ImageData);
}