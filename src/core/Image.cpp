#include "Image.h"

Image::Image(uint32_t width, uint32_t height, void* data)
	: m_Width(width), m_Height(height)
{
	glGenTextures(1, &m_ID);
	glBindTexture(GL_TEXTURE_2D, m_ID);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

	SetData(data);
}

void Image::SetData(void* data)
{
	glBindTexture(GL_TEXTURE_2D, m_ID);
	glTexImage2D(
		GL_TEXTURE_2D, 0, GL_RGBA8, m_Width, m_Height, 0,
		GL_RGBA, GL_UNSIGNED_BYTE, data
	);
}

void Image::Resize(uint32_t width, uint32_t height)
{
	if (width == m_Width && height == m_Height)
		return;
	m_Width = width;
	m_Height = height;
	SetData(nullptr);
}

void Image::Release()
{
	glDeleteTextures(1, &m_ID);
	m_ID = 0;
}
