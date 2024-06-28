#pragma once

#include <stdint.h>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

class Image
{
public:
	enum Format
	{
		RGBA8,
		RGBA32F
	};

	Image(uint32_t width, uint32_t height, Format format, void* data = nullptr);
	~Image() { Release(); }

	// Allocate memory and set the image's data.
	// If data is null, just allocate the memory for the image.
	void SetData(void* data);

	// Reallocate the image's memory to fit the given dimensions.
	void Resize(uint32_t width, uint32_t height);

	GLuint GetID() const { return m_ID; }
	GLuint GetWidth() const { return m_Width; }
	GLuint GetHeight() const { return m_Height; }

	// Forbid copying of OpenGL wrapper objects
	// https://www.khronos.org/opengl/wiki/Common_Mistakes#RAII_and_hidden_destructor_calls

	Image(const Image&) = delete;
	Image& operator=(const Image&) = delete;

	Image(Image&& other) noexcept
		: m_ID(other.m_ID), m_Width(other.m_Width), m_Height(other.m_Height), m_Format(other.m_Format)
	{
		other.m_ID = 0; // Use the "null" texture for the old object.
	}

	Image& operator=(Image&& other) noexcept
	{
		if (this != &other) // ALWAYS check for self-assignment
		{
			Release(); // set m_ID to 0
			m_ID = other.m_ID;
			m_Width = other.m_Width;
			m_Height = other.m_Height;
			m_Format = other.m_Format;
		}
		return *this;
	}

private:
	void Release();
	GLuint m_ID = 0;
	uint32_t m_Width, m_Height;
	Format m_Format;
};