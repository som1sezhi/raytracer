#pragma once

#include <memory>
#include "core/Image.h"

class Renderer
{
public:
	virtual ~Renderer() {};

	virtual void OnResize(uint32_t width, uint32_t height) = 0;
	virtual void Render() = 0;
	virtual Image* GetImage() = 0;
};