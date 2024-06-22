#pragma once

#include <memory>
#include "core/Image.h"

class CPURenderer
{
public:
	CPURenderer() = default;

	void OnResize(uint32_t width, uint32_t height);
	void Render();
	Image* GetImage() { return m_Image.get(); }
private:
	std::unique_ptr<Image> m_Image;
	uint32_t* m_ImageData = nullptr;
};