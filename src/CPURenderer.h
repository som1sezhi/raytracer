#pragma once

#include <memory>
#include "core/Image.h"
#include "Renderer.h"

class CPURenderer : public Renderer
{
public:
	CPURenderer() = default;

	virtual void OnResize(uint32_t width, uint32_t height) override;
	virtual void Render(Camera& camera) override;
	virtual Image* GetImage() override { return m_Image.get(); }
private:
	glm::vec4 TraceRay(Camera& camera);
	std::unique_ptr<Image> m_Image;
	uint32_t* m_ImageData = nullptr;
};