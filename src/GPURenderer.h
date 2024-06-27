#pragma once

#include <memory>

#include "core/Image.h"
#include "Renderer.h"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

class GPURenderer : public Renderer
{
public:
	GPURenderer() = default;

	virtual void OnResize(uint32_t width, uint32_t height) override;
	virtual void Render(Camera& camera) override;
	virtual Image* GetImage() override { return m_Image.get(); }
private:
	std::unique_ptr<Image> m_Image;
	cudaGraphicsResource_t m_ImageCudaResource = nullptr;
};