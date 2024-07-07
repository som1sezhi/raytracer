#pragma once

#include <memory>
#include "core/Image.h"
#include "Renderer.h"
#include "Ray.h"

class CPURenderer : public Renderer
{
public:
    CPURenderer() = default;

    virtual void OnResize(uint32_t width, uint32_t height) override;
    virtual void Render(Scene& scene, Camera& camera, const RenderSettings& settings) override;
    virtual Image* GetImage() override { return m_Image.get(); }
private:
    std::unique_ptr<Image> m_Image;
    float* m_ImageData = nullptr;
};