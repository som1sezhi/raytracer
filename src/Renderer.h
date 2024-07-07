#pragma once

#include <memory>
#include "core/Image.h"
#include "Camera.h"
#include "Scene.h"

struct RenderSettings
{
    int bounceLimit = 1;
    glm::vec3 skyColor1;
    glm::vec3 skyColor2;
};

class Renderer
{
public:
    virtual ~Renderer() {};

    virtual void OnResize(uint32_t width, uint32_t height) = 0;
    virtual void Render(Scene& scene, Camera& camera, const RenderSettings& settings) = 0;
    virtual Image* GetImage() = 0;
    uint32_t GetCurNumSamples() const { return m_CurNumSamples; }
    void ResetCurNumSamples() { m_CurNumSamples = 0; }

protected:
    uint32_t m_CurNumSamples = 0;
};