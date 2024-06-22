#pragma once

#include "core/App.h"
#include "core/Image.h"
#include "CPURenderer.h"

class RayTracerApp : public App
{
public:
	RayTracerApp(const AppSpec &spec);
	virtual void Update() override;
	virtual void RenderUI() override;
private:
	// Our state
	bool show_demo_window = true;
	bool show_another_window = true;
	CPURenderer m_CPURenderer;
	uint32_t m_ViewportWidth = 0, m_ViewportHeight = 0;
};