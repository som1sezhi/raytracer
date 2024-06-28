#pragma once

#include "core/App.h"
#include "core/Image.h"
#include "CPURenderer.h"
#include "GPURenderer.h"
#include "Camera.h"
#include "Scene.h"

class RayTracerApp : public App
{
public:
	RayTracerApp(const AppSpec &spec);
	virtual void Update() override;
	virtual void RenderUI() override;

private:
	CPURenderer m_CPURenderer;
	GPURenderer m_GPURenderer;
	int m_CurrRendererIdx = 1;

	uint32_t m_ViewportWidth = 0, m_ViewportHeight = 0;
	glm::vec2 m_LastMousePos{ 0, 0 };

	Scene m_Scene;
	Camera m_Camera;
	bool m_CameraControlsActive = false;
	float m_CameraMovementSpeed = 5.0f;
	float m_CameraRotationSpeed = 0.9f;

	bool m_ShowDebugInfoWindow = true;
	bool m_ShowImGuiDemoWindow = false;
	bool m_ShowOptionsWindow = true;

	// Debug info
	bool m_WantCaptureMouse = false;
	bool m_WantCaptureKeyboard = false;
};