#pragma once

#include <string>

#include <glad/glad.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <imgui_stdlib.h>
#define GL_SILENCE_DEPRECATION
#include <GLFW/glfw3.h> // Will drag system OpenGL headers

#include "Input.h"

struct AppSpec
{
	std::string Title = "App";
	uint32_t Width = 1280;
	uint32_t Height = 720;
};

class App
{
public:
	App(const AppSpec &spec);
	~App();

	// Start the app's main loop.
	void Run();

	// Called after polling for events and before Dear ImGui begins a new frame.
	// Intended for responding to keyboard/mouse events.
	virtual void Update() {}

	// Render the GUI.
	virtual void RenderUI() {}

	GLFWwindow* GetWindow() const { return m_Window; }

protected:
	const Input m_Input;

private:
	GLFWwindow *m_Window;
};