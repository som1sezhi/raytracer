#pragma once

#include <string>

#include <glad/glad.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <imgui_stdlib.h>
#define GL_SILENCE_DEPRECATION
#include <GLFW/glfw3.h> // Will drag system OpenGL headers

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
	void Run();
	virtual void Update() {}
	virtual void RenderUI() {}

private:
	GLFWwindow *m_Window;
};