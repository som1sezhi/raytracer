#include "RayTracerApp.h"

#include <iostream>
#include "core/Timer.h"

RayTracerApp::RayTracerApp(const AppSpec &spec)
	: App(spec),
	show_demo_window(true), show_another_window(true)
{}

void RayTracerApp::Update()
{
    
}

void RayTracerApp::RenderUI()
{
    ImGuiIO& io = ImGui::GetIO();

    m_Renderer.OnResize(m_ViewportWidth, m_ViewportHeight);

    Timer timer;
    m_Renderer.Render();
    float renderTimeMs = timer.GetElapsedSecs() * 1000.0f;

    ImGui::DockSpaceOverViewport(0, ImGui::GetMainViewport());

    if (ImGui::BeginMainMenuBar())
    {
        if (ImGui::BeginMenu("File"))
        {
            if (ImGui::MenuItem("(test)")) {}
            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();
    }

    // 1. Show the big demo window (Most of the sample code is in ImGui::ShowDemoWindow()! You can browse its code to learn more about Dear ImGui!).
        if (show_demo_window)
            ImGui::ShowDemoWindow(&show_demo_window);

    // 2. Show a simple window that we create ourselves. We use a Begin/End pair to create a named window.
    {
        static float f = 0.0f;
        static int counter = 0;

        ImGui::Begin("Hello, world!");                          // Create a window called "Hello, world!" and append into it.

        ImGui::Text("This is some useful text.");               // Display some text (you can use a format strings too)
        ImGui::Checkbox("Demo Window", &show_demo_window);      // Edit bools storing our window open/close state
        ImGui::Checkbox("Another Window", &show_another_window);

        ImGui::SliderFloat("float", &f, 0.0f, 1.0f);            // Edit 1 float using a slider from 0.0f to 1.0f

        if (ImGui::Button("Button"))                            // Buttons return true when clicked (most widgets return true when edited/activated)
            counter++;
        ImGui::SameLine();
        ImGui::Text("counter = %d", counter);

        ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate);
        ImGui::End();
    }

    // 3. Show another simple window.
    if (show_another_window)
    {
        ImGui::Begin("Another Window", &show_another_window);   // Pass a pointer to our bool variable (the window will have a closing button that will clear the bool when clicked)
        ImGui::Text("Hello from another window!");
        if (ImGui::Button("Close Me"))
            show_another_window = false;
        ImGui::Text("Render time: %.3f ms", renderTimeMs);
        ImGui::End();
    }

    // ================ Viewport window ================

    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
    ImGui::Begin("Viewport");

    m_ViewportWidth = ImGui::GetContentRegionAvail().x;
    m_ViewportHeight = ImGui::GetContentRegionAvail().y;
        
    Image* image = m_Renderer.GetImage();
    if (image)
    {
        ImGui::Image(
            (void*)(intptr_t)image->GetID(),
            { (float)m_ViewportWidth, (float)m_ViewportHeight },
            {0, 1}, {1,0}
        );
    }
    ImGui::End();
    ImGui::PopStyleVar();
}
