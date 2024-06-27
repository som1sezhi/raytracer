#include "RayTracerApp.h"

#include <iostream>
#include "core/Timer.h"

RayTracerApp::RayTracerApp(const AppSpec &spec)
	: App(spec),
    m_Camera(45, 0.1f, 100.0f)
{
    m_Camera.Move({ 0, 0, 3 });
    m_Scene.spheres.push_back(Sphere{});
}

void RayTracerApp::Update()
{
    ImGuiIO& io = ImGui::GetIO();
    m_WantCaptureMouse = io.WantCaptureMouse;
    m_WantCaptureKeyboard = io.WantCaptureKeyboard;
    float dt = io.DeltaTime;
    glm::vec2 curMousePos = m_Input.GetMousePosition();

    // ================ Camera controls ================

    // Begin enabling camera controls if LMB is held
    // AND the mouse is not captured by ImGui.
    // If camera controls were active last frame, only disable them if
    // RMB is released (so that if the mouse leaves the area for 1 frame it 
    // doesn't instantly disable the camera controls)
    m_CameraControlsActive =
        (!m_WantCaptureMouse || m_CameraControlsActive)
        && m_Input.IsMouseButtonDown(GLFW_MOUSE_BUTTON_2);

    if (!m_CameraControlsActive)
    {
        // Unlock the cursor
        m_Input.SetCursorMode(GLFW_CURSOR_NORMAL);
    }
    else
    {
        // Lock the cursor
        m_Input.SetCursorMode(GLFW_CURSOR_DISABLED);

        bool moved = false;
        
        // Directional movement
        const glm::vec3& forward = m_Camera.GetForwardDir();
        constexpr glm::vec3 up{ 0, 1, 0 };
        glm::vec3 right = glm::cross(forward, up);
        glm::vec3 delta{ 0, 0, 0 };
        if (m_Input.IsKeyDown(GLFW_KEY_W))
        {
            delta += forward * m_CameraMovementSpeed * dt;
            moved = true;
        }
        if (m_Input.IsKeyDown(GLFW_KEY_S))
        {
            delta -= forward * m_CameraMovementSpeed * dt;
            moved = true;
        }
        if (m_Input.IsKeyDown(GLFW_KEY_A))
        {
            delta -= right * m_CameraMovementSpeed * dt;
            moved = true;
        }
        if (m_Input.IsKeyDown(GLFW_KEY_D))
        {
            delta += right * m_CameraMovementSpeed * dt;
            moved = true;
        }
        if (m_Input.IsKeyDown(GLFW_KEY_Q))
        {
            delta -= up * m_CameraMovementSpeed * dt;
            moved = true;
        }
        if (m_Input.IsKeyDown(GLFW_KEY_E))
        {
            delta += up * m_CameraMovementSpeed * dt;
            moved = true;
        }
        if (moved)
            m_Camera.Move(delta);

        // Mouse rotation controls
        glm::vec2 mouseDelta = curMousePos - m_LastMousePos;
        if (mouseDelta.x != 0.0f || mouseDelta.y != 0.0f)
        {
            mouseDelta *= m_CameraRotationSpeed * -0.002f;

            glm::quat rotation = glm::normalize(
                glm::angleAxis(mouseDelta.y, right)
                * glm::angleAxis(mouseDelta.x, up)
            );
            m_Camera.Rotate(rotation);
        }
    }

    m_LastMousePos = curMousePos;
}

void RayTracerApp::RenderUI()
{
    static Renderer* renderers[2] = { &m_CPURenderer, &m_GPURenderer };
    Renderer* renderer = renderers[m_CurrRendererIdx];

    renderer->OnResize(m_ViewportWidth, m_ViewportHeight);
    m_Camera.OnResize(m_ViewportWidth, m_ViewportHeight);

    Timer timer;
    renderer->Render(m_Scene, m_Camera);
    float renderTimeMs = timer.GetElapsedSecs() * 1000.0f;

    ImGui::DockSpaceOverViewport(0, ImGui::GetMainViewport());

    if (ImGui::BeginMainMenuBar())
    {
        if (ImGui::BeginMenu("View"))
        {
            ImGui::MenuItem("Debug Info", "", &m_ShowDebugInfoWindow);
            ImGui::MenuItem("Options", "", &m_ShowOptionsWindow);
            ImGui::MenuItem("ImGui Demo Window", "", &m_ShowImGuiDemoWindow);
            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();
    }

    if (m_ShowImGuiDemoWindow)
        ImGui::ShowDemoWindow(&m_ShowImGuiDemoWindow);

    if (m_ShowDebugInfoWindow)
    {
        ImGuiIO& io = ImGui::GetIO();
        ImGui::Begin("Debug Info", &m_ShowDebugInfoWindow);
        ImGui::Text("Render time: %.3f ms", renderTimeMs);
        ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate);
        ImGui::Text("WantCaptureMouse: %d", m_WantCaptureMouse);
        ImGui::Text("WantCaptureKeyboard: %d", m_WantCaptureKeyboard);
        ImGui::End();
    }

    // ================ Viewport window ================
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
    ImGui::Begin("Viewport");

    m_ViewportWidth = ImGui::GetContentRegionAvail().x;
    m_ViewportHeight = ImGui::GetContentRegionAvail().y;
        
    Image* image = renderer->GetImage();
    if (image)
    {
        ImGui::Image(
            (void*)(intptr_t)image->GetID(),
            { (float)m_ViewportWidth, (float)m_ViewportHeight },
            {0, 1}, {1,0}
        );
        // Only enable camera controls if user is interacting with the viewport
        if (ImGui::IsItemHovered()) // if image if being hovered over
            ImGui::SetNextFrameWantCaptureMouse(false);
        if (ImGui::IsWindowFocused()) // if viewport window is focused
            ImGui::SetNextFrameWantCaptureKeyboard(false);
    }
    ImGui::End();
    ImGui::PopStyleVar();
    
    // ================ Options window ================
    if (m_ShowOptionsWindow)
    {
        ImGui::Begin("Options", &m_ShowOptionsWindow);

        ImGui::SeparatorText("Renderer");
        ImGui::Combo("Viewport renderer", &m_CurrRendererIdx, "CPU\0GPU\0\0");

        ImGui::SeparatorText("Camera");
        ImGui::DragFloat(
            "Movement speed", &m_CameraMovementSpeed,
            0.005f, 0.0f, +FLT_MAX
        );
        ImGui::DragFloat(
            "Rotation speed", &m_CameraRotationSpeed,
            0.005f, -FLT_MAX, +FLT_MAX
        );

        ImGui::End();
    }
}
