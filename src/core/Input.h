#pragma once

#include <glm/glm.hpp>

class App;

class Input
{
public:
    Input(const App& app) : m_App(app) {};

    bool IsKeyDown(int keycode) const;
    bool IsMouseButtonDown(int button) const;
    glm::vec2 GetMousePosition() const;
    void SetCursorMode(int mode) const;

private:
    const App& m_App;
};