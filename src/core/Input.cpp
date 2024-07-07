#include "Input.h"

/* Adapted from Walnut by TheCherno under the MIT License.

    Copyright (c) 2022 Studio Cherno

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
*/

#include "App.h"
#include <GLFW/glfw3.h>

bool Input::IsKeyDown(int keycode) const
{
    GLFWwindow* window = m_App.GetWindow();
    int state = glfwGetKey(window, keycode);
    return state == GLFW_PRESS || state == GLFW_REPEAT;
}

bool Input::IsMouseButtonDown(int button) const
{
    GLFWwindow* window = m_App.GetWindow();
    int state = glfwGetMouseButton(window, button);
    return state == GLFW_PRESS;
}

glm::vec2 Input::GetMousePosition() const
{
    GLFWwindow* window = m_App.GetWindow();
    double x, y;
    glfwGetCursorPos(window, &x, &y);
    return { (float)x, (float)y };
}

void Input::SetCursorMode(int mode) const
{
    GLFWwindow* window = m_App.GetWindow();
    glfwSetInputMode(window, GLFW_CURSOR, mode);
}
