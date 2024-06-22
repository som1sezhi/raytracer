cmake_minimum_required(VERSION 3.20)

project(raytracer LANGUAGES CXX C)

# dependencies
find_package(OpenGL REQUIRED)
add_subdirectory(extern)

add_library(core STATIC
    src/core/App.cpp
    src/core/App.h
    src/core/Image.cpp
)
target_link_libraries(core PUBLIC
    ${OPENGL_gl_LIBRARY}
    glad
    glfw
    imgui
)

add_executable(raytracer
    src/main.cpp
    src/RayTracerApp.cpp
    src/RayTracerApp.h
    src/CPURenderer.cpp
    src/CPURenderer.h
)
target_link_libraries(raytracer PRIVATE
    core
)