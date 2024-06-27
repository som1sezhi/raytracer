cmake_minimum_required(VERSION 3.24)

project(raytracer LANGUAGES CXX C CUDA)

# External dependencies  
find_package(OpenGL REQUIRED)
add_subdirectory(extern)

set(CMAKE_CXX_STANDARD 20)

# Core app library (stuff I could potentially reuse in future Dear ImGui projs)
add_library(core STATIC
    "src/core/App.cpp"
    "src/core/Image.cpp"
    "src/core/Timer.cpp"
    "src/core/Input.cpp"
)
target_link_libraries(core PUBLIC
    ${OPENGL_gl_LIBRARY}
    glad
    glfw
    imgui
    glm
)

add_executable(raytracer
    "src/Main.cpp"
    "src/RayTracerApp.cpp"
    "src/CPURenderer.cpp"
    "src/GPURenderer.cpp"
    "src/Kernels.cu"
    "src/ErrorCheck.cpp"
    "src/Camera.cpp"
)
target_link_libraries(raytracer PRIVATE
    core
)
target_include_directories(raytracer PRIVATE
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)
set_target_properties(raytracer PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "50"
)