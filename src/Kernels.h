#pragma once

#include <stdint.h>
#include <glm/glm.hpp>
#include <cuda_runtime.h>

#include "Camera.h"

// A struct of rendering parameters to be uploaded to the GPU.
struct RenderParams
{
	cudaSurfaceObject_t surface;
	Camera camera;
	uint32_t width;
	uint32_t height;
};

void traceRays(RenderParams& renderParams);