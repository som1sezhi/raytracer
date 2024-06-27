#pragma once

#include <stdint.h>
#include <glm/glm.hpp>
#include <cuda_runtime.h>

#include "Camera.h"

// A struct of rendering parameters to be uploaded to the GPU.
struct RenderParams
{
	cudaSurfaceObject_t Surface;
	Camera Camera;
	uint32_t Width;
	uint32_t Height;
};

void traceRays(RenderParams& renderParams);