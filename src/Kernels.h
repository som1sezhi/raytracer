#pragma once

#include <stdint.h>
#include <glm/glm.hpp>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "Camera.h"
#include "Scene.h"
#include "TracingRoutines.h"

// A struct of rendering parameters and info to be uploaded to the GPU.
struct RenderKernelParams
{
	// Actual rendering parameters
	RenderParams renderParams;

	// Image to read from/render to
	cudaSurfaceObject_t surface;

	// Viewport dimensions
	uint32_t width;
	uint32_t height;

	uint32_t curNumSamples;		// How many samples we accumulated so far
};

void render(RenderKernelParams& kernelParams);