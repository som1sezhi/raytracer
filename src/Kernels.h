#pragma once

#include <stdint.h>
#include <glm/glm.hpp>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "Camera.h"
#include "Scene.h"

// A struct of rendering parameters to be uploaded to the GPU.
struct RenderParams
{
	// Image to read from/render to
	cudaSurfaceObject_t surface;

	// Scene
	Sphere* spheres;
	size_t spheresCount;

	Camera camera;

	// Viewport dimensions
	uint32_t width;
	uint32_t height;

	curandState* randStates;	// Random states for each pixel
	uint32_t curNumSamples;		// How many samples we accumulated so far
};

void renderInit(curandState* states, uint32_t width, uint32_t height);
void render(RenderParams& renderParams);