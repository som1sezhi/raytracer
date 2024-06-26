#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

void traceRays(cudaSurfaceObject_t surfObj, uint32_t width, uint32_t height);