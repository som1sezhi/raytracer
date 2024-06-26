#pragma once

#include <cuda_runtime.h>
#include <iostream>

#define CU_CHECK(val) checkCuda( (val), #val, __FILE__, __LINE__ )

void checkCuda(cudaError_t result, char const* const func, const char* const file, int const line);