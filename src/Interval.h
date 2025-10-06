#pragma once

#include <float.h>
#include <cuda_runtime.h>

class Interval
{
public:
    float min, max;

    // Create an empty interval
    __host__ __device__ Interval() : min(FLT_MAX), max(-FLT_MAX) {}

    __host__ __device__ Interval(float min, float max) : min(min), max(max) {}

    __host__ __device__ static Interval Universe() { return { -FLT_MAX, FLT_MAX }; }

    __host__ __device__ float Size() const { return max - min; }

    // Is x contained in [min, max]?
    __host__ __device__ bool Contains(float x) const {
        return min <= x && x <= max;
    }

    // Is x contained in (min, max)?
    __host__ __device__ bool Surrounds(float x) const {
        return min < x && x < max;
    }

    __host__ __device__ float Clamp(float x) const {
        return x < min ? min : x > max ? max : x;
    }
};
