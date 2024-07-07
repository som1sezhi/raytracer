#pragma once

#include <random>
#include <glm/glm.hpp>

__host__ __device__
inline bool nearZero(const glm::vec3& v)
{
    constexpr float eps = 1e-10f;
    glm::vec3 absV = glm::abs(v);
    return absV.x < eps && absV.y < eps && absV.z < eps;
}

// Convert a gamma-transformed color to a linear color for gamma = 2.
__host__ __device__
inline glm::vec3 gammaToLinear(const glm::vec3& color)
{
    return color * color;
}

// Convert a linear color to a gamma-transformed color for gamma = 2.
__host__ __device__
inline glm::vec3 linearToGamma(const glm::vec3& color)
{
    glm::vec3 c = glm::max(color, glm::vec3(0.0));
    return glm::sqrt(c);
}

__host__ __device__
inline uint32_t pcgHash(uint32_t input)
{
    uint32_t state = input * 747796405u + 2891336453u;
    uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Generate a random value in the interval [0, 1).
__host__ __device__
inline float randomFloat(uint32_t& seed)
{
    seed = pcgHash(seed);
    float x = (float)seed / (float)UINT32_MAX;
    return x == 1.0f ? 0.0f : x;
}

__host__ __device__
inline glm::vec3 randomVec(uint32_t& seed)
{
    return {
        randomFloat(seed),
        randomFloat(seed),
        randomFloat(seed)
    };
}

__host__ __device__
inline float randomNormalDist(uint32_t& seed)
{
    // https://stackoverflow.com/a/6178290 ported to C++
    float theta = 6.28318530718f * randomFloat(seed);
    float rho = glm::sqrt(-2.0f * glm::log(1.0f - randomFloat(seed)));
    return rho * glm::cos(theta);
}

__host__ __device__
inline glm::vec3 randomUnitVec(uint32_t& seed)
{
#ifdef __CUDA_ARCH__
    glm::vec3 v = {
        randomNormalDist(seed),
        randomNormalDist(seed),
        randomNormalDist(seed)
    };
#else
    glm::vec3 v = randomVec(seed) * 2.0f - 1.0f;
    while (glm::dot(v, v) > 1.0f)
        v = randomVec(seed) * 2.0f - 1.0f;
#endif
    
    if (v == glm::vec3(0.0f))
        return { 1.0f, 0.0f ,0.0f };
    return glm::normalize(v);
}

__host__ __device__
inline glm::vec3 randomOnHemisphere(const glm::vec3& normal, uint32_t& seed)
{
    glm::vec3 v = randomUnitVec(seed);
    if (glm::dot(v, normal) > 0.0f)
        return v;
    return -v;
}